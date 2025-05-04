import logging
import time
import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy
from .global_analyzer import PerformanceMetric, CommunicatorClique
from .dp_planner import get_time_array, solve_dp
from .pp_planner import apply_layer_split
from .cost_estimator import CostEstimator


class MitigationPlan(object):
    def __init__(self, redis_client):
        self.client = redis_client
        self.dp_version = 0
        self.estimator = CostEstimator()
        self.rs_cost = self.estimator.get_restart_adjustment_cost()

    def group_comp_time_by_dp(self,
                              all_cliques: Dict[int, CommunicatorClique],
                              comp_results: Dict[int, PerformanceMetric]):
        comp_results = deepcopy(comp_results)
        # 1st pass: Synchronize all results within the same PP group
        for c in all_cliques.values():
            if c.is_pp:
                ts = [comp_results[i].avg_lat for i in c.ranks]
                max_ts = max(ts)
                for r in c.ranks:
                    comp_results[r] = PerformanceMetric(
                        max_ts, max_ts, max_ts, 0
                    )
        # 2nd pass: Synchronize all results within the same TP group
        for c in all_cliques.values():
            if c.is_tp:
                ts = [comp_results[i].avg_lat for i in c.ranks]
                max_ts = max(ts)
                for r in c.ranks:
                    comp_results[r] = PerformanceMetric(
                        max_ts, max_ts, max_ts, 0
                    )
        # 3rd pass: aggregate to DP group
        dp_results = {}
        for c in all_cliques.values():
            if c.is_dp:
                for (dpr, r) in enumerate(c.ranks):
                    dp_results[dpr] = comp_results[r]
        logging.info(f"DP computation results: {dp_results}")
        return dp_results

    def parse_communication_results(self, comm_results: Dict[Tuple[int], List]):
        # Example of 8 GPU communication topos
        # TP: [0, 1], [2, 3], [4, 5], [6, 7]
        # DP: [0, 2], [1, 3], [4, 6], [5, 7]
        # PP: [0, 4], [1, 5], [2, 6], [3, 7]
        # comm_times = {0: t0, 1: t1, 4: t4, 5: t5}
        comm_times = {}
        for communicator_ranks, communicator_results in comm_results.items():
            comm_max_time = 0
            for task_result in communicator_results:
                # task_result: Dict[int(rank), PerformanceMetric]
                task_max_time = max([i.avg_lat for i in task_result.values()])
                comm_max_time = max(comm_max_time, task_max_time)
            comm_times[communicator_ranks[0]] = comm_max_time
        logging.info(f"Max times of each communicator: {comm_times}")
        tp_ranks = self.client.get("0_tp").decode()
        tp_degree = len(tp_ranks.split("_"))
        logging.info(f"TP degree: {tp_degree}")
        comm_items = sorted(list(comm_times.items()))
        max_comm_time_per_pp_stage = []
        for i in range(len(comm_items)):
            max_comm_time_per_pp_stage.append(
                max([comm_items[j][1] for j in range(i, i + tp_degree)])
            )
        logging.info(f"max_comm_time_per_pp_stage: {max_comm_time_per_pp_stage}")
        return max_comm_time_per_pp_stage

    def adjust_batchsize_distribution(self,
                                      comp_results: Dict[int, PerformanceMetric],
                                      cliques: Dict[int, CommunicatorClique]):
        comp_results_by_dp = self.group_comp_time_by_dp(cliques, comp_results)
        time_array = get_time_array(self.client, comp_results_by_dp)
        # Mitigation by adjusting batch size distribution across DP groups
        micro_bsz = self.client.get("micro_batch_size")
        global_bsz = self.client.get("global_batch_size")
        if micro_bsz is not None and global_bsz is not None:
            micro_bsz = int(micro_bsz.decode())
            global_bsz = int(global_bsz.decode())
            new_dp = solve_dp(time_array, micro_bsz, global_bsz)
            # microbatch changed
            self.dp_version += 1
            self.client.set('batch_distribution', str(new_dp))
            self.client.set("dp_version", self.dp_version)
        else:
            logging.warning(f"batch size not found: microbsz={micro_bsz}, globalbsz={global_bsz}")

    def adjust_pipeline_parallel(self,
                                 comm_cliques: Dict[int, CommunicatorClique],
                                 comm_tasks: List,
                                 comm_results: List[Dict[int, PerformanceMetric]]):
        max_comm_time_per_pp_stage = np.array(self.parse_communication_results(comm_results))
        max_comm_time_per_pp_stage = 1 / max_comm_time_per_pp_stage  # inverse of latency -> throughput
        max_comm_time_per_pp_stage = max_comm_time_per_pp_stage / np.sum(max_comm_time_per_pp_stage)
        layer_split = []
        total_num_layers = int(self.client.get("total_num_layers").decode())
        for i in range(len(max_comm_time_per_pp_stage) - 1):
            layer_split.append(round(total_num_layers * max_comm_time_per_pp_stage[i]))
        last_num_layers = total_num_layers - sum(layer_split)
        layer_split.append(last_num_layers)
        apply_layer_split(self.client, layer_split)
        # pause the training process and invoke restart from memory
        self.client.set("terminate_ctl", 123)

    def find_slow_reason(self,
                         comp_results: Dict[int, PerformanceMetric],
                         comm_results: Dict[Tuple[int], List]):
        comp_results = np.array([i.avg_lat for i in comp_results.values()])
        if np.max(comp_results) > 2 * np.median(comp_results):
            return "comp"
        comm_parsed = np.array(self.parse_communication_results(comm_results))
        if np.max(comm_parsed) > 1.1 * np.median(comm_parsed):
            return "comm"

    def mitigate_failslow(self,
                          dp_check_interval: int,
                          comp_results: Dict[int, PerformanceMetric],
                          comm_cliques: Dict[int, CommunicatorClique],
                          comm_tasks: Dict[Tuple[int], List],
                          comm_results: Dict[Tuple[int], List]):
        logging.info("Mitigating fail-slow problem...")
        slow_start = time.time()
        dp_adjusted, pp_adjusted = False, False
        slow_iter_time_init = float(self.client.get("cur_iter_time").decode())
        min_iter_time_init = float(self.client.get('min_iter_time').decode())
        reason = self.find_slow_reason(comp_results, comm_results)
        logging.info(f"Reason of fail-slow is {reason}")
        if slow_iter_time_init <= min_iter_time_init * 1.1:
            self.adjust_batchsize_distribution(comp_results, comm_cliques)
            return

        while True:
            # normal iteration time
            min_iter_time = float(self.client.get('min_iter_time').decode())
            # iteration time after fail-slow
            slow_iter_time = float(self.client.get("cur_iter_time").decode())
            dp_cost = self.estimator.get_dp_adjustment_cost(dp_check_interval//2, min_iter_time, slow_iter_time)
            pp_cost = self.estimator.get_pp_adjustment_cost()
            time_since_slow = 1000 * (time.time() - slow_start)
            logging.info(f"[Mitigation Plan] DPcost={dp_cost}, PPcost={pp_cost}, time_since_slow={time_since_slow}, min_iter={min_iter_time}, slow_iter={slow_iter_time}")
            if (reason != 'comm') and (time_since_slow >= dp_cost) and (not dp_adjusted):
                logging.info("[Mitigation Plan] Adjust DP")
                self.adjust_batchsize_distribution(comp_results, comm_cliques)
                dp_adjusted = True
            if (reason == 'comm') and (time_since_slow >= pp_cost) and (not pp_adjusted):
                logging.info("[Mitigation Plan] Adjust PP")
                self.adjust_pipeline_parallel(comm_cliques, comm_tasks, comm_results)
                pp_adjusted = True
            if time_since_slow >= 2 * pp_cost:
                break
            time.sleep(1)

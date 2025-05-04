import redis
import time
import struct
import logging
import numpy as np
from .communicator import Communicator
from typing import List, Dict


class PerformanceMetric(object):
    def __init__(self, min_lat, max_lat, avg_lat, std_lat) -> None:
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.avg_lat = avg_lat
        self.std_lat = std_lat

    @classmethod
    def from_bytes(cls, redis_data):
        data = struct.unpack('dddd', redis_data[:32])
        return PerformanceMetric(*data)
    
    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"(Perf: min={self.min_lat}, max={self.max_lat}, avg={self.avg_lat}, std={self.std_lat})"


class CommunicatorClique(object):
    def __init__(self, comms: List[Communicator], is_special: bool, is_tp: bool,
                 is_dp: bool, is_pp: bool, pp_stage=None) -> None:
        self.comms = comms
        self.ranks = sorted([i.global_rank for i in self.comms])
        self.is_special = is_special
        self.is_tp = is_tp
        self.is_dp = is_dp
        self.is_pp = is_pp
        self.pp_stage = pp_stage

    @property
    def clique_id(self):
        return tuple(self.ranks)

    @property
    def tag(self):
        if self.is_special:
            return f"special_{self.ranks}"
        if self.is_pp:
            return "pp"
        assert (self.is_dp or self.is_tp) and (self.pp_stage is not None)
        if self.is_dp:
            return f"ppstage{self.pp_stage}_dp"
        else:
            return f"ppstage{self.pp_stage}_tp"

    def __str__(self):
        return f"Clique(tag={self.tag}, ranks={self.ranks}, ncomms={len(self.comms)})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.tag + str(sorted(self.ranks)))

    @classmethod
    def build_clique(cls, comms: List[Communicator], tp: List, dp: List, pp: List):
        def find_ppstage(rep_rank):
            pp_stage = -1
            for ppg in pp:
                if rep_rank in ppg:
                    pp_stage = ppg.index(rep_rank)
                    break
            assert pp_stage != -1
            return pp_stage

        comm_ranks = sorted([i.global_rank for i in comms])
        for pp_group in pp:
            if comm_ranks == pp_group:
                logging.info(f"Build PP clique: {comm_ranks}!")
                return CommunicatorClique(comms, False, False, False, True, None)
        for tp_group in tp:
            if comm_ranks == tp_group:
                pp_stage = find_ppstage(comm_ranks[0])
                logging.info(f"Build TP clique: {comm_ranks}!")
                return CommunicatorClique(comms, False, True, False, False, pp_stage)
        for dp_group in dp:
            if comm_ranks == dp_group:
                pp_stage = find_ppstage(comm_ranks[0])
                logging.info(f"Build DP clique: {comm_ranks}!")
                return CommunicatorClique(comms, False, False, True, False, pp_stage)
        logging.warning(f"Can only build a special clique (not TP/DP/PP) for {comm_ranks}!")
        return CommunicatorClique(comms, True, False, False, False, None)


class GlobalAnalyzer(object):
    def __init__(self, storage: redis.StrictRedis):
        self.storage = storage
        self.world_size = None
        while self.world_size is None:
            self.world_size = self.storage.get("world_size")
            time.sleep(1)
        self.world_size = int(self.world_size.decode())
        self.tp, self.dp, self.pp = self.get_parallel_states()
        self.comms = None

    def build_comm_cliques(self, comms: List[Communicator]):
        idhash2clique: Dict[int, List[Communicator]] = {}
        for c in comms:
            if c.id_hash not in idhash2clique:
                idhash2clique[c.id_hash] = [c]
            else:
                idhash2clique[c.id_hash].append(c)
        addr2clique = {}
        added_clis = set()
        for comm_clique in idhash2clique.values():
            for c in comm_clique:
                cli = CommunicatorClique.build_clique(comm_clique, self.tp, self.dp, self.pp)
                if cli not in added_clis:
                    addr2clique[c.comm_addr] = cli
                    added_clis.add(cli.__hash__())
        return addr2clique
    
    def get_parallel_states(self):
        tp, dp, pp = [], [], []
        parse_rankstr = lambda s: [int(i) for i in s.split("_")]
        for rank in range(self.world_size):
            tp_r = parse_rankstr(self.storage.get(f"{rank}_tp").decode())
            dp_r = parse_rankstr(self.storage.get(f"{rank}_dp").decode())
            pp_r = parse_rankstr(self.storage.get(f"{rank}_pp").decode())
            tp.append(tuple(tp_r))
            dp.append(tuple(dp_r))
            pp.append(tuple(pp_r))
        # remove duplicates
        tp = list(set(tp))
        dp = list(set(dp))
        pp = list(set(pp))
        return [list(i) for i in tp], [list(i) for i in dp], [list(i) for i in pp]

    def wait_and_build_performance_map(self):
        comm_perfs = {}
        old_num_records = 0
        while True:
            logging.info(f"waiting for profiling results, current #res={len(comm_perfs)}")
            for key in self.storage.scan_iter("Perf_*"):
                key = key.decode()
                comm_addr = int(key.strip("Perf_"))  # the commnuicator address
                perf_str = self.storage.get(key).decode()
                perf_metrics = [float(i) for i in perf_str.split("_")]
                comm_perfs[comm_addr] = PerformanceMetric(*perf_metrics)
            # still empty
            if len(comm_perfs) == 0:
                time.sleep(1)
            # new records arrive, but may not contains all
            elif len(comm_perfs) != old_num_records:
                old_num_records = len(comm_perfs)
                time.sleep(1)
            # num records is not growing
            elif len(comm_perfs) == old_num_records and old_num_records != 0:
                break
            else:
                break

        perf_keys = self.storage.scan_iter("Perf_*")
        self.storage.delete(*perf_keys)
        return comm_perfs

    def find_slow_clique(self, perfs: Dict[int, PerformanceMetric],
                         cliques: Dict[int, CommunicatorClique]) -> List[CommunicatorClique]:
        '''
        ppstage0_tp 
        {
            (0, 1): [
                (94778436013328, (Perf: min=223819.0, max=226838.0, avg=224894.52631578947, std=929.7831874277297)),
                (94256251751520, (Perf: min=460153.0, max=466781.0, avg=464133.2631578947, std=1665.1690554954178))
            ],
            (2, 3): [
                (94443506796688, (Perf: min=130819.0, max=135620.0, avg=132527.57894736843, std=1377.0912256518493)),
                (94571181099680, (Perf: min=169943.0, max=176125.0, avg=172741.63157894736, std=1595.8498218001632))
            ]
        } 
        same_pos_comm_times = [(94256251751520, 464133.2631578947), (94571181099680, 172741.63157894736)]
        verify: 94256251751520
        '''
        group_perf = {}
        need_perf_cliques = []
        for (comm_addr, perf_metric) in perfs.items():
            c = cliques[comm_addr]
            if c.tag not in group_perf:
                group_perf[c.tag] = [(comm_addr, c.ranks, perf_metric)]
            else:
                group_perf[c.tag].append((comm_addr, c.ranks, perf_metric))

        for tag, same_pos_perfs in group_perf.items():
            group_by_clique = {}
            for item in same_pos_perfs:
                comm_addr, ranks, metrics = item
                ranks = tuple(ranks)
                if ranks not in group_by_clique:
                    group_by_clique[ranks] = [(comm_addr, metrics)]
                else:
                    group_by_clique[ranks].append((comm_addr, metrics))
            
            same_pos_comm_times = []
            for perf_info_list in group_by_clique.values():
                max_lat, max_addr = 0, None
                for i_addr, i_metric in perf_info_list:
                    if i_metric.avg_lat >= max_lat:
                        max_lat = i_metric.avg_lat
                        max_addr = i_addr
                same_pos_comm_times.append((max_addr, max_lat))
            # Only one same position clique, cannot determine if it is a slow one, so valid it
            if len(same_pos_comm_times) == 1:
                logging.info(
                    f"Only one clique {cliques[same_pos_comm_times[0][0]]} in same position, verify it!")
                need_perf_cliques.append(cliques[same_pos_comm_times[0][0]])
            # >=2 same-position cliques, if an clique slower than 20% of the median, valid it
            elif len(same_pos_comm_times) >= 2:
                median_time = np.median([i[1] for i in same_pos_comm_times])
                for (addr, t) in same_pos_comm_times:
                    if t >= 1.2 * median_time:
                        need_perf_cliques.append(cliques[addr])
                        logging.info(
                            f"Clique {cliques[addr]} is too slow (it={t}, median={median_time}), verify it!")
        return need_perf_cliques

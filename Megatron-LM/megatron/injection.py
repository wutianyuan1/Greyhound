import subprocess
import time
import sys
import socket
import os
import torch.distributed as dist
from megatron.dp_planner import PerformanceMetric, get_time_array, solve_dp


class FailSlowInjector(object):
    def __init__(self, injection_conf, redis_cli):
        self.init_time = time.time()
        self.comp_injection_info = []
        self.comm_injection_info = []
        with open(injection_conf, 'r') as f:
            content = f.read().split("\n")
        for line in content:
            if len(line) <= 1:
                continue
            if 'comm' not in line:
                start, end, global_ranks, delay_time = line.split(";")
                self.comp_injection_info.append([float(start), float(end), eval(global_ranks), float(delay_time)])
            else:
                _, start, end, pair = line.split(";")
                self.comm_injection_info.append([float(start), float(end), eval(pair)])
        print(f"!!!!Computation:{self.comp_injection_info}, Communication:{self.comm_injection_info}")
        self.comp_line_no = 0
        self.comm_line_no = 0
        self.version = 1
        self.in_slow = False
        self.iter_slow_start = -1
        self.port_version = 0
        self.redis_cli = redis_cli

    def check_communication(self, iteration, rank, device_id):
        if self.comm_line_no >= len(self.comm_injection_info):
            print("All COMM fail-slows are injected!!!")
            return

        # get current global ranks
        start_t = time.time() - self.init_time
        slow_start, duration, slow_pair = self.comm_injection_info[self.comm_line_no]
        print(f"====Check COMM Time={start_t}, RANK={rank}, iter={iteration}, lineno={self.comm_line_no}, line={self.comm_injection_info[self.comm_line_no]}")
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if start_t >= slow_start:
            print(f"Inject slow comm: {slow_pair}")
            self.comm_line_no += 1
            cmd_base = "python ./single_comm.py --tensor-size 200 --duration {} --device {}"
            self.port_version += 1
            # Sender
            if rank == slow_pair[0]:
                cmd = cmd_base.format(duration, device_id)
                print("Sender!!!!", rank, cmd, os.getcwd())
                p_sender = subprocess.Popen(cmd, shell=True, stdout=sys.stderr, stderr=sys.stderr,
                                            env={'MASTER_ADDR': str(ip), 'MASTER_PORT': str(9969 + self.port_version), 'WORLD_SIZE': '2', 'RANK': '0'})
            # Recver
            elif rank == slow_pair[1]:
                cmd = cmd_base.format(duration, device_id)
                print("Recver!!!!", rank, cmd, os.getcwd())
                p_recver = subprocess.Popen(cmd, shell=True, stdout=sys.stderr, stderr=sys.stderr,
                                            env={'MASTER_ADDR': str(ip), 'MASTER_PORT': str(9969 + self.port_version), 'WORLD_SIZE': '2', 'RANK': '1'})
            else:
                return


    def check_computation(self, iteration):
        if self.comp_line_no >= len(self.comp_injection_info):
            print("All COMP fail-slows are injected!!!")
            return

        # get current global ranks
        start_t = time.time() - self.init_time
        global_ranks = self.comp_injection_info[self.comp_line_no][2]
        delay_time = self.comp_injection_info[self.comp_line_no][3]
        print(f"====Check COMP Time={start_t}, iter={iteration}, lineno={self.comp_line_no}, line={self.comp_injection_info[self.comp_line_no]}, slow_iter_start={self.iter_slow_start}")

        # get number of DP groups
        dp_data = self.redis_cli.get("0_dp")
        tp_data = self.redis_cli.get("0_tp")
        pp_data = self.redis_cli.get("0_pp")
        if dp_data is not None:
            dp_data = dp_data.decode().split("_")
            num_dps = len(dp_data)
            tp_data = tp_data.decode().split("_")
            num_tps = len(tp_data)
            pp_data = pp_data.decode().split("_")
            num_pps = len(pp_data)
        else:
            num_dps = dist.get_world_size()
            num_tps = dist.get_world_size()
            num_pps = dist.get_world_size()

        # get micro and global batch size
        micro_bsz = self.redis_cli.get("micro_batch_size")
        global_bsz = self.redis_cli.get("global_batch_size")
        if micro_bsz is not None and global_bsz is not None:
            micro_bsz = int(micro_bsz.decode())
            global_bsz = int(global_bsz.decode())
        else:
            micro_bsz, global_bsz = 2, 256

        if start_t >= self.comp_injection_info[self.comp_line_no][0] and (not self.in_slow):
            print(f"Set delay for global rank {global_ranks}, delay_time = {delay_time}", file=sys.stderr)
            for grank in global_ranks:
                self.redis_cli.set(f"delay_time_{grank}", delay_time)
            self.in_slow = True
            self.iter_slow_start = iteration
            return
        # Handle fail-slow at 5 iterations after injection
        if start_t >= self.comp_injection_info[self.comp_line_no][0] and self.in_slow and iteration - self.iter_slow_start == 5:
            locked_dp_ranks = []
            for grank in global_ranks:
                in_stage_rank = grank % (num_dps * num_tps)
                locked_dp_ranks.append(in_stage_rank // num_tps)
            print(f"TP/DP/PP world size: {num_tps}/{num_dps}/{num_pps}, locked_dp_ranks: {locked_dp_ranks}, micro_bsz: {micro_bsz}, global_bsz: {global_bsz}", file=sys.stderr)
            # generate performance metrics
            compute_time = {i: PerformanceMetric(65, 65, 65, 0.01) for i in range(num_dps)}
            for locked_rank in locked_dp_ranks:
                compute_time[locked_rank] = PerformanceMetric(515, 515, 515, 0.01)

            # adjust DP by setting redis
            time_array = get_time_array(self.redis_cli, compute_time)
            print("Iter times:", time_array, file=sys.stderr)
            dp_ret = solve_dp(time_array/1000.0, micro_bsz, global_bsz)
            print(f"DP: {dp_ret}, version={self.version}", file=sys.stderr)
            self.redis_cli.set('batch_distribution', str(dp_ret))
            self.redis_cli.set("dp_version", self.version)
            return
        if start_t >= self.comp_injection_info[self.comp_line_no][1] and (self.in_slow):
            print(f"End of delay for global rank {global_ranks}", file=sys.stderr)
            for grank in global_ranks:
                self.redis_cli.set(f"delay_time_{grank}", 0)
            # Adjust to fair DP
            fair_dp = [global_bsz // (micro_bsz * num_dps) for _ in range(num_dps)]
            print("Fair DP: ", fair_dp, file=sys.stderr)
            self.redis_cli.set('batch_distribution', str(fair_dp))
            self.redis_cli.set("dp_version", self.version + 1)
            self.version += 2
            self.in_slow = False
            self.comp_line_no += 1

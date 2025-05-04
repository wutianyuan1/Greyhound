import torch
import time
import os
import logging
from typing import List


class CostEstimator(object):
    def __init__(self) -> None:
        pass

    def perf_ckpt_restore_time(self):
        # profile a 200MB tensor (100M params), get checkpoint and restore time in milliseconds
        # checkpointing
        tensor = torch.zeros((100, 1000, 1000), dtype=torch.float16, device='cuda:0')
        f = open("tmp_tensor.pt", 'wb')
        start = time.time()
        torch.save(tensor, f)
        f.flush()
        f.close()
        end = time.time()
        # cleanup system cache
        os.system("echo 3 > /proc/sys/vm/drop_caches")
        # restoring
        f = open("tmp_tensor.pt", 'rb')
        start2 = time.time()
        torch.load(f, map_location='cuda:0')
        end2 = time.time()
        f.close()
        # cleanup
        os.remove("tmp_tensor.pt")
        logging.critical(f"write BW={200/(end - start):.2f}MB/s, read BW={200/(end2 - start2):.2f}MB/s")
        return (end - start) * 1000, (end2 - start2) * 1000

    def get_dp_adjustment_cost(self, dp_check_interval: int, min_iter: float, slow_iter: float) -> float:
        # If we adjust DP, the training system will response in dp_check_interval/2 expected time
        # During this period, it will slow down by (slow_iter - min_iter)
        # This equals to we can adjust it back but shut down by the following time
        return dp_check_interval * (slow_iter - min_iter) / 2

    def get_pp_adjustment_cost(self) -> float:
        return 60 * 1000

    def get_restart_adjustment_cost(self, model_params: List[int]=[]) -> float:
        # @param model_params: #parameters per node
        ckpt_time, restore_time = self.perf_ckpt_restore_time()
        init_time = 10000
        max_ckpt_time = 0
        max_restore_time = 0
        for param in model_params:
            node_ckpt_time = param * ckpt_time / (100 * 1000 * 1000)
            node_restore_time = param * restore_time / (100 * 1000 * 1000)
            max_ckpt_time = max(node_ckpt_time, max_ckpt_time)
            max_restore_time = max(node_restore_time, max_restore_time)
        logging.critical(f"checkpointing={max_ckpt_time}ms, restoring={max_restore_time}ms, initialization={init_time}ms")
        return max_ckpt_time + max_restore_time + init_time


if __name__ == '__main__':
    est = CostEstimator()
    print(est.get_restart_adjustment_cost([100*1024*1024*1024, 200*1024*1024*1024]))

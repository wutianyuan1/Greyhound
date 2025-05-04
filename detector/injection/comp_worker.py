import os
import time
import torch
import argparse
import logging
import torch.distributed as dist
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--logdir", type=str, default='/workspace/Megatron-failslow/trainlog/')
    args = parser.parse_args()

    time_str = str(datetime.now()).replace(" ", '_').replace('-', '_').replace(':', '_').replace('.', '_')
    logpath = args.logdir + f"/comp_worker_{time_str}.log"
    device_id = args.device
    duration = args.duration

    logging.getLogger().setLevel(logging.INFO)
    # logging.basicConfig(filename=logpath)
    t0 = time.time()
    running_time = 0
    a = torch.rand((20, 1024, 1024), dtype=torch.float32, device=f"cuda:{device_id}")
    b = torch.rand((20, 1024, 1024), dtype=torch.float32, device=f"cuda:{device_id}")
    while running_time < duration:
        c = torch.bmm(a, b)
        running_time = time.time() - t0


if __name__ == "__main__":
    main()
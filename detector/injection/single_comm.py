import os
import time
import torch
import argparse
import logging
import torch.distributed as dist
from datetime import datetime


def send_recv(rank, tensor_size = 1024 * 1024 * 25, repeat=1):
    logging.info(f"[{datetime.now()}] Rank {rank} invoked, tensor_size={tensor_size * 4 / (1024 * 1024)}MB!")
    send_tensor = torch.randn(tensor_size, device='cuda:0') if rank == 0 else None
    recv_tensor = torch.empty(tensor_size, device='cuda:1') if rank == 1 else None

    if rank == 0:
        start_time = time.time()
        for _ in range(repeat):
            dist.send(send_tensor, dst=1)
        torch.cuda.synchronize()
        end_time = time.time()
        bandwidth = tensor_size * 4 * repeat / (end_time - start_time) / (1024 * 1024)  # MB/s
        logging.info(f"[{datetime.now()}] Rank {rank} sent data. Bandwidth: {bandwidth:.2f} MB/s")

    if rank == 1:
        start_time = time.time()
        for _ in range(repeat):
            dist.recv(recv_tensor, src=0)
        torch.cuda.synchronize()
        end_time = time.time()
        bandwidth = tensor_size * 4 * repeat / (end_time - start_time) / (1024 * 1024)  # MB/s
        logging.info(f"[{datetime.now()}] Rank {rank} received data. Bandwidth: {bandwidth:.2f} MB/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-size", type=int, default=100)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--logdir", type=str, default='/workspace/Megatron-failslow/trainlog/')
    args = parser.parse_args()

    tensor_size = 1024 * 1024 * (args.tensor_size // 4)
    time_str = str(datetime.now()).replace(" ", '_').replace('-', '_').replace(':', '_').replace('.', '_')
    logpath = args.logdir + f"/comm_worker_{time_str}.log"
    duration = args.duration

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    logging.getLogger().setLevel(logging.INFO)
    # logging.basicConfig(filename=logpath)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.all_reduce(torch.tensor([1], device=f'cuda:{rank}'))
    t0 = time.time()
    running_time = 0
    while running_time < duration:
        send_recv(rank, tensor_size=tensor_size, repeat=1500)
        running_time = time.time() - t0


if __name__ == "__main__":
    main()
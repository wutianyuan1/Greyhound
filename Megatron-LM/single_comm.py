import os
import time
import torch
import argparse
import logging
import torch.distributed as dist
from datetime import datetime


def send_recv(rank, tensor_size = 1024 * 1024 * 25, repeat=1, device=0):
    print(f"@@@@@@@@@@[{datetime.now()}] Rank {rank} invoked, tensor_size={tensor_size * 4 / (1024 * 1024)}MB!")
    send_tensor = torch.randn(tensor_size, device=f'cuda:{device}') if rank == 0 else None
    recv_tensor = torch.empty(tensor_size, device=f'cuda:{device}') if rank == 1 else None

    if rank == 0:
        start_time = time.time()
        for _ in range(repeat):
            dist.send(send_tensor, dst=1)
            torch.cuda.synchronize()
        dist.all_reduce(torch.tensor([1], device=f'cuda:{device}'))
        end_time = time.time()
        bandwidth = tensor_size * 4 * repeat / (end_time - start_time) / (1024 * 1024)  # MB/s
        print(f"@@@@@@@@@@[{datetime.now()}] Rank {rank} sent data. Bandwidth: {bandwidth:.2f} MB/s")

    if rank == 1:
        start_time = time.time()
        for _ in range(repeat):
            dist.recv(recv_tensor, src=0)
            torch.cuda.synchronize()
        dist.all_reduce(torch.tensor([1], device=f'cuda:{device}'))
        end_time = time.time()
        bandwidth = tensor_size * 4 * repeat / (end_time - start_time) / (1024 * 1024)  # MB/s
        print(f"@@@@@@@@@@[{datetime.now()}] Rank {rank} received data. Bandwidth: {bandwidth:.2f} MB/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-size", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    master_addr = os.getenv("MASTER_ADDR")
    master_port = os.getenv("MASTER_PORT")
    print(f"@@@@@@@ INIT rank{rank} worldsize={world_size}, master_addr={master_addr} master_port={master_port} device={args.device}")

    tensor_size = 1024 * 1024 * (args.tensor_size // 4)
    device = args.device
    duration = int(args.duration)

    # logging.basicConfig(filename=logpath)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"@@@@@@@ AFTERINIT rank{rank}, deivce={device} worldsize={world_size}!!!")
    dist.all_reduce(torch.tensor([1], device=f'cuda:{device}'))
    print(f"@@@@@@@ here RUNNING rank{rank}, deivce={device} worldsize={world_size}!!!")
    for i in range(int(duration)):
        send_recv(rank, tensor_size=tensor_size, repeat=100, device=device)
        print(f"@@@@@@@ ITERATION{i}/{duration} rank{rank}, deivce={device} worldsize={world_size}!!!")
    print(f"@@@@@@@ DONE rank{rank}, deivce={device} worldsize={world_size}!!!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
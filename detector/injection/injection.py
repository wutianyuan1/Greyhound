import os
import time
import numpy as np
import pickle
import random
import redis
import socket
import torch
import multiprocessing as mp
import multiprocessing.pool
import torch.distributed as dist
import matplotlib.pyplot as plt
from slow_comp import set_computation_slow
from slow_comm import set_communication_slow


sim_factor = 1


def print_error(e):
    print("Error:", e)


def generate_slow_durations(nnodes=1, load_fn=None):
    num_gpus = 4
    pp_stages = 4
    dp_groups = nnodes / pp_stages
    if load_fn is None:
        ret = []
        for node_id in range(1):
            timestamps = []
            current_time = 0
            while current_time < 6000:
                # poisson process
                interarrival_time = np.random.exponential(scale=240)
                start_time = current_time + interarrival_time
                duration = max(1, int(np.random.normal(loc=120, scale=20)))
                reason = np.random.randint(0, 2)  # 0: comp; 1: comm
                if reason == 0:
                    gpu_1 = np.random.randint(0, num_gpus)
                    gpu_2 = gpu_1
                    node_1 = node_id
                    node_2 = node_id
                else:
                    gpu_1 = 0
                    gpu_2 = 0
                    congestion_stage = np.random.choice(list(range(pp_stages)))
                    candidate_nodes = [i for i in range(int(congestion_stage * dp_groups), int((congestion_stage + 1) * dp_groups))]
                    node_1 = np.random.choice([i for i in candidate_nodes])
                    node_2 = np.random.choice([i for i in candidate_nodes if i != node_1])
                timestamps.append((start_time, duration, node_1, gpu_1, node_2, gpu_2, reason))
                current_time = start_time + duration + 60
            ret.append(np.array(timestamps, dtype=int))
        with open(f"{nnodes}node.pkl", 'wb') as f:
            pickle.dump(ret, f)
        return ret
    else:
        with open(load_fn, 'rb') as f:
            return pickle.load(f)


def plot_slow_node(data):
    coords = [[(0, 3000)] for _ in range(8)]
    max_t = 0
    for line in data:
        start, dur, gpu = line[0], line[1], line[-4]
        coords[gpu].append((start, 3000))
        coords[gpu].append((start + 0.01, 100))
        coords[gpu].append((start + dur, 100))
        coords[gpu].append((start + dur + 0.01, 3000))
        max_t = start + dur + 100
    for i in range(len(coords)):
        coords[i].append((max_t, 3000))
    for i, line in enumerate(coords):
        line = np.array(line)
        plt.plot(line[:, 0], line[:, 1], label=f"GPU{i}")
    plt.xlabel("Time / s")
    plt.ylabel("Frequency / MHz")
    plt.legend()
    plt.savefig("1node.png")



def main():
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])
    rank = int(os.environ['RANK'])

    # Initialize the process group
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )

    # node_id is rank_id because each node only run one proc
    node_id = dist.get_rank()
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    ip_tensor = torch.tensor([int(i) for i in ip.split(".")], dtype=torch.int32)
    print(f"My rank: {node_id}, IP={ip}, ip_tensor={ip_tensor}")

    ip_table = [torch.zeros(4, dtype=torch.int32) for _ in range(world_size)]
    dist.all_gather(ip_table, ip_tensor)
    ip_table = [".".join([str(i) for i in line.tolist()]) for line in ip_table]
    print(f"IP table: {ip_table}")

    # reset GPU frequency
    os.system("nvidia-smi -lgc 3000")
    time.sleep(1)


    # Get the traces to run
    # all_data = generate_slow_durations(load_fn=f'{8}node.pkl')
    # my_data = [
        # [180, 240, 0, 4, 1, 3, 1], # comm: rank0->rank1, in ppstage-2
        # [180, 50, 1, [4, 5, 6], 1, [4, 5, 6], 0], # comm: rank0->rank1, in ppstage-2
        # [280, 240, 0, 4, 1, 3, 1], # comm: rank0->rank1, in ppstage-2
        # [250, 500, 5, 0, 5, 0, 0] # comp: rank5, in dpgroup-1
        # [80, 120, [0,1,2], 0, 1, 0, 0],
        # [260, 120, [0,1,2,3], 0, 1, 0, 0],
        # [440, 120, [0,1,2], 0, 1, 0, 0],
        # [620, 120, [0,1,2,3], 0, 1, 0, 0],
        # [120, 180, 0, 0, 1, 0, 1],
        # [120, 180, 4, 0, 5, 0, 1],
        # [120, 180, 6, 0, 7, 0, 1],
        # [25, 80, 2, 0, 3, 0, 1],
        # [130, 30, 2, 0, 2, 0, 0]

    ### Largescale, part1
    my_data = [
        [212, 97, 1, [4], 0.2, [4], 0],                     # comp-1, dp_12
        [510, 126, 1, [3, 7], 0.3, [3, 7], 0],              # comp-2, dp_11, 15
        [710, 1025, 6, 0, 7, 0, 1],                         # comm-1, last stage
        [1191, 140, 3, [2], 0.4, [2], 0],                   # comp-3, dp_10                            
        [1396, 108, 0, [0, 1, 2], 0.25, [0, 1, 2], 0],      # comp-4, dp_0, 1, 2
        [1612, 209, 7, [2, 5], 0.4, [2, 5], 0],             # comp-5: dp_10, 13
        [1868, 67, 5, [2, 4, 5], 0.3, [2, 4, 5], 0],        # comp-6, dp_10, 12, 13
        [2078, 339, 0, 0, 1, 0, 1],                         # comm-2, first stage
        [2504, 111, 2, [0], 0.35, [0], 0],                  # comp-7, dp_0
        [2785, 155, 4, [1, 4, 7], 0.4, [1, 4, 7], 0],       # comp-8, dp_1,4,7
        [3096, 507, 7, 0, 7, 0, 1],                         # comm-3, last stage
    ]
    print(my_data)


    pool = mp.Pool(len(my_data))
    rets = []
    version = 1
    st = time.time()
    for (i, line) in enumerate(my_data):
        start, duration, src_node, src_gpu, dst_node, dst_gpu, reason = line
        if reason == 0:
            level = dst_node
            ret = pool.apply_async(set_computation_slow,
                                   args=(i, start, duration, src_node, src_gpu, st, ip_table, sim_factor, version, level),
                                   error_callback=print_error)
            version += 2
        else:
            assert isinstance(src_node, int)
            assert isinstance(dst_node, int)
            ret = pool.apply_async(set_communication_slow,
                                   args=(i, start, duration, src_node, src_gpu, dst_node, dst_gpu, 9969+i, st, ip_table, sim_factor, torch.cuda.device_count()),
                                   error_callback=print_error)
        rets.append(ret)
    pool.close()
    pool.join()

    for r in rets:
        print("wait", r)
        r.wait()

    # x = dist.all_reduce(torch.tensor([1]))
    # print("allreduce2 done", x)

    # Finalize the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
    # print(generate_slow_durations(8))
    # print(generate_slow_durations(8, '8node.pkl'))

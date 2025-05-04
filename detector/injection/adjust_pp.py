import time
import numpy as np
from sys import argv
from redis import StrictRedis


def adjust_layer_split(redis_host, my_rank, world_size, sleep_time=5, is_back=0, gpus_per_node=8):
    time.sleep(sleep_time)
    print("Adjust!!!")
    client = StrictRedis(host=redis_host, port=6379)
    offset = 0

    dp_data = client.get("0_dp")
    if dp_data is not None:
        dp_data = dp_data.decode().split("_")
        num_dps = len(dp_data)
    else:
        num_dps = world_size
    my_pp_stage = my_rank * gpus_per_node // num_dps
    total_pp_stages = world_size * gpus_per_node // num_dps
    total_num_layers = client.get("total_num_layers")
    if total_num_layers is not None:
        total_num_layers = int(total_num_layers.decode())
    else:
        total_num_layers = 64
    print(f"[Rank {my_rank}] Apply layer split: gpus_per_node={gpus_per_node}, DP size={num_dps}, my_pp_stage={my_pp_stage}, total_pp_stages={total_pp_stages}, total_num_layers={total_num_layers}")
    if is_back == 0:
        layer_split = []
        throughput = np.array([1 for _ in range(total_pp_stages)], dtype=float)
        throughput[my_pp_stage] = 0.6
        throughput = throughput / np.sum(throughput)
        for i in range(len(throughput) - 1):
            layer_split.append(round(total_num_layers * throughput[i]))
        last_num_layers = total_num_layers - sum(layer_split)
        layer_split.append(last_num_layers)
        print(f"[Rank {my_rank}] Throughput={throughput} Layer split={layer_split}")
    else:
        layer_split = [total_num_layers//total_pp_stages for _ in range(total_pp_stages)]
        print(f"[Rank {my_rank}] Split back: Layer split={layer_split}")


    for i in range(len(layer_split)):
        client.set(f"pp_offset_{i}", offset)
        client.set(f"pp_num_layers_{i}", layer_split[i])
        offset += layer_split[i]
    
    # pause the training process and invoke restart from memory
    client.set("terminate_ctl", 123)

    print(f"Apply layer split {layer_split} done")

if __name__ == '__main__':
    redis_host, my_rank, world_size, sleep_time, is_back, gpus_per_node = argv[1], int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5]), int(argv[6])
    adjust_layer_split(redis_host, my_rank, world_size, sleep_time, is_back, gpus_per_node)
    print("Adjust done!!!!")

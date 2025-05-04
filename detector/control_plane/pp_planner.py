import redis
import logging
from typing import List


def apply_layer_split(client: redis.StrictRedis, layer_split: List[int]):
    offset = 0
    for i in range(len(layer_split)):
        client.set(f"pp_offset_{i}", offset)
        client.set(f"pp_num_layers_{i}", layer_split[i])
        offset += layer_split[i]
    logging.critical(f"Apply layer split {layer_split} done")


if __name__ == '__main__':
    client = redis.StrictRedis("localhost", 6379)
    layer_split = [8, 5, 10, 9]
    apply_layer_split(client, layer_split)

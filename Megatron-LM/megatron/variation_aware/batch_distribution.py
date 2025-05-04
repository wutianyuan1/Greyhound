import torch
import os
import torch.distributed
import redis

from megatron.core import mpu


GLOBAL_BATCH_DISTRIBUTION = None
REDIS_CLIENT = None
DP_MODIFICATION_VER = 0
INITIAL_MB_NUM = 0
CURRENT_MB_NUM = 0

def sync_global_batch_distribution():
    global REDIS_CLIENT, GLOBAL_BATCH_DISTRIBUTION
    content = REDIS_CLIENT.get('batch_distribution')
    if content is not None:
        GLOBAL_BATCH_DISTRIBUTION = eval(content.decode())


def init_batch_distribution():
    global REDIS_CLIENT
    if REDIS_CLIENT is None:
        REDIS_CLIENT = redis.StrictRedis(
            host=os.getenv("MASTER_ADDR", 'localhost'),
            port=int(os.getenv("REDIS_PORT", '6379')),
            db=0
        )
    REDIS_CLIENT.set("dp_version", 0)
    sync_global_batch_distribution()


def set_initial_micro_batch_num(my_dp_rank, my_mb_num):
    global REDIS_CLIENT, CURRENT_MB_NUM, INITIAL_MB_NUM
    assert REDIS_CLIENT is not None
    CURRENT_MB_NUM = my_mb_num
    INITIAL_MB_NUM = my_mb_num
    REDIS_CLIENT.set(f"init_mb_num_{my_dp_rank}", my_mb_num)


def get_my_micro_batch_num():
    global GLOBAL_BATCH_DISTRIBUTION, DP_MODIFICATION_VER, CURRENT_MB_NUM, REDIS_CLIENT
    if mpu.is_unitialized():
        return None
    my_dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
    new_version = int(REDIS_CLIENT.get("dp_version"))
    # Case 1: My DP batch distribution is the newest version => Do nothing
    if new_version == DP_MODIFICATION_VER:
        return CURRENT_MB_NUM
    # Case 2: My DP batch distribution should be updated => Update it
    sync_global_batch_distribution()
    print(f"###### DP changed: from {CURRENT_MB_NUM} to {GLOBAL_BATCH_DISTRIBUTION[my_dp_rank]}")
    CURRENT_MB_NUM = GLOBAL_BATCH_DISTRIBUTION[my_dp_rank]
    DP_MODIFICATION_VER = new_version
    return CURRENT_MB_NUM


def get_loss_and_grad_weight():
    return CURRENT_MB_NUM / INITIAL_MB_NUM

# mpu.get_data_parallel_group()
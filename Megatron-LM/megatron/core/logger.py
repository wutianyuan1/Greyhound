from multiprocessing import Process
import time
import os
import redis
import redis.exceptions


def do_log(redis_addr, redis_port, my_rank, tp_ranks, dp_ranks, pp_ranks):
    db = redis.StrictRedis(redis_addr, redis_port, db=0)
    # Wait for redis initialization
    while True:
        try:
            db.ping()
            break
        except redis.exceptions.ConnectionError:
            time.sleep(1)
    tp_ranks = "_".join(str(i) for i in tp_ranks)
    dp_ranks = "_".join(str(i) for i in dp_ranks)
    pp_ranks = "_".join(str(i) for i in pp_ranks)
    db.set(f"{my_rank}_tp", tp_ranks)
    db.set(f"{my_rank}_dp", dp_ranks)
    db.set(f"{my_rank}_pp", pp_ranks)
    print("Parallel states are logged to redis!")
    

def log_model_parallel_to_redis(my_rank, tp_ranks, dp_ranks, pp_ranks):
    redis_ip = os.getenv('MASTER_ADDR', 'localhost')
    redis_port = os.getenv('REDIS_PORT', '6379')
    proc = Process(
        target=do_log,
        args=(redis_ip, redis_port, my_rank, tp_ranks, dp_ranks, pp_ranks)
    )
    proc.start()

import redis
import numpy as np
import cvxpy as cp
import logging
from typing import Dict


class PerformanceMetric(object):
    def __init__(self, min_lat, max_lat, avg_lat, std_lat) -> None:
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.avg_lat = avg_lat
        self.std_lat = std_lat

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"(Perf: min={self.min_lat}, max={self.max_lat}, avg={self.avg_lat}, std={self.std_lat})"



def get_time_array(redis_client: redis.StrictRedis, compute_time: Dict[int, PerformanceMetric], threshold: float = 1.1):
    # normal iteration time
    min_iter_time = float(redis_client.get('min_iter_time').decode())
    # iteration time after fail-slow
    slow_iter_time = float(redis_client.get("cur_iter_time").decode())
    logging.info(f"Min iter time: {min_iter_time}, slow time: {slow_iter_time}, compute_time: {compute_time}")
    # find the stats of compute validation
    time_array = np.zeros(len(compute_time), dtype=np.float32)
    vals = [i.avg_lat for i in compute_time.values()]
    min_compute_time, max_compute_time = np.min(vals), np.max(vals)
    median_compute_time = 70
    for rank in compute_time:
        rank_lat = compute_time[rank].avg_lat
        # faster than 1.1*median_compute => not fail slow
        if rank_lat <= threshold * median_compute_time:
            time_array[rank] = min_iter_time
        # fail slow => compute its slowdown percentage
        else:
            # assume compute time is linear to iteration time: iter=k*compute+b
            slope = (slow_iter_time - min_iter_time) / (max_compute_time - min_compute_time)
            intercept = slow_iter_time - slope * max_compute_time
            time_array[rank] = slope * rank_lat + intercept
    return time_array


def solve_dp(time_array: np.ndarray, micro_bsz: int, global_bsz: int):
    num_dp_groups = len(time_array)
    num_microbatches = cp.Variable(shape=num_dp_groups, integer=True)
    N_t = cp.multiply(num_microbatches, time_array)
    avg_N_t = cp.mean(N_t)
    variance = cp.sum_squares(N_t - avg_N_t)
    constraints = [
        num_microbatches >= 1,  # Each N_i must be positive
        cp.sum(num_microbatches * micro_bsz) == global_bsz  # The batch size sum constraint
    ]
    problem = cp.Problem(cp.Minimize(variance), constraints)
    problem.solve(solver=cp.ECOS_BB, verbose=False)
    num_microbatches = [round(i) for i in num_microbatches.value]
    logging.info(f"[DP solver] new DP plan is {num_microbatches}")
    return num_microbatches


if __name__ == '__main__':
    # redis_host = 'localhost'
    # redis_port = 6379
    # client = redis.StrictRedis(redis_host, redis_port, db=0)
    # compute_time = {
    #     0: PerformanceMetric(65, 65, 65, 0.01),
    #     1: PerformanceMetric(515, 515, 515, 50.01),
    # }
    # time_array = get_time_array(client, compute_time)
    time_array = np.array([3063.01, 3063.01, 3063.01, 3063.01, 3063.01, 3063.01, 3063.01, 3063.01, 3063.01, 3063.01, 3063.01, 3063.01, 4250.766, 3063.01, 3063.01, 3063.01])/1000
    print(time_array)
    ret = solve_dp(time_array, 2, 256)
    print(ret)

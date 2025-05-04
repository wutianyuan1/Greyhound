import numpy as np

def generate_slow_durations(nnodes=1, load_fn=None):
    ret = []
    for node_id in range(1):
        timestamps = []
        current_time = 200
        while current_time < 3300:
            # poisson process
            interarrival_time = np.random.exponential(scale=240)
            start_time = current_time + interarrival_time
            duration = max(1, int(np.random.normal(loc=120, scale=20)))
            timestamps.append((start_time, duration))
            current_time = start_time + duration + 60
        ret.append(np.array(timestamps, dtype=int))
    return ret[0]


np.random.seed(42)
print(generate_slow_durations())

'''
[
 [212, 97, 1, [4], 1, [4], 0]               # comp-1, dp_12
 [510, 126, 1, [3, 7], 1, [3, 7], 0]        # comp-2, dp_11, 15
 [710, 1025, 6, 0, 7, 0, 1]                 # comm-1, last stage
 [1191, 140, 3, [2], 3, [2], 0]             # comp-3, dp_10                            
 [1396, 108, 0, [0, 1, 2], 0, [0, 1, 2], 0] # comp-4, dp_0, 1, 2
 [1612, 109, 7, [2, 5], 7, [2, 5], 0]       # comp-5: dp_10, 13
 [1868   67]
 [2078  139]
 [2504  111]
 [2785  105]
 [3096  107]
 [3436  131]]
'''
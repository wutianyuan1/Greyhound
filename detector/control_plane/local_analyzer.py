import json
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from .slow_detection import find_period, find_performance_drop
from .visualizer import ValueLogger
from multiprocessing import shared_memory, resource_tracker

OPS = ['Send', 'Recv', 'Bcast', 'Broadcast', 'AllGather', 'ReduceScatter', 'AllReduce']
SIZEOF_INT64 = 8


def sizestr(size):
    if size < 1024:
        return str(size)
    elif size < 1024 * 1024:
        return str(size // 1024) + "KB"
    elif size < 1024 * 1024 * 1024:
        return str(size // (1024**2)) + "MB"
    else:
        return str(size // (1024**3)) + "GB"


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return json.load(f)


def remove_shm_from_resource_tracker():
    """This is a bug of multiprocessing.shared_memory, manully fix it here
    Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked
    More details at: https://bugs.python.org/issue38119
    """
    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


class NcclRecord(object):
    attrs = ['comm_addr', 'call_number', 'count', 'buff1', 'buff2',
             'datatype', 'pid', 'call_time', 'device', 'global_rank',
             'aux', 'duration', 'num_devices', 'event_id']

    def __init__(self, config):
        remove_shm_from_resource_tracker()
        self.shm_size = (config['NUM_FIELDS'] * config['BUFFER_SIZE'] + config['METADATA_FIELDS']) * SIZEOF_INT64
        # Wait until the training process creates a shared memory, then we can access it.
        shm_exists = False
        while not shm_exists:
            try:
                self.shm = shared_memory.SharedMemory("ncclRecord", create=False, size=self.shm_size)
                shm_exists = True
            except FileNotFoundError:
                logging.info("Shared memory not found. Waiting...")
                time.sleep(1)  # Wait for 1 second before checking again
        logging.info("Linked to the shm buffer!")

        self.data = np.frombuffer(self.shm.buf, np.int64)
        self.buffer = self.data[config['METADATA_FIELDS']:]
        self.num_fields = self.data[0]
        self.max_records = self.data[1]

    def __del__(self):
        # Remove the mmap from the shared memory regions
        if not hasattr(self, 'buffer'):
            return
        if self.buffer is not None:
            del self.buffer
        if self.data is not None:
            del self.data
    
    def clear(self):
        self.data[2] = 0
        self.data[3] = 0
        self.data[4] = 0

    @property
    def num_records(self):
        return self.data[2]

    def get_profile_data(self, metric_name):
        if metric_name == 'event_id':
            return [i[-1] for i in self]
        metric_id = self.attrs.index(metric_name)
        ret = []
        for record in self:
            ret.append(record[metric_id])
        return np.array(ret)

    def __getitem__(self, idx):
        if idx >= self.num_records:
            raise StopIteration
        head = self.data[3]
        start = ((head + idx) % self.max_records) * self.num_fields
        end = start + self.num_fields
        return self.buffer[start: end]

    def __str__(self):
        return 'RingBuffer[' + ','.join(str(self[i]) for i in range(self.num_records)) + ']'

    def __repr__(self):
        return str(self)


def plot_call_interval(record: NcclRecord):
    field_keys = record.attrs
    record_df = pd.DataFrame([i for i in record], columns=field_keys)
    f, axs = plt.subplots(1, 4, sharey='row', figsize=(12, 3))
    colors = ['powderblue', 'grey', 'lightblue', 'red', 'lightyellow', 'pink', 'lightgreen']
    stats = []
    for global_rank, per_gpu_calls in record_df.groupby('global_rank'):
        dts = {}
        for op_id, per_op_calls in per_gpu_calls.groupby("call_number"):
            per_op_calls = per_op_calls.sort_values(by=['event_id'])
            ts = per_op_calls['call_time'].to_numpy()
            dt = (ts[1:] - ts[:-1]) / (1000 * 1000)  # convert microsecond to second
            # skip the rare calls
            if len(dt) < 50:
                continue
            dts[op_id] = dt
            stats.append([global_rank, OPS[op_id], len(dt), np.mean(dt), np.std(dt)])
        bplot = axs[global_rank].boxplot(
            list(dts.values()), notch=True, vert=True, patch_artist=True,
            showfliers=False, labels=[OPS[key] for key in dts])
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        axs[global_rank].set_xlabel("NCCL Operation")
        axs[global_rank].set_title(f"GPU {global_rank}")
        if global_rank == 0:
            axs[global_rank].set_ylabel(r"$\Delta$ t / s")
    stats_df = pd.DataFrame(stats, columns=['GPU_ID', 'OP_ID', 'LEN', 'INTERVAL_MEAN', 'INTERVAL_STD'])
    stats_df.to_csv("logs/interval_stats.csv")
    plt.tight_layout()
    plt.savefig("figs/new_dt.png")


def detect_failslow(record: NcclRecord, plot=False):
    field_keys = record.attrs
    record_df = pd.DataFrame([i for i in record], columns=field_keys)
    if plot:
        f, axs = plt.subplots(1, 4, sharey='row', figsize=(12, 3))
        colors = ['powderblue', 'grey', 'pink', 'green']

    performance_drops = {}
    last_event_ids = {}
    estimated_iter_time = {}
    for (global_rank, per_gpu_record) in record_df.groupby("global_rank"):
        per_gpu_record.sort_values(by='event_id', inplace=True)
        last_event_ids[global_rank] = per_gpu_record['event_id'].iloc[-1]
        call_time = per_gpu_record['call_time'].to_numpy()
        call_id = per_gpu_record['call_number'].to_numpy()
        start, period = find_period(call_id, nlags=200, significance_level=0.95)
        if period is None:
            return None
        if plot:
            pargs = {"ax": axs[global_rank], "color": colors[global_rank], "label": f"GPU_{global_rank}",
                    "xlabel": "Execution Time / us", "ylabel": "Iteration Time / us"}
        else:
            pargs = None
        logging.info(f"Rank {global_rank}: repeat pattern starts from {start}, period = {period}, pattern = {call_id[start: start + period]}")
        performance_drops[global_rank], estimated_iter_time['rank' + str(global_rank)] = find_performance_drop(
            call_id, call_time, period, start, plot=plot, plot_args=pargs)
    logging.info(f"[{datetime.now()}] Esimated iteration time: {estimated_iter_time}\n")
    # logger = ValueLogger()
    # logger.push_val("EstimatedIterationTime", 'rank', estimated_iter_time)

    if plot:
        plt.tight_layout()
        plt.savefig("figs/period.png")
    return last_event_ids, performance_drops, estimated_iter_time


def get_profile_results(record: NcclRecord):
    field_keys = record.attrs
    record_df = pd.DataFrame([i for i in record], columns=field_keys)
    communication_times = {}

    for (global_rank, per_gpu_record) in record_df.groupby("global_rank"):
        per_gpu_record.sort_values(by='event_id', inplace=True)
        call_id = per_gpu_record['call_number'].to_numpy()
        start, period = find_period(call_id, nlags=200, significance_level=0.8)
        print(global_rank, start, period)

        record_iter = False
        for i in range(start, len(per_gpu_record), period):
            if i + period >= len(per_gpu_record):
                continue
            period_table = per_gpu_record[i:i+period]
            if np.sum(period_table['duration']) == 0:
                continue
            # skip additional one period
            if record_iter == False:
                record_iter = True
                continue
            for comm_addr, subtable in period_table.groupby("comm_addr"):
                comm_duration = np.sum(subtable['duration'])
                comm_count = np.sum(subtable['count'])
                if comm_duration != 0.0:
                    record_iter = True
                if comm_addr not in communication_times:
                    communication_times[comm_addr] = [(comm_count, comm_duration)]
                else:
                    communication_times[comm_addr].append((comm_count, comm_duration))

    result = {}
    for comm_key, comm_data in communication_times.items():
        result[comm_key] = np.array(comm_data)
    return result


if __name__ == '__main__':
    logging.basicConfig(filename='logs/analyzer.log')
    logging.getLogger().setLevel(logging.INFO)
    conf = load_config("./control_plane/config.json")
    record = NcclRecord(conf)

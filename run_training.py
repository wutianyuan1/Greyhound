import os
import abc
import sys
import time
import torch
from datetime import datetime
import dataclasses
import redis
from math import sqrt, floor
import subprocess
import argparse
import re
import socket


@dataclasses.dataclass
class BaseConfig(abc.ABC):
    def to_config_string(self) -> str:
        config_str = ""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) and value:
                config_str += f"--{field.name.replace('_', '-')} "
            elif not isinstance(value, bool):
                config_str += f"--{field.name.replace('_', '-')} {value} "
        return config_str.strip()


@dataclasses.dataclass
class DistributedConfig(BaseConfig):
    nproc_per_node: int
    nnodes: int
    node_rank: int
    master_addr: str
    master_port: str


@dataclasses.dataclass
class DatasetConfig(BaseConfig):
    vocab_file: str
    merge_file: str
    data_path: str
    split: str
    mock_data: bool


@dataclasses.dataclass
class ModelConfig(BaseConfig):
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    seq_length: int
    max_position_embeddings: int
    micro_batch_size: int
    global_batch_size: int
    lr: float
    train_iters: int
    lr_decay_iters: int
    lr_decay_style: str
    min_lr: float
    weight_decay: float
    lr_warmup_fraction: str
    clip_grad: float
    fp16: bool


@dataclasses.dataclass
class TrainConfig(BaseConfig):
    distributed_config: DistributedConfig
    dataset_config: DatasetConfig
    model_config: ModelConfig
    log_interval: int
    save_interval: int
    eval_interval: int
    eval_iters: int
    distributed_backend: str
    save: str   # checkpoint save path
    load: str   # checkpoint load path

    def to_config_string(self) -> str:
        other_config_str = ""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, BaseConfig):
                other_config_str += f"--{field.name.replace('_', '-')} {value} "
        return "torchrun {} pretrain_gpt.py {} {} --sequence-parallel {}".format(
            self.distributed_config.to_config_string(),
            self.model_config.to_config_string(),
            self.dataset_config.to_config_string(),
            other_config_str
        ).strip()


def run_and_log_megatron(megatron_cmd_args, log_file_handle, log_file_dir, distributed_config):
    # Start the subprocess
    print(megatron_cmd_args)
    my_env = os.environ
    my_env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    my_env['OMP_NUM_THREADS'] = '1'
    my_env['LD_PRELOAD'] = '/workspace/Greyhound/detector/build/libncclprobe.so'
    my_env['CONTROL_PLANE_WHL_PATH'] = '/workspace/Greyhound/detector/dist/control_plane-1.0-py3-none-any.whl'
    my_env['NCCLPROBE_LOG_PATH'] = log_file_dir
    my_env['GLOBAL_CONTROLLER_LOG_PATH'] = log_file_dir
    my_env['LOCAL_CONTROLLER_LOG_PATH'] = log_file_dir
    process = subprocess.Popen(megatron_cmd_args, text=True)
    iteration_time_pattern = re.compile(r'iteration \(ms\): ([\d.]+)')
    current_iteration_pattern = re.compile(r'iteration\s+(\d+)/')
    log_file_handle.write("Iteration, IterationTime(ms)\n")
    output = None
    # Read the output line by line
    while True:
        try:
            if output is not None:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output and ('[2024' in output):
                    # Find the iteration time and current iteration
                    iteration_time_match = iteration_time_pattern.search(output)
                    current_iteration_match = current_iteration_pattern.search(output)
                    if iteration_time_match and current_iteration_match:
                        iteration_time = iteration_time_match.group(1)
                        current_iteration = current_iteration_match.group(1)
                        # Log the parsed information to the file
                        log_entry = f"{current_iteration}, {iteration_time}\n"
                        print(log_entry)
                        log_file_handle.write(log_entry)
                        log_file_handle.flush()  # Ensure it writes immediately
        except KeyboardInterrupt:
            process.terminate()
            log_file_handle.write("Training terminated\n")
            log_file_handle.flush()
            break



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='/workspace/Greyhound/trainlog')
    parser.add_argument('--iter', type=int, default=10000)
    # parser.add_argument('--nnodes', type=int, default=1)
    # parser.add_argument('--rank', type=int, default=0)
    # parser.add_argument('--master', type=str, default='localhost')
    return parser.parse_args()


def main():
    os.chdir('./Megatron-LM/')
    args = get_args()
    log_file_dir, iter_1000 = args.logdir, args.iter
    master = "localhost"  # os.getenv("MASTER_ADDR")
    nnodes = 1  # int(os.getenv("WORLD_SIZE"))
    rank = 0  # int(os.getenv("RANK"))

    # start redis
    redis_cmd = ["redis-server", "--save", "\"\"", "--appendonly", "no", "--bind", f"{master}"]
    if rank == 0:
        redis_proc = subprocess.Popen(redis_cmd)
        redis_logstr = "Rank 0 starts redis: [" + " ".join(redis_cmd) + "]\n"
        time.sleep(2)
        time_str = str(datetime.now()).replace(" ", '_').replace('-', '_').replace(':', '_').replace('.', '_')
        log_file_dir = log_file_dir + '/log_' + time_str
        client = redis.StrictRedis(host=master, port=6379, db=0)
        client.set("trainlog_dir", log_file_dir)
    else:
        redis_proc = None
        redis_logstr = ""
        client = redis.StrictRedis(host=master, port=6379, db=0)
        while True:
            try:
                logdir_ret = client.get("trainlog_dir")
                if len(logdir_ret) > 5:
                    break
            except:
                pass
        print("Logdir worker!", logdir_ret)
        log_file_dir = logdir_ret.decode()
    
    log_file_dir += f"_rank{rank}"
    if not os.path.exists(log_file_dir):
        os.mkdir(log_file_dir)
    log_file_path = log_file_dir + f"/megatron_output_{rank}.log"

    tp = {1:1, 2:2, 4:2, 8:4}
    pp = {1:1, 2:1, 4:2, 8:2}

    num_gpus = torch.cuda.device_count()
    gpu_properties = torch.cuda.get_device_properties("cuda:0")
    gpu_memory = gpu_properties.total_memory / (1024**3)  # GB
    total_gmem = gpu_memory * num_gpus
    # Find a proper hidden size to fulfill the GPU memory
    hsize = int(1024 * (floor(sqrt(total_gmem / 18) * 2) / 2))
    hostname = socket.gethostname()
    ipaddr = socket.gethostbyname(hostname)

    info_str = f"***** log={log_file_path}, master={master}, nnodes={nnodes}, rank={rank},\
                num_gpus={num_gpus}, gpu_type={gpu_properties.name} gpu_memory={gpu_memory},\
                total_gmem={total_gmem}, hidden_size={hsize}, [My IP={ipaddr} Master IP={master}]\n"
    
    print(info_str)

    distributed_config = DistributedConfig(
        nproc_per_node=num_gpus, nnodes=nnodes, node_rank=rank, master_addr=master, master_port=6000
    )
    model_config = ModelConfig(
        tensor_model_parallel_size=tp[num_gpus], pipeline_model_parallel_size=pp[num_gpus], num_layers=64,
        hidden_size=hsize, num_attention_heads=32, seq_length=32, max_position_embeddings=1024, micro_batch_size=4,
        global_batch_size=16, lr=0.00015, train_iters=int(iter_1000), lr_decay_iters=int(0.64*iter_1000), lr_decay_style='cosine',
        min_lr=1.0e-5, weight_decay=0.01, lr_warmup_fraction='.01', clip_grad=1.0, fp16=True
    )
    dataset_config = DatasetConfig(
        vocab_file='/workspace/dataset/gpt2-vocab.json',
        merge_file='/workspace/dataset/gpt2-merges.txt',
        data_path='/workspace/dataset/gpt2_text_document',
        split='949,50,1', mock_data=True
    )
    train_config = TrainConfig(
        distributed_config=distributed_config,
        dataset_config=dataset_config,
        model_config=model_config,
        log_interval=1, save_interval=10000, eval_interval=10000, eval_iters=1, distributed_backend='nccl',
        save='/workspace/checkpoints', load='/workspace/checkpoints'
    )

    # Execute the command
    run_args = train_config.to_config_string().split(' ')
    with open(log_file_path, 'w') as log_file:
        log_file.write(info_str)
        log_file.write(redis_logstr)
        log_file.flush()
        run_and_log_megatron(run_args, log_file, log_file_dir, distributed_config)

    if redis_proc:
        redis_proc.terminate()

if __name__ == '__main__':
    main()

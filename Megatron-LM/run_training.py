import os
import time
import torch
import redis
import subprocess
import argparse
import socket
from datetime import datetime
from math import sqrt, floor
from train_config import DistributedConfig, TrainConfig, DatasetConfig, ModelConfig


def clean_all_redis_keys(redis_cli: redis.StrictRedis, preserves):
    to_dels = []
    for key in redis_cli.scan_iter("*"):
        key = key.decode()
        pattern_match = False
        for pattern in preserves:
            if pattern in key:
                pattern_match = True
        if not pattern_match:
            to_dels.append(key)
    redis_cli.delete(*to_dels)
    print([i for i in redis_cli.scan_iter("*")])


def run_and_log_megatron(megatron_cmd_args, log_file_path, log_file_dir, distributed_config, probe):
    # Start the subprocess
    print(megatron_cmd_args)
    my_env = os.environ
    my_env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    my_env['OMP_NUM_THREADS'] = '1'
    if probe != 0:
        my_env['LD_PRELOAD'] = '/workspace/ncclprobe/build/libncclprobe.so'
    my_env['CONTROL_PLANE_WHL_PATH'] = '/workspace/ncclprobe/dist/control_plane-1.0-py3-none-any.whl'
    my_env['NCCLPROBE_LOG_PATH'] = log_file_dir
    my_env['GLOBAL_CONTROLLER_LOG_PATH'] = log_file_dir
    my_env['LOCAL_CONTROLLER_LOG_PATH'] = log_file_dir
    my_env['NCCL_IB_GID_INDEX'] = '3'
    log_file_handle = open(log_file_path, 'a+')
    process = subprocess.Popen(megatron_cmd_args, stdout=log_file_handle, stderr=log_file_handle, text=True)
    redis_cli = redis.StrictRedis(host=os.getenv("MASTER_ADDR"), port=6379, db=0)
    # Read the output line by line
    ii = 0
    while True:
        try:
            time.sleep(1)
            ii += 1
            ctl_signal = redis_cli.get("terminate_ctl")
            if ii % 20 == 0:
                print(f"Rank {os.getenv('RANK', '0')}: ctl_signal={ctl_signal}")
            if ctl_signal is not None:
                ctl_signal = ctl_signal.decode()
                if ctl_signal == '123' or ctl_signal == '321':
                    print("######### partial restart: PP adjusted #########")
                    if int(os.getenv("RANK", '0')) == 0:
                        redis_cli.set("terminate", 1 if ctl_signal == '123' else 2)
                        print("RANK 0: delete old keys in redis")
                    process.wait()
                    if int(os.getenv("RANK", '0')) == 0:
                        clean_all_redis_keys(redis_cli, ['pp_offset', 'pp_num_layers'])
                    time.sleep(2)
                    process = subprocess.Popen(megatron_cmd_args, stdout=log_file_handle, stderr=log_file_handle, text=True)
                    if int(os.getenv("RANK", '0')) == 0:
                        redis_cli.set("terminate_ctl", "None")
            fin = redis_cli.get("finished")
            if fin is not None:
                process.terminate()
                log_file_handle.write("Training terminated\n")
                log_file_handle.flush()
                break
        except KeyboardInterrupt:
            process.terminate()
            log_file_handle.write("Training terminated\n")
            log_file_handle.flush()
            break


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='/workspace/Megatron-failslow/trainlog')
    parser.add_argument('--iter', type=int, default=10000)
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--pp', type=int, default=2)
    parser.add_argument('--probe', type=int, default=0)
    parser.add_argument('--hsize', type=int, default=2048)
    # parser.add_argument('--nnodes', type=int, default=1)
    # parser.add_argument('--rank', type=int, default=0)
    # parser.add_argument('--master', type=str, default='localhost')
    return parser.parse_args()


def main():
    os.chdir('/workspace/Megatron-failslow/')
    args = get_args()
    log_file_dir, iter_1000 = args.logdir, args.iter
    master = os.getenv("MASTER_ADDR", 'localhost')
    nnodes = int(os.getenv("WORLD_SIZE", '1'))
    rank = int(os.getenv("RANK", '0'))

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

    tp = args.tp
    pp = args.pp

    num_gpus = torch.cuda.device_count()
    gpu_properties = torch.cuda.get_device_properties("cuda:0")
    gpu_memory = gpu_properties.total_memory / (1024**3)  # GB
    total_gmem = gpu_memory * num_gpus
    # Find a proper hidden size to fulfill the GPU memory
    hsize = args.hsize  #int(1024 * (floor(sqrt(total_gmem / 18) * 2) / 2))
    hostname = socket.gethostname()
    ipaddr = socket.gethostbyname(hostname)

    info_str = f"***** log={log_file_path}, master={master}, nnodes={nnodes}, rank={rank},\
                num_gpus={num_gpus}, gpu_type={gpu_properties.name} gpu_memory={gpu_memory},\
                total_gmem={total_gmem}, hidden_size={hsize}, tp={tp}, pp={pp}, probe={args.probe},\
                [My IP={ipaddr} Master IP={master}]\n"
    
    print(info_str)

    distributed_config = DistributedConfig(
        nproc_per_node=num_gpus, nnodes=nnodes, node_rank=rank, master_addr=master, master_port=6000
    )
    model_config = ModelConfig(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, num_layers=64, #64,
        hidden_size=hsize, num_attention_heads=32, seq_length=32, max_position_embeddings=1024, micro_batch_size=4,
        global_batch_size=512, lr=0.00015, train_iters=int(iter_1000), lr_decay_iters=int(0.64*iter_1000), lr_decay_style='cosine',
        min_lr=1.0e-5, weight_decay=0.01, lr_warmup_fraction='.01', clip_grad=1.0, loss_scale=16384, fp16=True, failslow_aware=True
    )
    dataset_config = DatasetConfig(
        vocab_file='/workspace/dataset/gpt2-vocab.json',
        merge_file='/workspace/dataset/gpt2-merges.txt',
        # data_path='/workspace/dataset/gpt2_text_document',
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
    
    run_and_log_megatron(run_args, log_file_path, log_file_dir, distributed_config, args.probe)

    if redis_proc:
        redis_proc.terminate()

if __name__ == '__main__':
    main()

import os
import sys
import subprocess
import socket
import time
import torch.distributed as dist


nnodes = 8
hostname = socket.gethostname()
ipaddr = socket.gethostbyname(hostname)
if len(sys.argv) < 2:
    RANK = int(os.getenv("RANK"))
else:
    RANK = int(sys.argv[1])

if RANK == 0:
    os.system(f"echo {ipaddr} > /workspace/master.txt")
    MASTER_ADDR = ipaddr
else:
    while not os.path.exists("/workspace/master.txt"):
        pass
    with open("/workspace/master.txt", 'r') as f:
        content = f.read().strip("\n")
    MASTER_ADDR = content


print(f"My IP={ipaddr}, Master IP={MASTER_ADDR}, RANK={RANK}", file=sys.stderr)

os.chdir("/workspace/Megatron-failslow")
cmd = "MASTER_ADDR={} WORLD_SIZE={} RANK={} python run_training.py --tp 1 --pp 4 --probe 0 --hsize 3072"
cmd = cmd.format(MASTER_ADDR, nnodes, RANK)
train_proc = subprocess.Popen(cmd, shell=True)

os.chdir("/workspace/ncclprobe/injection")
cmd2 = "MASTER_ADDR={} WORLD_SIZE={} RANK={} MASTER_PORT=12345 python injection.py"
cmd2 = cmd2.format(MASTER_ADDR, nnodes, RANK)
inject_proc = subprocess.Popen(cmd2, shell=True)

print('='*80, file=sys.stderr)
while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        print("!!!! stop", file=sys.stderr)
        if RANK == 0:
            print("Removed!!", file=sys.stderr)
            os.system("rm /workspace/master.txt")
        train_proc.terminate()
        inject_proc.terminate()
        time.sleep(3)
        train_proc.kill()
        inject_proc.kill()
        break

import subprocess
from time import sleep

confs = [(1, 2), (1, 1), (1, 4), (4, 1)]

for c in confs:
    hsize = 2048
    if c[0] == 1 and c[1] == 1:
        hsize = 1024
    for probe in [0, 1]:
        cmd = f'python run_training.py --tp {c[0]} --pp {c[1]} --probe {probe} --hsize {hsize} --iter 100'
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
        sleep(2)
    
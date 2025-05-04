import re
import sys
import subprocess
from collections import Counter


path = sys.argv[1]
proc = subprocess.Popen(f"grep -r torch.distributed. {path}",
                        shell=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, close_fds=True)
try:
    stdout, stderr = proc.communicate(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
    print("Timeout!!!")
    exit(-1)

stdout = stdout.decode()
primitives = re.findall(r"(torch\.distributed\.[A-Za-z][A-Za-z0-9_]*)", stdout)
primitives = Counter(primitives)
max_plen = max(len(i) for i in primitives.keys()) + 4
for (k, v) in primitives.items():
    print(k, v, sep=(" " * (max_plen - len(k))))

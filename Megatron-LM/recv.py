import torch
import torch.distributed as dist
import os
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
torch.cuda.set_device(1)
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8000', rank=0, world_size=2)

def f(A, B):
    h = dist.irecv(B, 1)
    if h:
        h.wait()
    else:
        print("None!!!")
    output = A + B
    return output

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
A = torch.ones((5,5)).cuda()
B = torch.ones((5,5)).cuda()
C = torch.ones((5,5)).cuda()
with torch.cuda.stream(s):
    for i in range(11):
        f(A, B)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool

with torch.cuda.graph(g):
    C = f(A, B)

torch.cuda.synchronize()
torch.cuda.synchronize()
print(f"A.py C : {C}")

A[:, :] = 6
B[:, :] = 10
for _ in range(10):
    g.replay()
print(f"A.py C : {C}")
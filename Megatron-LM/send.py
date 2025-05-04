import os
import torch
import torch.distributed as dist
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
torch.cuda.set_device(0)
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8000', rank=1, world_size=2)

a = torch.ones((5,5)).cuda()
b = torch.ones((5,5)).cuda()

def f():
    # c = a*b
    h = dist.isend(b, 0)
    if h:
        h.wait()
    else:
        print("None!!!")
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(s):
    for i in range(11):
        f()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool

with torch.cuda.graph(g):
    f()
torch.cuda.synchronize()
b[:, :] = 114514
for _ in range(6):
    g.replay()
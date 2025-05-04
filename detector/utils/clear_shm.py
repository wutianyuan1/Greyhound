from multiprocessing import shared_memory


try:
    shm_size = (7 * 1000 + 5) * 8
    shm = shared_memory.SharedMemory("ncclRecord", create=False, size=shm_size)
    shm.close()
    shm.unlink()
except FileNotFoundError:
    print("No record")

try:
    shm = shared_memory.SharedMemory("recordLock", create=False)
    shm.close()
    shm.unlink()
except FileNotFoundError:
    print("No lock")

try:
    shm = shared_memory.SharedMemory("ncclTopo", create=False)
    shm.close()
    shm.unlink()
except FileNotFoundError:
    print("No Topo")

import struct
from typing import List

MAXCHANNELS = 32
NCCL_MAX_TREE_ARITY = 3


class RingNode(object):
    size = 4 + 4 + 8 + 4 + 4  # 4 extra bytes because of memory alignment
    def __init__(self, prev, next, user_ranks, index) -> None:
        self.prev: int = prev
        self.next: int = next
        self.user_ranks: int = user_ranks
        self.index: int = index

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"RingNode({self.prev}, {self.index}, {self.next})"


class TreeNode(object):
    size = 4 + 4 + 4 * NCCL_MAX_TREE_ARITY
    def __init__(self, depth, up, down) -> None:
        self.depth: int = depth
        self.up: int = up
        self.down: List[int] = down

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"TreeNode(d={self.depth}, up={self.up}, down={self.down})"


class Communicator(object):
    def __init__(self, id_hash, num_channels, last_ring_id, last_tree_id, comm_addr,
                 num_devices, global_rank, local_rank, group_rank, rings, trees):
        self.id_hash = id_hash
        self.num_channels = num_channels
        self.last_ring_id = last_ring_id
        self.last_tree_id = last_tree_id
        self.comm_addr = comm_addr
        self.num_devices = num_devices
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.group_rank = group_rank
        self.rings: List[RingNode] = rings
        self.trees: List[TreeNode] = trees
    
    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "Communicator(\n"
        s += f"  id: {self.id_hash}, num_channels: {self.num_channels}, global_rank: {self.global_rank},\n"
        s += f"  group_rank:{self.group_rank} local_rank:{self.local_rank}, num_devs:{self.num_devices},\n"
        s += f"  RingNodes:{self.rings[:self.num_channels]}\n"
        s += f"  TreeNodes:{self.trees[:self.num_channels]}\n)"
        return s

def deserialize_communicator_from_redis(redis_data) -> Communicator:
    pos = 0

    id_hash, num_channels, last_ring_id, last_tree_id, comm_addr, num_devices,\
        global_rank, local_rank, group_rank = struct.unpack('QQQQQiiii', redis_data[:56])
    pos += 56

    rings = []
    trees = []

    for _ in range(MAXCHANNELS):
        ring_data = struct.unpack('iiQii', redis_data[pos : pos + RingNode.size])
        rings.append(RingNode(*ring_data[:-1]))
        pos += RingNode.size
    
    for _ in range(MAXCHANNELS):
        depth, up, *down = struct.unpack('iiiii', redis_data[pos : pos + TreeNode.size])
        assert len(down) == NCCL_MAX_TREE_ARITY
        trees.append(TreeNode(depth, up, down))
        pos += TreeNode.size
    
    communicator = Communicator(id_hash, num_channels, last_ring_id, last_tree_id, comm_addr,
                                num_devices, global_rank, local_rank, group_rank, rings, trees)

    return communicator

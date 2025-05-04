from typing import List, Dict
from .communicator import Communicator, RingNode, TreeNode


def ring_key(ring: List[int]):
    return tuple(sorted(ring))


class GlobalTopo(object):
    def __init__(self, comms: List[Communicator]) -> None:
        self.comms = {c.comm_addr : c for c in comms}
        self.comms_by_id_hash = {}
        for comm in self.comms.values():
            if comm.id_hash in self.comms_by_id_hash:
                self.comms_by_id_hash[comm.id_hash].append(comm)
            else:
                self.comms_by_id_hash[comm.id_hash] = [comm]
        self.rings = self.build_rings(self.comms_by_id_hash)
        # FIXME: modify it to real trees
        self.trees = self.build_trees(self.comms_by_id_hash)

    def build_rings(self, comms_by_id_hash: Dict[int, List[Communicator]]):
        rings = []
        for (id_hash, comms) in comms_by_id_hash.items():
            comms_by_rank = {cm.group_rank: cm for cm in comms}
            # for ch in range(comms[0].num_channels):
            # Now always use the 0th channel because all channels are identical
            curr_ring = []
            visited = set()
            curr_node: RingNode = comms[0].rings[0]
            while len(visited) != len(comms):
                curr_ring.append(curr_node.index)
                visited.add(curr_node.index)
                curr_node = comms_by_rank[curr_node.next].rings[0]
            rings.append(curr_ring)
        
        unique_rings = {}
        final_rings = []
        for r in rings:
            ring_hash = ring_key(r)
            if ring_hash not in unique_rings:
                unique_rings[ring_hash] = True
                final_rings.append(r)
        return final_rings
        

    def build_trees(self, comms_by_id_hash: Dict[int, List[Communicator]]):
        trees = []
        for (id_hash, comms) in comms_by_id_hash.items():
            comms_by_rank = {cm.group_rank: cm for cm in comms}
            # for ch in range(comms[0].num_channels):
            # Now always use the 0th channel because all channels are identical
            root: Communicator = None
            for i in range(len(comms)):
                curr_comm: Communicator = comms[i]
                if curr_comm.trees[0].up == -1:
                    root = curr_comm
                    break
            # BFS to label the nodes by layer
            q = [(root, 0)]
            curr_tree_by_layer = {}
            while len(q) != 0:
                comm_node, lvl = q.pop(0)
                if lvl not in curr_tree_by_layer:
                    curr_tree_by_layer[lvl] = [comm_node]
                else:
                    curr_tree_by_layer[lvl].append(comm_node)
                left_child_rank, right_child_rank = comm_node.trees[0].down[0], comm_node.trees[0].down[1]
                if left_child_rank != -1:
                    q.append((comms_by_rank[left_child_rank], lvl + 1))
                if right_child_rank != -1:
                    q.append((comms_by_rank[right_child_rank], lvl + 1))
            trees.append(curr_tree_by_layer)
        return trees

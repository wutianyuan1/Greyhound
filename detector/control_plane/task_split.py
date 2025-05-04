from typing import List, Tuple, Dict
from .communicator import Communicator


def split_ring(ring: List[int]) -> List[Tuple[List[int]]]:
    if len(ring) == 2:
        first_send = [ring[0]]
        first_recv = [ring[1]]
        second_send = [ring[1]]
        second_recv = [ring[0]]
        return [(first_send, first_recv), (second_send, second_recv)]
    if len(ring) % 2 == 0:
        first_send = ring[0::2]
        first_recv = ring[1::2]
        second_send = ring[1::2]
        second_recv = ring[2:-1:2] + [ring[0]]
        return [(first_send, first_recv), (second_send, second_recv)]
    else:
        first_send = ring[0:-1:2]
        first_recv = ring[1::2]
        second_send = ring[1::2]
        second_recv = ring[2::2]
        third_send = [ring[-1]]
        third_recv = [ring[0]]
        return [(first_send, first_recv), (second_send, second_recv), (third_send, third_recv)]


def split_tree(tree: Dict[int, List[Communicator]]) -> List[Tuple[List[int]]]:
    total_layers = len(tree)
    first_send, second_send, first_recv, second_recv = [], [], [], []
    for i in range(0, total_layers, 2):
        # even layers send at the first and second tasks
        # 1st task: recvers are the left node
        if tree[i][0].trees[0].down[0] != -1:
            first_send.append(tree[i][0].group_rank)
            first_recv.append(tree[i][0].trees[0].down[0])
        # 2nd task: recvers are the right node
        if tree[i][0].trees[0].down[1] != -1:
            second_send.append(tree[i][0].group_rank)
            second_recv.append(tree[i][0].trees[0].down[1])

    third_send, third_recv, forth_send, forth_recv = [], [], [], []
    for i in range(1, total_layers, 2):
        # odd layers send at the 3rd and 4th tasks
        # 3rd task: recvers are the left node
        if tree[i][0].trees[0].down[0] != -1:
            third_send.append(tree[i][0].group_rank)
            third_recv.append(tree[i][0].trees[0].down[0])
        # 4th task: recvers are the right node
        if tree[i][0].trees[0].down[1] != -1:
            forth_send.append(tree[i][0].group_rank)
            forth_recv.append(tree[i][0].trees[0].down[1])
    return [(first_send, first_recv), (second_send, second_recv), (third_send, third_recv), (forth_send, forth_recv)]

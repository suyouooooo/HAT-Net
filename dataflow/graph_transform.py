import random

import torch
from torch_cluster import grid_cluster
from torch_scatter import scatter
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos


def _avg_pool_x(cluster, x, size=None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='mean')

def avg_pooling(data, size):
    cluster = grid_cluster(data.pos, torch.tensor([size, size])) # for crc 64 is 32
    cluster, perm = consecutive_cluster(cluster)
    x = None if data.x is None else _avg_pool_x(cluster, data.x)
    pos = None if data.pos is None else pool_pos(cluster, data.pos)
    data.x = x
    data.pos = pos
    #print(data.x.shape)
    #import sys; sys.exit()

    return data

def random_sample(data, sampled_nodes):
    num_nodes = data.x.shape[0]
    node_id = random.sample(range(num_nodes), k=sampled_nodes)
    data.x = data.x[node_id]
    data.pos = data.pos[node_id]
    #print(data)
    #import sys; sys.exit()

    return data

def dropnodes(data, ratio):
    length = len(data.x)
    sample = torch.rand(length).topk(int(length * (1 - ratio))).indices
    mask = torch.zeros(length, dtype=torch.bool)
    mask.scatter_(dim=0, index=sample, value=True)

    data.x = data.x[mask]
    data.pos = data.pos[mask]

    return data
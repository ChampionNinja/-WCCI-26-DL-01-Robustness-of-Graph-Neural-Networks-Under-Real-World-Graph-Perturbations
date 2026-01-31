import torch
from torch_geometric.utils import dropout_edge, degree

def random_removal(edge_index, p):
    new_edge_index, _ = dropout_edge(edge_index, p=p)
    return new_edge_index

def degree_aware_removal(edge_index, num_nodes, p):
    deg = degree(edge_index[0], num_nodes=num_nodes)
    threshold = torch.quantile(deg, 0.8)

    src, dst = edge_index
    mask = (deg[src] >= threshold) | (deg[dst] >= threshold)

    cand = edge_index[:, mask]
    kept, _ = dropout_edge(cand, p=p)

    non_cand = edge_index[:, ~mask]
    return torch.cat([kept, non_cand], dim=1)

def class_aware_removal(edge_index, y, p):
    src, dst = edge_index
    mask = (y[src] != y[dst])

    cand = edge_index[:, mask]
    kept, _ = dropout_edge(cand, p=p)

    non_cand = edge_index[:, ~mask]
    return torch.cat([kept, non_cand], dim=1)

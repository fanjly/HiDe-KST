import os
import math
import pickle
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from fastdtw import fastdtw
# from .utils import log_string
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L


def largest_k_lamb(L, k):
    n = L.shape[0]
    if k >= n:
        lamb, U = np.linalg.eigh(L)
        pad_size = k - n
        lamb = np.pad(lamb, (pad_size, 0), 'constant')
        U = np.pad(U, ((0, 0), (pad_size, 0)), 'constant')
    else:
        lamb, U = sp.linalg.eigsh(L, k=k, which="LM")
    return (lamb, U)


def get_eigv(adj, k):
    L = laplacian(adj)
    eig = largest_k_lamb(L, k)
    return eig


def construct_tem_adj(data, num_node, steps_per_day=288):
    # data: [T, N, D] or [T, N]
    if len(data.shape) == 3:
        data = data[..., 0]  # Use first feature for graph construction

    num_days = data.shape[0] // steps_per_day
    if num_days == 0:
        # Not enough data for one cycle, use all data
        data_mean = data.T
    else:
        data_mean = np.mean(
            [
                data[steps_per_day * i: steps_per_day * (i + 1)]
                for i in range(num_days)
            ],
            axis=0,
        )
        data_mean = data_mean.T  # [N, steps_per_day]

    dtw_distance = np.zeros((num_node, num_node))
    for i in tqdm(range(num_node)):
        for j in range(i, num_node):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
    for i in range(num_node):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]

    nth = np.sort(dtw_distance.reshape(-1))[
        int(np.log2(dtw_distance.shape[0]) * dtw_distance.shape[0]): int(
            np.log2(dtw_distance.shape[0]) * dtw_distance.shape[0]
        )
        + 1
    ]  # NlogN edges
    tem_matrix = np.zeros_like(dtw_distance)
    tem_matrix[dtw_distance <= nth] = 1
    tem_matrix = np.logical_or(tem_matrix, tem_matrix.T).astype(int)
    return tem_matrix


def loadGraph(spatial_graph, temporal_graph, dims, data, steps_per_day=288):
    def _load_adj_pkl(path: str) -> np.ndarray:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        adj = np.array(obj, dtype=np.float32)
        return adj

    # calculate spatial and temporal graph wavelets
    adj = _load_adj_pkl(spatial_graph)
    adj = adj + np.eye(adj.shape[0])
    if os.path.exists(temporal_graph):
        tem_adj = np.load(temporal_graph)
    else:
        tem_adj = construct_tem_adj(data, adj.shape[0], steps_per_day)
        np.save(temporal_graph, tem_adj)
    spawave = get_eigv(adj, dims)
    temwave = get_eigv(tem_adj, dims)
    # log_string(
    #     log,
    #     f"Shape of graphwave eigenvalue and eigenvector: {spawave[0].shape}, {spawave[1].shape}",
    # )

    # derive neighbors
    sampled_nodes_number = int(math.log(adj.shape[0], 2))
    graph = csr_matrix(adj)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix == 0] = dist_matrix.max() + 10
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[
        :, :sampled_nodes_number
    ]

    # log_string(log, f"Shape of localadj: {localadj.shape}")
    return localadj, spawave, temwave

"""Build the graph, semantic and pattern assets required by PDFormer."""

import argparse
import json
import os
import pickle

import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm


def load_series(path, variable=None):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == '.dat':
        with open(os.path.join(os.path.dirname(path), 'desc.json'), 'r') as file:
            shape = tuple(json.load(file)['shape'])
        values = np.memmap(path, dtype='float32', mode='r', shape=shape)
        return np.asarray(values[..., 0])
    if suffix == '.nc':
        if not variable:
            raise ValueError('--variable is required for NetCDF data')
        with nc.Dataset(path) as dataset:
            return np.asarray(dataset.variables[variable][:], dtype=np.float32)
    if suffix == '.csv':
        return pd.read_csv(path).iloc[:, 1:].to_numpy(dtype=np.float32)
    raise ValueError('Only .dat, .nc and .csv inputs are supported')


def load_adjacency(path):
    with open(path, 'rb') as file:
        return np.asarray(pickle.load(file), dtype=np.float32)


def calculate_hops(adjacency):
    graph = sp.csr_matrix((adjacency > 0).astype(np.float32))
    hops = shortest_path(graph, directed=False, unweighted=True)
    hops[np.isinf(hops)] = 511
    return hops.astype(np.float32)


def calculate_lape(adjacency, lape_dim):
    adjacency = sp.coo_matrix(adjacency)
    degree = np.asarray(adjacency.sum(1)).flatten()
    isolated = int(np.sum(degree == 0))
    degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    degree_inv_sqrt[degree == 0] = 0
    degree_matrix = sp.diags(degree_inv_sqrt)
    laplacian = sp.eye(adjacency.shape[0]) - degree_matrix @ adjacency @ degree_matrix
    eigenvalues, eigenvectors = np.linalg.eig(laplacian.toarray())
    order = eigenvalues.argsort()
    eigenvectors = np.real(eigenvectors[:, order])
    start = isolated + 1
    return eigenvectors[:, start:start + lape_dim].astype(np.float32)


def calculate_dtw(series, steps_per_day):
    try:
        from fastdtw import fastdtw
    except ImportError as error:
        raise RuntimeError('Install fastdtw before preprocessing PDFormer') from error
    complete_days = series.shape[0] // steps_per_day
    daily = series[:complete_days * steps_per_day].reshape(
        complete_days, steps_per_day, series.shape[1]
    ).mean(axis=0)
    distances = np.zeros((series.shape[1], series.shape[1]), dtype=np.float32)
    for i in tqdm(range(series.shape[1]), desc='DTW'):
        for j in range(i, series.shape[1]):
            distance, _ = fastdtw(daily[:, i], daily[:, j], radius=6)
            distances[i, j] = distances[j, i] = distance
    return distances


def calculate_pattern_keys(series, train_size, steps_per_day, pattern_len,
                           candidate_days, n_clusters, max_iter):
    try:
        from tslearn.clustering import KShape
    except ImportError as error:
        raise RuntimeError('Install tslearn before generating PDFormer pattern keys') from error
    train_end = int(round(series.shape[0] * train_size))
    train = series[:train_end]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True, ddof=1)
    std[std == 0] = 1
    train = (train - mean) / std
    candidate_steps = min(len(train), candidate_days * steps_per_day)
    candidates = np.stack([
        train[i:i + pattern_len]
        for i in range(candidate_steps - pattern_len + 1)
    ], axis=0)
    candidates = candidates.transpose(0, 2, 1).reshape(-1, pattern_len, 1)
    model = KShape(n_clusters=n_clusters, max_iter=max_iter, random_state=0)
    return model.fit(candidates).cluster_centers_.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--adj_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--variable')
    parser.add_argument('--steps_per_day', type=int, required=True)
    parser.add_argument('--train_size', type=float, required=True)
    parser.add_argument('--lape_dim', type=int, default=8)
    parser.add_argument('--pattern_len', type=int, default=3)
    parser.add_argument('--candidate_days', type=int, default=14)
    parser.add_argument('--n_clusters', type=int, default=16)
    parser.add_argument('--cluster_max_iter', type=int, default=5)
    parser.add_argument('--dtw_path')
    parser.add_argument('--pattern_keys_path')
    args = parser.parse_args()

    series = load_series(args.data_path, args.variable)
    adjacency = load_adjacency(args.adj_path)
    if adjacency.shape != (series.shape[1], series.shape[1]):
        raise ValueError('Adjacency and time-series node counts differ')
    train_end = int(round(series.shape[0] * args.train_size))
    dtw_matrix = (np.load(args.dtw_path) if args.dtw_path
                  else calculate_dtw(series[:train_end], args.steps_per_day))
    pattern_keys = (np.load(args.pattern_keys_path) if args.pattern_keys_path
                    else calculate_pattern_keys(
                        series, args.train_size, args.steps_per_day,
                        args.pattern_len, args.candidate_days,
                        args.n_clusters, args.cluster_max_iter
                    ))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.savez_compressed(
        args.output_path,
        dtw_matrix=dtw_matrix.astype(np.float32),
        pattern_keys=pattern_keys.astype(np.float32),
        hop_matrix=calculate_hops(adjacency),
        laplacian_pe=calculate_lape(adjacency, args.lape_dim),
    )
    print('Saved PDFormer assets to', args.output_path)
    print('dtw_matrix:', dtw_matrix.shape, 'pattern_keys:', pattern_keys.shape)


if __name__ == '__main__':
    main()

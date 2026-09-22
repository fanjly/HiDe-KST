import numpy as np
import scipy.sparse as sp
import csv
import pickle


def get_normalized_adj(path):
    """
    Returns the degree normalized adjacency matrix. 度归一化邻接矩阵
    """
    def _load_adj_pkl(path: str) -> np.ndarray:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        adj = np.array(obj, dtype=np.float32)
        return adj

    A = _load_adj_pkl(path)
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float64))

    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))

    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    del A, D, diag
    return A_wave


def get_adj_from_csv(path, num_of_vertices, id_filename=None):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, "r") as f:
            id_dict = {
                int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))
            }

        with open(path, "r") as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distaneA[id_dict[i], id_dict[j]] = distance
        return A, distaneA
    else:
        with open(path, "r") as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                # A[j,i] = 1
                distaneA[i, j] = distance
        return A, distaneA


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # print(f"Number of isolated points: {isolated_point_num}")
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = (
        sp.eye(adj.shape[0])
        - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )
    return normalized_laplacian, isolated_point_num


def cal_lape(adj_mx):
    lape_dim = 32
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]

    # Pad with zeros if we don't have enough eigenvectors
    if laplacian_pe.shape[1] < lape_dim:
        pad_size = lape_dim - laplacian_pe.shape[1]
        padding = np.zeros((laplacian_pe.shape[0], pad_size), dtype=laplacian_pe.dtype)
        laplacian_pe = np.concatenate([laplacian_pe, padding], axis=1)

    return laplacian_pe


def get_adj_from_npy(path):
    data = np.load(path)
    num_of_vertices = data.shape[0]
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    for i in range(num_of_vertices):
        for j in range(num_of_vertices):
            if i != j and data[i, j] != 0:
                A[i, j] = 1
    return A

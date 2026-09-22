import numpy as np
import scipy.sparse as sp


def calc_gso(dir_adj: np.ndarray, gso_type: str) -> np.ndarray:
    """ https://github.com/hazdzz/STGCN
    """
    n_vertex = dir_adj.shape[0]

    if not sp.issparse(dir_adj):
        dir_adj = sp.csc_matrix(dir_adj)  # 转换为稀疏格式
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)

    if gso_type in ('sym_renorm_adj', 'rw_renorm_adj', 'sym_renorm_lap', 'rw_renorm_lap'):
        adj = adj + id

    if gso_type in ('sym_norm_adj', 'sym_renorm_adj', 'sym_norm_lap', 'sym_renorm_lap'):
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type in ('rw_norm_adj', 'rw_renorm_adj', 'rw_norm_lap', 'rw_renorm_lap'):
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso.astype(np.float32)

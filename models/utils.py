import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
import torch

def normalized_laplacian(adjacency_matrix):
    # 将邻接矩阵转换为PyTorch张量
    A = adjacency_matrix
    # 计算度矩阵
    D = torch.diag(torch.sum(A, dim=1))
    # 计算拉普拉斯矩阵
    L = D - A
    # 计算度矩阵的逆平方根
    D_inv_sqrt = torch.inverse(torch.sqrt(D))
    # 计算标准化拉普拉斯矩阵
    normalized_laplacian_matrix = torch.eye(A.size(0), device=A.device) - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)

    return normalized_laplacian_matrix

def laplacian_energy_distribution(feature, adj_lap):
    values, U = torch.symeig(adj_lap, eigenvectors=True)
    _, eigen_indices = torch.sort(values)
    eigenvectors = U[:, eigen_indices]
    X_hat = torch.matmul(eigenvectors.t(), feature)
    led = (X_hat**2)/torch.sum(X_hat**2)
    return led, U

def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]

    return res

def linfball_proj(center, radius, t, in_place=True):

    #print(torch.max(center-t), torch.mean(torch.abs(center-t)), torch.min(center-t))
    return tensor_clamp(t, min=center-radius, max=center+radius, in_place=in_place)
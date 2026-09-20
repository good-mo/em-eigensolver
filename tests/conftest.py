"""测试辅助工具。"""

import numpy as np
import scipy.sparse as sp

from em_eigensolver.samples import generate_sample_matrix


def make_small_matrix(n: int = 100, seed: int = 42, type: str = "indefinite"):
    """生成小型测试矩阵（带已知特征值）。"""
    A, meta = generate_sample_matrix(type, n, sigma_offset=2.0, seed=seed)
    return A, np.array(meta["true_eigenvalues"])


def make_hermitian_positive(n: int = 100, seed: int = 42):
    """生成正定厄密矩阵（用于基础测试）。"""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    A = B @ B.T + n * sp.eye(n)
    return sp.csr_matrix(A)


def make_hermitian_indefinite(n: int = 100, seed: int = 42):
    """生成非正定厄密矩阵。"""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    A = B @ B.T
    # 减去一个大的对角项使谱跨越正负
    A = A - 0.5 * n * sp.eye(n)
    return sp.csr_matrix(A)


def residual_norm(A, eigenvalues, eigenvectors):
    """计算相对残差。"""
    k = len(eigenvalues)
    res = np.zeros(k)
    for i in range(k):
        v = eigenvectors[:, i]
        r = A @ v - eigenvalues[i] * v
        res[i] = np.linalg.norm(r) / max(np.linalg.norm(v), 1e-14)
    return res

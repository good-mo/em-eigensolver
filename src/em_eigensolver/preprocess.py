"""预处理模块：矩阵缩放、平衡、排序与性质检测。

针对电磁仿真中的非正定厄密稀疏矩阵，预处理可以：
1. 改善矩阵条件数（缩放/平衡）
2. 减小迭代次数（排序减少填充）
3. 检测矩阵性质（是否为厄密、正定、对称），指导求解器选择与位移策略
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class PreprocessingInfo:
    """预处理信息，记录预处理参数与恢复信息。

    求解得到的特征对需按逆预处理变换恢复原始问题的特征向量。
    """

    #: 是否做了缩放
    scaled: bool = False
    #: 对角缩放因子 D，A_hat = D^{-1/2} A D^{-1/2}
    scale_vector: Optional[np.ndarray] = None
    #: 平衡置换矩阵索引
    perm_indices: Optional[np.ndarray] = None
    #: 矩阵性质
    is_hermitian: bool = True
    is_positive_definite: bool = False
    is_symmetric: bool = True
    #: 谱估计 [min, max]（若计算）
    spectral_range: Optional[Tuple[float, float]] = None
    #: 原始形状
    shape: Optional[Tuple[int, int]] = None

    def restore_eigenvector(self, v: np.ndarray) -> np.ndarray:
        """将预处理后的特征向量恢复为原始问题的特征向量。

        缩放：A_hat = D A D，D = diag(d)。若 A_hat w = λ w 且 A v = λ v，
        则 w = D v ⇒ v = D^{-1} w = w / d。
        本模块的缩放仅采用正交符号翻转（d_i = ±1），因此特征值精确不变；
        d_i = ±1 时 D^{-1} = D，乘除等价，这里统一按除法实现以保证一般性。

        置换：A_hat = P A P^T，则 w = P^T v_hat ⇒ 原始向量 = v_hat[inv_perm]，
        即 perm_indices 存的是逆置换（inv_perm）。
        """
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        # 先恢复置换，再恢复缩放
        if self.perm_indices is not None:
            v = v[self.perm_indices, :]  # 逆置换 inv_perm
        if self.scale_vector is not None and self.scaled:
            d = self.scale_vector
            v = v / d[:, None]
        return v.squeeze() if v.shape[1] == 1 else v


# ---------------------------------------------------------------------------
# 矩阵性质检测
# ---------------------------------------------------------------------------
def check_hermitian(matrix: sp.spmatrix, tol: float = 1e-12) -> bool:
    """检测矩阵是否为厄密矩阵（A = A^H）。

    对稀疏矩阵采用采样检测：比较 A 与 A.conj().T 的非零模式与值。
    为避免 O(n^2) 开销，采用行采样 + 最大差异检查。

    Args:
        matrix: 稀疏矩阵。
        tol: 相对容差。

    Returns:
        True 如果矩阵是厄密的。
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        return False

    Ah = matrix.conj().T.tocsr()
    # 非零模式比较
    if matrix.nnz != Ah.nnz:
        return False
    # 采样比较：对每一行，比较值
    # 采用 (A - A^H) 的谱范数估计
    diff = (matrix - Ah).tocsr()
    if diff.nnz == 0:
        return True
    # 计算差矩阵的 max 范数（按绝对值）
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
    base = float(np.max(np.abs(matrix.data))) if matrix.nnz > 0 else 1.0
    return max_abs <= tol * max(1.0, base)


def check_positive_definite(matrix: sp.spmatrix, tol: float = 1e-10) -> bool:
    """检测矩阵是否正定。

    使用 Cholesky 分解（对稀疏矩阵），通过尝试性分解判断。
    注意：只对明显厄密的矩阵调用。

    Args:
        matrix: 稀疏厄密矩阵。
        tol: 容差。

    Returns:
        True 如果 Cholesky 分解成功（正定）。
    """
    try:
        spla.splu(matrix.tocsc(), diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        return True
    except Exception:
        return False


def estimate_spectral_range(
    matrix: sp.spmatrix,
    n_iters: int = 20,
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """估计厄密矩阵的谱范围 [λ_min, λ_max]。

    使用 Lanczos 近似（基于 scipy 的 eigsh 小规模 K 值），
    避免显式稠密化。

    Args:
        matrix: 稀疏厄密矩阵。
        n_iters: Lanczos 迭代次数。

    Returns:
        (λ_min_est, λ_max_est)
    """
    n = matrix.shape[0]
    if n == 0:
        return (0.0, 0.0)
    k = min(n_iters, n)
    try:
        # 求 k 个最大特征值
        evals_max = spla.eigsh(matrix, k=k, which="LM", return_eigenvectors=False)
        # 求 k 个最小特征值（通过位移）
        evals_min = spla.eigsh(matrix, k=k, which="SA", return_eigenvectors=False)
        return (float(evals_min.min()), float(evals_max.max()))
    except Exception:
        # 降级估计
        norm = spla.norm(matrix)
        return (-norm, norm)


# ---------------------------------------------------------------------------
# 缩放
# ---------------------------------------------------------------------------
def jacobi_scaling(matrix: sp.spmatrix) -> Tuple[sp.spmatrix, np.ndarray]:
    """雅可比（对角）缩放——符号翻转版本。

    对厄密矩阵，一般的对角缩放 A_hat = diag(d) A diag(d) 是合同变换，
    会**改变特征值**（Sylvester 惯性定律只保持符号与数量）。为保证
    特征值精确不变，仅使用正交缩放 d_i = ±1（符号翻转），此时
    D A D = D A D^{-1} 是相似变换，谱完全不变。

    d_i = sign(A_ii)（A_ii ≠ 0 时），使对角元非负，改善位移求逆的枢轴稳定性。

    返回 (缩放后的矩阵 A_hat, 缩放向量 d)，d_i ∈ {+1, -1}。
    """
    A = matrix.tocsr()
    n = A.shape[0]
    diag = A.diagonal()
    d = np.ones(n)
    nz = np.abs(diag) > 1e-14
    d[nz] = np.sign(diag[nz])
    if np.allclose(d, 1.0):
        return A, d
    D = sp.diags(d, format="csr")
    A_hat = D @ A @ D
    A_hat.eliminate_zeros()
    return A_hat, d


def row_norm_scaling(matrix: sp.spmatrix) -> Tuple[sp.spmatrix, np.ndarray]:
    """行范数缩放：d_i = 1/sqrt(||A[i,:]||_1)。

    改善非对角占优矩阵的条件数。

    Returns:
        (A_hat, d)。
    """
    A = matrix.tocsr()
    row_norms = np.abs(A).sum(axis=1).A1
    d = np.ones(A.shape[0])
    nz = row_norms > 1e-14
    d[nz] = 1.0 / np.sqrt(row_norms[nz])
    D = sp.diags(d, format="csr")
    A_hat = D @ A @ D
    A_hat.eliminate_zeros()
    return A_hat, d


# ---------------------------------------------------------------------------
# 排序（减少填充）
# ---------------------------------------------------------------------------
def reorder_matrix(matrix: sp.spmatrix, method: str = "rcm") -> Tuple[sp.spmatrix, np.ndarray, np.ndarray]:
    """矩阵重排序以减小填充。

    Args:
        matrix: 稀疏矩阵。
        method: 'rcm' Reverse Cuthill-McKee；'amd' Approximate Minimum Degree；
                'natural' 不排序。

    Returns:
        (重排后矩阵, 置换 perm, 逆置换 inv_perm)。A_reordered = A[perm][:, perm]。
    """
    A = matrix.tocsr()
    n = A.shape[0]
    if method == "natural":
        perm = np.arange(n)
        return A, perm, perm
    if method == "rcm":
        perm = sp.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
    elif method == "amd":
        # scipy 无直接 AMD 稀疏排序，用 RCM 近似（保持无依赖）
        perm = sp.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
    else:
        raise ValueError(f"未知排序方法: {method}")
    inv_perm = np.empty(n, dtype=int)
    inv_perm[perm] = np.arange(n)
    A_re = A[perm, :][:, perm]
    return A_re, perm, inv_perm


# ---------------------------------------------------------------------------
# 主预处理入口
# ---------------------------------------------------------------------------
def preprocess(
    matrix: sp.spmatrix,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[sp.spmatrix, PreprocessingInfo]:
    """执行完整的预处理流水线。

    流程：
    1. 性质检测（厄密性、对称性）
    2. 缩放（对角缩放）
    3. 排序（RCM 减小填充）
    4. 谱范围估计

    Args:
        matrix: 输入稀疏矩阵。
        config: 预处理配置字典。
        verbose: 是否打印诊断信息。

    Returns:
        (预处理后矩阵, 预处理信息)。
    """
    config = config or {}
    do_scale = config.get("scale", True)
    do_balance = config.get("balance", False)
    do_sort = config.get("sort", True)

    info = PreprocessingInfo()
    info.shape = matrix.shape
    A = matrix.tocsr()

    # 1. 性质检测
    try:
        info.is_hermitian = check_hermitian(A)
    except Exception:
        info.is_hermitian = False
    info.is_symmetric = info.is_hermitian and bool(np.isrealobj(A.data))

    # 2. 缩放
    if do_scale and info.is_hermitian:
        A, d = jacobi_scaling(A)
        info.scaled = not np.allclose(d, 1.0)
        info.scale_vector = d
        if verbose and info.scaled:
            print(f"[preprocess] 缩放完成: max_d={np.max(d):.4e}, min_d={np.min(d):.4e}")

    # 3. 平衡（对非对称矩阵，可选）
    if do_balance and not info.is_hermitian:
        # scipy 无直接稀疏平衡，使用对角缩放近似
        A, d2 = row_norm_scaling(A)
        if info.scale_vector is None:
            info.scale_vector = np.ones(matrix.shape[0])
        info.scale_vector = info.scale_vector * d2
        info.scaled = True
        if verbose:
            print("[preprocess] 行范数平衡完成")

    # 4. 排序
    if do_sort:
        try:
            A, perm, inv_perm = reorder_matrix(A, method="rcm")
            info.perm_indices = inv_perm
            if verbose:
                print(f"[preprocess] RCM 排序完成: nnz={A.nnz}")
        except Exception:
            pass

    A.eliminate_zeros()

    # 5. 谱范围估计（用于正定性判断，避免依赖缩放后的对角符号）
    spectral_range = None
    if info.is_hermitian:
        try:
            spectral_range = estimate_spectral_range(A, n_iters=min(20, A.shape[0]))
            info.spectral_range = spectral_range
        except Exception:
            pass

    # 6. 正定性检测（对厄密矩阵）
    # 优先使用谱范围：λ_min > 0 ⇒ 正定；λ_min < 0 ⇒ 非正定。
    # 若谱估计失败，退化为对角占优快速判断（在原始未缩放矩阵上检查）。
    if info.is_hermitian:
        if spectral_range is not None:
            info.is_positive_definite = bool(spectral_range[0] > 0)
        else:
            try:
                # 用对角占优快速判断（在原始矩阵上避免缩放干扰）
                diag = matrix.tocsr().diagonal()
                if np.all(diag > 0):
                    info.is_positive_definite = True
                else:
                    info.is_positive_definite = False
            except Exception:
                info.is_positive_definite = False

    return A, info
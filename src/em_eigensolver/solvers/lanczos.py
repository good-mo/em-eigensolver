"""Lanczos 求解器。

针对厄密矩阵的 Lanczos 算法，带完全重正交化（full reorthogonalization）以
保证数值稳定性。支持非正定厄密矩阵（特征值可为负）。

算法原理：
- 对厄密矩阵 A，Lanczos 迭代生成 Krylov 子空间 K_m(A, v1) 的一组正交基 V_m。
- 三对角矩阵 T_m = V_m^T A V_m 的特征值近似 A 的特征值。
- 完全重正交化：每一步将新向量与所有已有基向量正交化，避免丢失正交性。

复杂度：每次迭代 O(nnz + m*n)，内存 O(n*m)。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import EigenSolver


class LanczosSolver(EigenSolver):
    """Lanczos 特征值求解器（带完全重正交化）。

    适用于厄密矩阵（含非正定）。对目标频段，可通过位移求逆（shift-invert）
    加速收敛到目标特征值附近。
    """

    def __init__(self, config=None):
        super().__init__(config)
        # 最大 Krylov 子空间维度
        self._max_krylov = min(
            self.config.convergence.max_iter,
            max(2 * self.config.n_eigenvalues + 20, 40),
        )

    def _solve(
        self,
        A: sp.spmatrix,
        n_eigenvalues: int,
        sigma: Optional[float],
        which: str,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """Lanczos 核心算法。

        使用位移求逆（shift-invert）策略：求解 (A - σI)^{-1} 的特征值，
        其最大模特征值对应 A 中最接近 σ 的特征值。
        """
        n = A.shape[0]
        sigma_eff, which_eff = self._select_target(A, n_eigenvalues, sigma, which)

        # 位移求逆：构造线性算子 (A - σI)^{-1}
        if which_eff == "SM" or (sigma is not None and which == "target"):
            # 使用 shift-invert
            op = self._make_shift_invert_op(A, sigma_eff)
            use_shift_invert = True
        else:
            op = A
            use_shift_invert = False

        # 运行 Lanczos
        eigenvalues, eigenvectors, iterations = self._lanczos(
            A, op, n_eigenvalues, use_shift_invert, sigma_eff, which_eff
        )

        extra = {
            "method": "lanczos",
            "shift_invert": use_shift_invert,
            "sigma_effective": sigma_eff,
            "max_krylov": self._max_krylov,
        }
        return eigenvalues, eigenvectors, iterations, extra

    def _make_shift_invert_op(self, A: sp.spmatrix, sigma: float):
        """构造位移求逆线性算子 (A - σI)^{-1}。

        使用稀疏 LU 分解（splu）或迭代求解器。对非正定矩阵，LU 分解
        可能不稳定，需使用带枢轴的分解。
        """
        n = A.shape[0]
        M = (A - sigma * sp.eye(n)).tocsc()
        try:
            # 对非正定矩阵使用默认部分枢轴（partial pivoting），
            # 避免 SymmetricMode 在不定矩阵上不稳定。
            lu = spla.splu(M)
            return spla.LinearOperator(
                (n, n), matvec=lambda x: lu.solve(x), dtype=A.dtype
            )
        except Exception:
            # 降级：使用迭代求解器（GMRES）
            return spla.LinearOperator(
                (n, n),
                matvec=lambda x: spla.gmres(M, x, atol=1e-12, rtol=1e-12)[0],
                dtype=A.dtype,
            )

    def _lanczos(
        self,
        A: sp.spmatrix,
        op,
        n_eigenvalues: int,
        use_shift_invert: bool,
        sigma: float,
        which: str,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Lanczos 迭代（带完全重正交化）。

        Returns:
            (eigenvalues, eigenvectors, iterations)
        """
        n = A.shape[0]
        m = self._max_krylov
        rng = np.random.default_rng(self.config.seed)

        # 初始向量
        v = rng.standard_normal(n)
        v = v / np.linalg.norm(v)

        V = np.zeros((n, m))
        V[:, 0] = v
        alpha = np.zeros(m)
        beta = np.zeros(m)

        # 三对角矩阵 T 的构建
        for j in range(m):
            w = op @ V[:, j]
            alpha[j] = np.vdot(V[:, j], w).real
            w = w - alpha[j] * V[:, j]
            # 完全重正交化
            for i in range(j + 1):
                w = w - np.vdot(V[:, i], w) * V[:, i]
            # 二次重正交化（提高稳定性）
            for i in range(j + 1):
                w = w - np.vdot(V[:, i], w) * V[:, i]
            if j < m - 1:
                beta[j] = np.linalg.norm(w)
                if beta[j] < 1e-14:
                    # 子空间封闭，提前终止
                    m = j + 1
                    break
                V[:, j + 1] = w / beta[j]

        # 构建三对角矩阵 T
        T = np.zeros((m, m))
        for j in range(m):
            T[j, j] = alpha[j]
            if j > 0:
                T[j, j - 1] = beta[j - 1]
                T[j - 1, j] = beta[j - 1]

        # 求解 T 的特征值
        theta, Y = np.linalg.eigh(T)

        # 选择目标特征值
        if use_shift_invert:
            # 位移求逆：T 的特征值 theta 对应 (A-σI)^{-1} 的特征值，
            # 原特征值 λ = σ + 1/theta
            # 选择 |theta| 最大的（对应最接近 σ 的 λ）
            idx = np.argsort(np.abs(theta))[::-1]
        else:
            if which == "LM":
                idx = np.argsort(np.abs(theta))[::-1]
            elif which == "SM":
                idx = np.argsort(np.abs(theta))
            else:
                idx = np.argsort(theta)

        k = min(n_eigenvalues, m)
        idx = idx[:k]

        # 恢复特征值
        if use_shift_invert:
            eigenvalues = sigma + 1.0 / theta[idx]
        else:
            eigenvalues = theta[idx]

        # 恢复特征向量
        eigenvectors = V[:, :m] @ Y[:, idx]
        # 归一化
        for i in range(k):
            nv = np.linalg.norm(eigenvectors[:, i])
            if nv > 1e-14:
                eigenvectors[:, i] /= nv

        return eigenvalues, eigenvectors, m

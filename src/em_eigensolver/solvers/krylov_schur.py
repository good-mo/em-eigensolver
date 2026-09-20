"""Krylov-Schur 求解器。

Krylov-Schur 算法是 Lanczos/Arnoldi 的现代变体，通过隐式重启（implicit restart）
在保持 Krylov 子空间结构的同时，将不想要的 Ritz 值过滤掉，从而高效收敛到目标
特征值。相比标准 Lanczos，Krylov-Schur 具有更好的数值稳定性和更低的存储需求。

算法原理：
1. 运行 m 步 Lanczos/Arnoldi，得到三对角/上 Hessenberg 矩阵 T_m。
2. 对 T_m 做 Schur 分解，按目标特征值排序。
3. 保留前 k 个 Ritz 值对应的 Schur 向量，丢弃其余。
4. 用保留的 Schur 向量作为新的 Krylov 子空间起点，继续扩展。
5. 重复直到收敛。

复杂度：每次重启 O(m^3 + m*nnz)，内存 O(n*m)。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import EigenSolver


class KrylovSchurSolver(EigenSolver):
    """Krylov-Schur 特征值求解器（隐式重启）。

    适用于厄密矩阵（含非正定）。通过隐式重启高效收敛到目标频段特征值。
    """

    def __init__(self, config=None):
        super().__init__(config)
        # Krylov 子空间最大维度
        self._max_krylov = min(
            self.config.convergence.max_iter,
            max(2 * self.config.n_eigenvalues + 30, 60),
        )
        # 保留的 Ritz 值个数（重启时保留）
        self._keep = max(self.config.n_eigenvalues + 5, 10)

    def _solve(
        self,
        A: sp.spmatrix,
        n_eigenvalues: int,
        sigma: Optional[float],
        which: str,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """Krylov-Schur 核心算法。"""
        n = A.shape[0]
        sigma_eff, which_eff = self._select_target(A, n_eigenvalues, sigma, which)

        # 位移求逆
        use_shift_invert = (which_eff == "SM") or (sigma is not None and which == "target")
        if use_shift_invert:
            op = self._make_shift_invert_op(A, sigma_eff)
        else:
            op = A

        eigenvalues, eigenvectors, iterations = self._krylov_schur(
            A, op, n_eigenvalues, use_shift_invert, sigma_eff, which_eff
        )

        extra = {
            "method": "krylov_schur",
            "shift_invert": use_shift_invert,
            "sigma_effective": sigma_eff,
            "max_krylov": self._max_krylov,
            "keep": self._keep,
        }
        return eigenvalues, eigenvectors, iterations, extra

    def _make_shift_invert_op(self, A: sp.spmatrix, sigma: float):
        """构造位移求逆线性算子。"""
        n = A.shape[0]
        M = (A - sigma * sp.eye(n)).tocsc()
        try:
            # 对非正定矩阵使用默认部分枢轴（partial pivoting）
            lu = spla.splu(M)
            return spla.LinearOperator((n, n), matvec=lambda x: lu.solve(x), dtype=A.dtype)
        except Exception:
            return spla.LinearOperator(
                (n, n),
                matvec=lambda x: spla.gmres(M, x, atol=1e-12, rtol=1e-12)[0],
                dtype=A.dtype,
            )

    def _krylov_schur(
        self,
        A: sp.spmatrix,
        op,
        n_eigenvalues: int,
        use_shift_invert: bool,
        sigma: float,
        which: str,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Krylov-Schur 迭代（隐式重启）。

        Returns:
            (eigenvalues, eigenvectors, iterations)
        """
        n = A.shape[0]
        m = self._max_krylov
        keep = self._keep
        rng = np.random.default_rng(self.config.seed)

        # 初始向量
        v = rng.standard_normal(n)
        v = v / np.linalg.norm(v)

        # 存储 Krylov 基
        V = np.zeros((n, m))
        V[:, 0] = v
        # 三对角矩阵（厄密情况）
        alpha = np.zeros(m)
        beta = np.zeros(m)

        total_iter = 0
        converged = False
        eigenvalues = np.array([])
        eigenvectors = np.zeros((n, 0))

        # 重启循环
        for restart in range(self.config.convergence.max_restarts):
            # 扩展 Krylov 子空间
            start = 0
            # 如果已有保留向量，从保留向量开始
            if restart > 0 and V[:, 0].any():
                start = 0

            # 运行 Lanczos 到 m 步
            m_eff = self._run_lanczos(op, V, alpha, beta, m, start)
            total_iter += m_eff

            # 构建三对角矩阵
            T = np.zeros((m_eff, m_eff))
            for j in range(m_eff):
                T[j, j] = alpha[j]
                if j > 0:
                    T[j, j - 1] = beta[j - 1]
                    T[j - 1, j] = beta[j - 1]

            # 求解 T 的特征值
            theta, Y = np.linalg.eigh(T)

            # 选择目标特征值
            if use_shift_invert:
                idx = np.argsort(np.abs(theta))[::-1]
            else:
                if which == "LM":
                    idx = np.argsort(np.abs(theta))[::-1]
                elif which == "SM":
                    idx = np.argsort(np.abs(theta))
                else:
                    idx = np.argsort(theta)

            # 检查收敛
            k = min(n_eigenvalues, m_eff)
            sel = idx[:k]
            if use_shift_invert:
                eig_vals = sigma + 1.0 / theta[sel]
            else:
                eig_vals = theta[sel]

            # 计算残差估计（Ritz 残差）
            ritz_residuals = np.abs(beta[m_eff - 1] * Y[m_eff - 1, sel]) if m_eff > 1 else np.zeros(k)
            tol = self.config.convergence.tol
            if np.all(ritz_residuals <= tol * max(1.0, np.max(np.abs(eig_vals)))):
                converged = True
                # 恢复特征向量
                eig_vecs = V[:, :m_eff] @ Y[:, sel]
                for i in range(k):
                    nv = np.linalg.norm(eig_vecs[:, i])
                    if nv > 1e-14:
                        eig_vecs[:, i] /= nv
                eigenvalues = eig_vals
                eigenvectors = eig_vecs
                break

            # 隐式重启：保留前 keep 个 Ritz 向量
            keep_sel = idx[:keep]
            # 更新 V 为保留的 Ritz 向量
            V_new = V[:, :m_eff] @ Y[:, keep_sel]
            # 重新正交化
            Q, _ = np.linalg.qr(V_new)
            V[:, :keep] = Q[:, :keep]
            # 重置 alpha/beta 的前 keep 个（近似）
            # 用 Rayleigh-Ritz 重新计算
            for j in range(keep):
                w = op @ V[:, j]
                alpha[j] = np.vdot(V[:, j], w).real
                if j < keep - 1:
                    beta[j] = np.vdot(V[:, j + 1], w).real

        if not converged:
            # 未收敛，返回当前最佳估计
            k = min(n_eigenvalues, m_eff)
            sel = idx[:k]
            if use_shift_invert:
                eig_vals = sigma + 1.0 / theta[sel]
            else:
                eig_vals = theta[sel]
            eig_vecs = V[:, :m_eff] @ Y[:, sel]
            for i in range(k):
                nv = np.linalg.norm(eig_vecs[:, i])
                if nv > 1e-14:
                    eig_vecs[:, i] /= nv
            eigenvalues = eig_vals
            eigenvectors = eig_vecs

        return eigenvalues, eigenvectors, total_iter

    def _run_lanczos(
        self,
        op,
        V: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        m: int,
        start: int,
    ) -> int:
        """运行 Lanczos 迭代，扩展 Krylov 子空间。

        Returns:
            实际达到的维度。
        """
        n = V.shape[0]
        # 从 start 开始，V[:, start] 已存在
        j = start
        while j < m:
            w = op @ V[:, j]
            alpha[j] = np.vdot(V[:, j], w).real
            w = w - alpha[j] * V[:, j]
            # 完全重正交化
            for i in range(j + 1):
                w = w - np.vdot(V[:, i], w) * V[:, i]
            for i in range(j + 1):
                w = w - np.vdot(V[:, i], w) * V[:, i]
            if j < m - 1:
                beta[j] = np.linalg.norm(w)
                if beta[j] < 1e-14:
                    return j + 1
                V[:, j + 1] = w / beta[j]
            j += 1
        return m

"""Jacobi-Davidson 求解器。

Jacobi-Davidson (JD) 算法是现代迭代特征值求解器中最稳健的方法之一，
特别适合处理：
1. 目标位移 σ 接近特征值导致 A - σI 近奇异的情况
2. 求解目标频段内的内部特征值
3. 非正定/不定厄密矩阵

算法原理：
1. 构建搜索子空间 span(V)，通过 Rayleigh-Ritz 得到 Ritz 对（θ, u）。
2. 求解校正方程（correction equation）：
   (I - u u^H)(A - θ I)(I - u u^H) t = -r
   其中 r = Au - θu 为残差。
3. 将校正向量 t 加入搜索子空间，重复。
4. 关键优势：校正方程中 A - θI 在子空间 {u}^⊥ 上求解，
   即使 θ 接近特征值（A-θI 接近奇异），校正方程仍然良态。

复杂度：每次迭代 O(nnz + n*|V| + 校正求解)，内存 O(n*|V| + n*|W|)。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import EigenSolver


class JacobiDavidsonSolver(EigenSolver):
    """Jacobi-Davidson 特征值求解器。

    适用于近奇异位移场景（目标频率接近谐振点）、内部特征值求解。
    """

    def __init__(self, config=None):
        super().__init__(config)
        # 搜索子空间最大维度
        self._max_subspace = min(
            self.config.convergence.max_iter,
            max(2 * self.config.n_eigenvalues + 20, 30),
        )
        # 校正方程迭代方法：'minres' | 'gmres' | 'cg'
        self._correction_solver = "minres"
        # 校正方程迭代容差
        self._inner_tol = 0.1

    def _solve(
        self,
        A: sp.spmatrix,
        n_eigenvalues: int,
        sigma: Optional[float],
        which: str,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """Jacobi-Davidson 核心算法。"""
        n = A.shape[0]
        sigma_eff, which_eff = self._select_target(A, n_eigenvalues, sigma, which)

        eigenvalues, eigenvectors, iterations = self._jacobi_davidson(
            A, n_eigenvalues, sigma_eff, which_eff
        )

        extra = {
            "method": "jacobi_davidson",
            "sigma_effective": sigma_eff,
            "correction_solver": self._correction_solver,
            "max_subspace": self._max_subspace,
        }
        return eigenvalues, eigenvectors, iterations, extra

    def _jacobi_davidson(
        self,
        A: sp.spmatrix,
        n_eigenvalues: int,
        sigma: float,
        which: str,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Jacobi-Davidson 迭代主循环。

        Returns:
            (eigenvalues, eigenvectors, iterations)
        """
        n = A.shape[0]
        rng = np.random.default_rng(self.config.seed)

        # 收敛的特征值/特征向量
        converged_vals: List[float] = []
        converged_vecs: List[np.ndarray] = []
        max_iter = self.config.convergence.max_iter
        tol = self.config.convergence.tol

        # 目标：求解最接近 sigma 的 n_eigenvalues 个特征值
        # 通过锁定（lock）已收敛特征向量，逐次求解
        target = sigma if sigma is not None else 0.0

        total_iter = 0
        # 已锁定（收敛）的特征向量正交基
        locked: List[np.ndarray] = []

        # 每个特征值的具体求解：使用 deflation
        for _ in range(n_eigenvalues):
            # 初始化搜索子空间（与锁定向量正交）
            v = rng.standard_normal(n)
            v = self._orthonormalize_against(v, locked)
            if np.linalg.norm(v) < 1e-14:
                break
            v = v / np.linalg.norm(v)
            V = [v]  # 搜索子空间基

            # 当前目标值（对第一个特征值用 sigma，之后用上一次的特征值附近）
            # 使用 locking: 求解第 i 个特征值时，将前 i-1 个锁定
            iters = 0
            theta = 0.0
            u = v.copy()
            r_norm = 1.0

            while iters < max_iter and r_norm > tol * max(1.0, abs(theta)):
                iters += 1
                total_iter += 1

                # 构建 V 矩阵
                V_mat = np.column_stack(V)
                # Rayleigh-Ritz：小矩阵投影
                AV = A @ V_mat
                H = V_mat.T.conj() @ AV
                # 确保 H 为厄密
                H = 0.5 * (H + H.conj().T)

                # 求解小特征值问题
                try:
                    theta_all, y_all = np.linalg.eigh(H)
                except np.linalg.LinAlgError:
                    break

                # 选择最接近 target 的 Ritz 对
                dist = np.abs(theta_all - target)
                j = int(np.argmin(dist))
                theta = float(theta_all[j])
                y = y_all[:, j]
                u = V_mat @ y
                u = u / np.linalg.norm(u)

                # 残差
                r = A @ u - theta * u
                # 去除已锁定方向
                r = self._orthogonalize(r, locked + [v for v in V])

                r_norm = np.linalg.norm(r)

                if r_norm <= tol * max(1.0, abs(theta)):
                    # 收敛
                    converged_vals.append(theta)
                    converged_vecs.append(u.copy())
                    locked.append(u.copy())
                    break

                # 求解校正方程
                t = self._solve_correction(A, u, theta, r, locked)
                t = self._orthonormalize_against(t, locked + list(V))
                if np.linalg.norm(t) < 1e-14:
                    # 校正失败，用随机扰动
                    t = rng.standard_normal(n)
                    t = self._orthonormalize_against(t, locked + list(V))
                    if np.linalg.norm(t) < 1e-14:
                        break
                t = t / np.linalg.norm(t)

                # 扩展搜索子空间（限制维度，超限时重启）
                V.append(t)
                if len(V) > self._max_subspace:
                    # 保留最优的 Ritz 向量
                    V = self._restart_subspace(V_mat, y_all, theta_all, target, locked)

            if not converged_vals or len(converged_vals) < len(converged_vecs) + 1:
                # 未收敛但仍有 best 估计
                if r_norm <= tol * max(1.0, abs(theta)):
                    if len(converged_vals) == len(converged_vecs):
                        converged_vals.append(theta)
                        converged_vecs.append(u.copy())
                        locked.append(u.copy())

        # 组装结果
        k = len(converged_vals)
        if k == 0:
            return np.array([]), np.zeros((n, 0)), total_iter

        eigenvalues = np.array(converged_vals)
        eigenvectors = np.column_stack(converged_vecs)
        # 归一化
        for i in range(k):
            nv = np.linalg.norm(eigenvectors[:, i])
            if nv > 1e-14:
                eigenvectors[:, i] /= nv

        return eigenvalues, eigenvectors, total_iter

    def _solve_correction(
        self,
        A: sp.spmatrix,
        u: np.ndarray,
        theta: float,
        r: np.ndarray,
        locked: List[np.ndarray],
    ) -> np.ndarray:
        """求解 Jacobi-Davidson 校正方程。

        求解 (I - u u^H)(A - θI)(I - u u^H) t = -r，t ⊥ u。
        使用 MINRES 迭代求解器。由于投影到 {u}^⊥，即使 A-θI 接近奇异，
        校正方程仍可稳定求解。

        Args:
            A: 稀疏矩阵。
            u: 当前 Ritz 向量。
            theta: 当前 Ritz 值。
            r: 残差。
            locked: 已锁定向量。

        Returns:
            校正向量 t。
        """
        n = A.shape[0]
        tol = self._inner_tol

        # 构造投影算子 matvec
        def matvec(x):
            # 投影 x 到 {u}^⊥：x - (u^H x) u
            xp = x - np.vdot(u, x) * u
            # 应用 (A - θI)
            Axp = A @ xp - theta * xp
            # 再投影：去除 u 分量
            y = Axp - np.vdot(u, Axp) * u
            return y

        # 预条件：对角近似（对投影后的系统）
        def precond(x):
            diag = A.diagonal()
            d = diag - theta
            # 避免除零
            safe_d = np.where(np.abs(d) < 1e-14, 1.0, d)
            y = x / safe_d
            return y - np.vdot(u, y) * u

        # 迭代求解
        rhs = -r
        # 去除 u 分量
        rhs = rhs - np.vdot(u, rhs) * u

        try:
            # scipy 1.17 的迭代求解器要求 A 与 M 均为 LinearOperator 或稀疏矩阵，
            # 裸函数会触发 "type not understood"。
            op = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
            mop = spla.LinearOperator((n, n), matvec=precond, dtype=float)
            if self._correction_solver == "minres":
                t, info = spla.minres(
                    op, rhs, x0=np.zeros(n),
                    rtol=tol, maxiter=self.config.convergence.max_inner_iter,
                    M=mop,
                )
            elif self._correction_solver == "gmres":
                t, info = spla.gmres(
                    op, rhs, x0=np.zeros(n),
                    rtol=tol, maxiter=self.config.convergence.max_inner_iter,
                    M=mop,
                )
            else:
                t, info = spla.cg(
                    op, rhs, x0=np.zeros(n),
                    rtol=tol, maxiter=self.config.convergence.max_inner_iter,
                    M=mop,
                )
            # 微调：去除锁定方向
            t = self._orthogonalize(t, locked)
            t = self._orthogonalize(t, [u])
            return t
        except Exception:
            return np.zeros(n)

    def _restart_subspace(
        self,
        V_mat: np.ndarray,
        y_all: np.ndarray,
        theta_all: np.ndarray,
        target: float,
        locked: List[np.ndarray],
    ) -> List[np.ndarray]:
        """子空间重启：保留最接近 target 的 Ritz 向量。"""
        dist = np.abs(theta_all - target)
        keep = int(self._max_subspace * 0.5)
        keep = max(keep, 2)
        idx = np.argsort(dist)[:keep]
        new_V = []
        for j in idx:
            v = V_mat @ y_all[:, j]
            v = self._orthonormalize_against(v, locked + new_V)
            if np.linalg.norm(v) > 1e-10:
                new_V.append(v / np.linalg.norm(v))
        if not new_V:
            new_V = [V_mat[:, 0].copy()]
        return new_V

    def _orthonormalize_against(
        self, v: np.ndarray, basis: List[np.ndarray]
    ) -> np.ndarray:
        """与正交基做正交化。"""
        for b in basis:
            v = v - np.vdot(b, v) * b
        return v

    def _orthogonalize(self, v: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
        """与任意向量集合做正交化（去除分量）。"""
        for b in basis:
            b = b / np.linalg.norm(b)
            v = v - np.vdot(b, v) * b
        return v
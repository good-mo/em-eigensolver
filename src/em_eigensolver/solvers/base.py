"""求解器基础类：定义统一的求解器接口与结果数据结构。

所有求解器（Lanczos、Krylov-Schur、Jacobi-Davidson、LOBPCG）都继承自
EigenSolver，实现统一的 solve() 接口，返回 EigenResult。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..config import SolverConfig
from ..preprocess import PreprocessingInfo


@dataclass
class EigenResult:
    """特征值求解结果。

    Attributes:
        eigenvalues: 特征值数组 (k,)。
        eigenvectors: 特征向量矩阵 (n, k)，列归一化。
        residuals: 每个特征对的相对残差 (k,)。
        converged: 是否全部收敛。
        n_iterations: 总迭代次数。
        n_matvecs: 矩阵-向量乘法次数。
        elapsed_time: 求解耗时（秒）。
        info: 附加诊断信息。
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    residuals: np.ndarray
    converged: bool
    n_iterations: int
    n_matvecs: int
    elapsed_time: float
    info: Dict[str, Any] = field(default_factory=dict)

    def sort(self, order: str = "ascending") -> "EigenResult":
        """按特征值排序。"""
        idx = np.argsort(self.eigenvalues)
        if order == "descending":
            idx = idx[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        self.residuals = self.residuals[idx]
        return self

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "eigenvectors": self.eigenvectors.tolist(),
            "residuals": self.residuals.tolist(),
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "n_matvecs": self.n_matvecs,
            "elapsed_time": self.elapsed_time,
            "info": self.info,
        }


class EigenSolver(ABC):
    """特征值求解器抽象基类。

    子类需实现 _solve() 方法。solve() 负责：
    1. 预处理（缩放、排序）
    2. 调用子类核心算法
    3. 恢复原始特征向量
    4. 计算残差与收敛状态
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.config.validate()
        self._n_matvecs = 0
        self._preprocess_info: Optional[PreprocessingInfo] = None

    # ------------------------------------------------------------------
    # 矩阵-向量乘法（带计数）
    # ------------------------------------------------------------------
    def _matvec(self, A: sp.spmatrix, x: np.ndarray) -> np.ndarray:
        """执行矩阵-向量乘法并计数。"""
        self._n_matvecs += 1
        return A @ x

    def _matvec_shift(self, A: sp.spmatrix, x: np.ndarray, sigma: float) -> np.ndarray:
        """执行 (A - σI) x 乘法。"""
        self._n_matvecs += 1
        return A @ x - sigma * x

    # ------------------------------------------------------------------
    # 残差计算
    # ------------------------------------------------------------------
    def _compute_residuals(
        self,
        A: sp.spmatrix,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> np.ndarray:
        """计算相对残差 ||A v - λ v|| / (||A|| ||v||)。

        Args:
            A: 稀疏矩阵。
            eigenvalues: 特征值数组 (k,)。
            eigenvectors: 特征向量矩阵 (n, k)。

        Returns:
            相对残差数组 (k,)。
        """
        n = A.shape[0]
        k = len(eigenvalues)
        residuals = np.zeros(k)
        # 估计 ||A|| 的谱范数（用幂迭代或 norm）
        try:
            norm_A = spla.norm(A, ord=2) if n < 5000 else spla.norm(A, ord=1)
        except Exception:
            norm_A = float(np.max(np.abs(A.data))) if A.nnz > 0 else 1.0
        if norm_A == 0:
            norm_A = 1.0
        for i in range(k):
            v = eigenvectors[:, i]
            Av = A @ v
            r = Av - eigenvalues[i] * v
            nv = np.linalg.norm(v)
            residuals[i] = np.linalg.norm(r) / (norm_A * max(nv, 1e-14))
        return residuals

    # ------------------------------------------------------------------
    # 主求解入口
    # ------------------------------------------------------------------
    def solve(
        self,
        matrix: sp.spmatrix,
        n_eigenvalues: Optional[int] = None,
        sigma: Optional[float] = None,
        which: Optional[str] = None,
    ) -> EigenResult:
        """求解特征值问题。

        Args:
            matrix: 输入稀疏矩阵（厄密或非正定厄密）。
            n_eigenvalues: 需要求解的特征值个数。
            sigma: 目标位移（目标频率）。
            which: 目标类型。

        Returns:
            EigenResult。
        """
        start = time.time()
        n_eigenvalues = n_eigenvalues or self.config.n_eigenvalues
        which = which or self.config.which

        # 预处理
        A, info = self._preprocess(matrix)
        self._preprocess_info = info

        # 调用子类核心算法
        eigenvalues, eigenvectors, iterations, extra = self._solve(
            A, n_eigenvalues, sigma, which
        )

        # 恢复原始特征向量
        eigenvectors = info.restore_eigenvector(eigenvectors)

        # 计算残差（在原始矩阵上）
        residuals = self._compute_residuals(matrix, eigenvalues, eigenvectors)

        # 收敛判断
        tol = self.config.convergence.tol
        converged = bool(np.all(residuals <= tol)) if len(residuals) > 0 else False

        elapsed = time.time() - start
        result = EigenResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            residuals=residuals,
            converged=converged,
            n_iterations=iterations,
            n_matvecs=self._n_matvecs,
            elapsed_time=elapsed,
            info={
                "solver": self.__class__.__name__,
                "which": which,
                "sigma": sigma,
                "preprocess": {
                    "is_hermitian": info.is_hermitian,
                    "is_positive_definite": info.is_positive_definite,
                    "scaled": info.scaled,
                    "spectral_range": info.spectral_range,
                },
                **extra,
            },
        )
        return result

    # ------------------------------------------------------------------
    # 预处理封装
    # ------------------------------------------------------------------
    def _preprocess(self, matrix: sp.spmatrix) -> Tuple[sp.spmatrix, PreprocessingInfo]:
        """执行预处理。"""
        from ..preprocess import preprocess

        A, info = preprocess(
            matrix,
            config=self.config.preprocessing,
            verbose=self.config.verbose,
        )
        return A, info

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------
    @abstractmethod
    def _solve(
        self,
        A: sp.spmatrix,
        n_eigenvalues: int,
        sigma: Optional[float],
        which: str,
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """核心求解算法。

        Returns:
            (eigenvalues, eigenvectors, iterations, extra_info)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _select_target(
        self,
        A: sp.spmatrix,
        n_eigenvalues: int,
        sigma: Optional[float],
        which: str,
    ) -> Tuple[float, str]:
        """根据 which 与 sigma 确定位移与目标。

        Returns:
            (sigma_effective, which_effective)
        """
        if which == "target":
            if sigma is None:
                # 自动选择谱中心
                if self._preprocess_info and self._preprocess_info.spectral_range:
                    lo, hi = self._preprocess_info.spectral_range
                    sigma = 0.5 * (lo + hi)
                else:
                    sigma = 0.0
            return sigma, "LM"  # 位移求逆后取最大模
        if which == "smallest_magnitude":
            return 0.0, "SM"
        if which == "largest_magnitude":
            return 0.0, "LM"
        if which == "closest":
            if sigma is None:
                sigma = 0.0
            return sigma, "LM"
        return sigma, "LM"

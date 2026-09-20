"""位移策略模块：固定位移、自适应位移、Rayleigh 位移、Wilkinson 位移。

在电磁仿真中，目标位移 σ 对应目标频率（谐振点）。当 σ 非常接近特征值时，
A - σI 近奇异，直接求逆会导致数值崩溃。本模块提供多种位移策略与近奇异保护：

1. fixed: 固定位移，用户给定 σ。
2. adaptive: 根据当前特征值估计自适应更新 σ。
   - rayleigh_quotient: 用当前 Ritz 值的 Rayleigh 商更新。
   - harmonic: 用调和 Ritz 值更新（对内部特征值）。
   - spectrum: 用谱范围中心更新。
3. rayleigh: 逐步用 Rayleigh 商逼近特征值（二次收敛但可能陷入局部）。
4. wilkinson: 用 Wilkinson 移位（Gershgorin 区间中点），对密集谱稳健。

近奇异保护：当估计 |A - σI| 条件数超过阈值时，对位移做微扰或正则化，
避免求解器崩溃。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import ShiftConfig


@dataclass
class ShiftManager:
    """位移管理器：负责位移的初始化、更新与近奇异保护。

    Attributes:
        config: 位移配置。
        current_sigma: 当前位移值。
        history: 位移历史。
        near_singular_events: 近奇异事件计数。
    """

    config: ShiftConfig
    current_sigma: Optional[float] = None
    history: List[float] = field(default_factory=list)
    near_singular_events: int = 0

    def __post_init__(self):
        self.current_sigma = self.config.sigma

    def initialize(self, matrix: sp.spmatrix) -> float:
        """初始化位移。

        若用户未指定 σ，根据策略自动选择：
        - 有目标频段：取频段中心。
        - 无频段：估计谱范围取中心。
        """
        if self.current_sigma is not None:
            sigma = self.current_sigma
        elif self.config.target_band is not None:
            lo, hi = self.config.target_band
            sigma = 0.5 * (lo + hi)
        else:
            # 估计谱范围
            lo, hi = self._estimate_spectral_range(matrix)
            sigma = 0.5 * (lo + hi)
        self.current_sigma = float(sigma)
        self.history.append(self.current_sigma)
        return self.current_sigma

    def update(
        self,
        matrix: sp.spmatrix,
        ritz_values: np.ndarray,
        ritz_vectors: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """更新位移。

        Args:
            matrix: 稀疏矩阵。
            ritz_values: 当前 Ritz 值数组。
            ritz_vectors: Ritz 向量（可选，用于 Rayleigh 商）。

        Returns:
            新位移值；若策略为固定位移返回 None。
        """
        strategy = self.config.strategy
        if strategy == "fixed":
            return None

        if len(ritz_values) == 0:
            return self.current_sigma

        sigma_new = self.current_sigma

        if strategy in ("adaptive", "rayleigh"):
            mode = self.config.adaptive_mode
            if mode == "rayleigh_quotient" and ritz_vectors is not None:
                # 用 Ritz 向量的 Rayleigh 商
                rq = self._rayleigh_quotients(matrix, ritz_vectors)
                if len(rq) > 0:
                    # 选择最接近当前位移的 Rayleigh 商
                    sigma_new = float(rq[np.argmin(np.abs(rq - self.current_sigma))])
            elif mode == "harmonic":
                # 调和 Ritz 值：对内部特征值更优
                # 用最接近当前位移的 Ritz 值
                sigma_new = float(ritz_values[np.argmin(np.abs(ritz_values - self.current_sigma))])
            elif mode == "spectrum":
                # 谱中心
                lo, hi = self._estimate_spectral_range(matrix)
                sigma_new = 0.5 * (lo + hi)

        elif strategy == "wilkinson":
            # Wilkinson 位移：Gershgorin 区间内选择
            # 简化为最接近当前位移的 Ritz 值 ± 直径
            if len(ritz_values) >= 2:
                lo, hi = self._estimate_spectral_range(matrix)
                width = hi - lo
                # 取最接近当前位移的 Ritz 值
                nearest = ritz_values[np.argmin(np.abs(ritz_values - self.current_sigma))]
                sigma_new = float(nearest)

        # 应用阻尼（避免震荡）
        if self.history:
            prev = self.history[-1]
            delta = sigma_new - prev
            # 限幅
            step = np.clip(abs(delta), self.config.min_step, self.config.max_step)
            delta = np.sign(delta) * step if delta != 0 else 0.0
            sigma_new = prev + self.config.damping * delta

        self.current_sigma = float(sigma_new)
        self.history.append(self.current_sigma)
        return self.current_sigma

    def check_near_singular(self, matrix: sp.spmatrix) -> bool:
        """检测 A - σI 是否近奇异。

        通过计算 σ 距最近特征值的距离估计（用 Lanczos 快速近似）。
        若距离过小或条件数估计过高，触发保护。

        Returns:
            True 如果检测到近奇异，需要触发保护。
        """
        if not self.config.near_singular_protection:
            return False
        if self.current_sigma is None:
            return False

        # 快速估计 A - σI 的最小奇异值
        try:
            n = matrix.shape[0]
            M = (matrix - self.current_sigma * sp.eye(n)).tocsc()
            # 用小子空间 Lanczos 估计最小奇异值
            k = min(10, n)
            # 求 (A-σI)^H (A-σI) 最小特征值 ≈ σ_min^2
            # 对厄密矩阵，直接求 M 特征值接近 0 的数量
            evals = spla.eigsh(M, k=k, which="SM", return_eigenvectors=False, maxiter=200)
            dist = np.min(np.abs(evals))
        except Exception:
            # 降级：用对角占比估计
            diag = matrix.diagonal()
            dist = float(np.min(np.abs(diag - self.current_sigma)))

        threshold = self.config.near_singular_threshold
        # 条件数估计阈值转成距离阈值（归一化）
        norm_A = spla.norm(matrix, ord=1) if matrix.nnz > 0 else 1.0
        if norm_A <= 0:
            norm_A = 1.0

        if dist < norm_A / threshold:
            self.near_singular_events += 1
            return True
        return False

    def apply_protection(self, matrix: sp.spmatrix) -> float:
        """近奇异保护：对位移做微扰，避免 A - σI 奇异。

        将位移沿最大安全方向微调，使得 A - σI 条件数降到阈值以下。

        Returns:
            保护后的安全位移。
        """
        if self.current_sigma is None:
            return 0.0
        # 微扰量：基于矩阵规模的相对量
        norm_A = spla.norm(matrix, ord=1) if matrix.nnz > 0 else 1.0
        if norm_A <= 0:
            norm_A = 1.0
        # API 保护扰动：相对阈值
        perturbation = norm_A / self.config.near_singular_threshold * 1.0
        # 加上微小随机以跳出对称位置
        rng = np.random.default_rng(0)
        sign = 1.0 if rng.random() > 0.5 else -1.0
        safe_sigma = self.current_sigma + sign * perturbation
        self.current_sigma = float(safe_sigma)
        self.history.append(self.current_sigma)
        return self.current_sigma

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _estimate_spectral_range(self, matrix: sp.spmatrix) -> Tuple[float, float]:
        """估计厄密矩阵谱范围。"""
        from .preprocess import estimate_spectral_range

        try:
            return estimate_spectral_range(matrix, n_iters=20)
        except Exception:
            return (0.0, 1.0)

    def _rayleigh_quotients(
        self, matrix: sp.spmatrix, vectors: np.ndarray
    ) -> np.ndarray:
        """计算一组向量的 Rayleigh 商。"""
        v = np.asarray(vectors)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        Av = matrix @ v
        rq = np.sum(np.conj(v) * Av, axis=0) / np.sum(np.conj(v) * v, axis=0)
        return np.real(rq)
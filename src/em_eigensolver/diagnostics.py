"""收敛控制与故障诊断模块。

覆盖电磁仿真中的典型故障场景：
1. 零附近密集谱（低频击穿）：谱在零附近高密度聚集。
2. 重特征值（多重特征值）：多个相同/接近的特征值。
3. 近奇异位移（目标频率非常接近谐振点）：A - σI 条件数极大。
4. 内存受限：峰值内存控制。

本模块提供：
- 收敛监控器（ConvergenceMonitor）：跟踪残差/特征值变化，检测停滞。
- 故障诊断（Diagnostics）：分析矩阵谱分布、检测密集谱、估计内存需求。
- 处置建议（recommendations）：针对每类故障给出处理建议。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# 收敛监控器
# ---------------------------------------------------------------------------
@dataclass
class ConvergenceMonitor:
    """跟踪迭代收敛状态，检测停滞。

    Attributes:
        window: 停滞检测窗口。
        tol: 残差阈值。
        tolerance_relative: 相对特征值变化阈值。
    """

    window: int = 10
    tol: float = 1e-8
    tolerance_relative: float = 1e-10

    residuals_history: List[float] = field(default_factory=list)
    eigenvalue_history: List[np.ndarray] = field(default_factory=list)
    iterations: int = 0
    is_stagnant: bool = False
    best_residual: float = float("inf")

    def update(self, residual: float, eigenvalues: Optional[np.ndarray] = None) -> bool:
        """记录本轮收敛状态。

        Args:
            residual: 当前最大残差。
            eigenvalues: 当前特征值估计。

        Returns:
            True 如果已收敛。
        """
        self.iterations += 1
        self.residuals_history.append(float(residual))
        self.best_residual = min(self.best_residual, float(residual))
        if eigenvalues is not None:
            self.eigenvalue_history.append(np.asarray(eigenvalues).copy())

        # 收敛判断
        if residual <= self.tol:
            return True

        # 停滞检测
        if len(self.residuals_history) >= self.window:
            recent = self.residuals_history[-self.window :]
            first, last = recent[0], recent[-1]
            if first > 0 and last / first > 1 - 1e-3 and last > self.tol * 10:
                self.is_stagnant = True

        return False

    def convergence_rate(self) -> float:
        """估计收敛率（每步残差缩减比）。"""
        if len(self.residuals_history) < 2:
            return 1.0
        r = np.array(self.residuals_history)
        # 用线性回归拟合 log 残差斜率
        try:
            x = np.arange(len(r))
            slope = np.polyfit(x, np.log(np.maximum(r, 1e-300)), 1)[0]
            return float(np.exp(slope))
        except Exception:
            return 1.0

    def is_converging(self) -> bool:
        """是否在收敛（残差单调下降）。"""
        if len(self.residuals_history) < 5:
            return True
        recent = self.residuals_history[-5:]
        return recent[-1] < recent[0]

    def diagnose(self) -> Dict[str, Any]:
        """生成收敛诊断报告。"""
        return {
            "iterations": self.iterations,
            "best_residual": self.best_residual,
            "is_stagnant": self.is_stagnant,
            "is_converging": self.is_converging(),
            "convergence_rate": self.convergence_rate(),
        }


# ---------------------------------------------------------------------------
# 谱分布分析
# ---------------------------------------------------------------------------
def analyze_spectrum(
    matrix: sp.spmatrix,
    n_samples: int = 20,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """分析厄密矩阵谱分布特征。

    通过少量 Lanczos 迭代估计谱分布，检测零附近密集谱。

    Returns:
        谱分析字典：
        - spectral_range: (min, max)
        - density_near_zero: 零附近特征值密度估计
        - eigenvalue_estimate: 采样特征值
        - degeneracy_hint: 重特征值线索
    """
    n = matrix.shape[0]
    if n == 0:
        return {"spectral_range": (0, 0), "density_near_zero": 0.0, "degeneracy_hint": 0.0}

    k = min(n_samples, n)
    result: Dict[str, Any] = {}
    try:
        # 估计谱范围
        evals = spla.eigsh(
            matrix, k=k, which="LM", return_eigenvectors=False, maxiter=500
        )
        evals_min = spla.eigsh(
            matrix, k=k, which="SA", return_eigenvectors=False, maxiter=500
        )
        lo, hi = float(evals_min.min()), float(evals.max())
        width = max(hi - lo, 1e-14)
        result["spectral_range"] = (lo, hi)
        result["eigenvalue_estimate"] = np.concatenate([evals_min, evals]).tolist()

        # 零附近特征值采样（位移求逆 σ≈0，捕获零附近密集簇）
        # 只靠 LM/SA 极值采样会漏掉内部密集簇，此处补采样最小模特征值。
        evals_zero: Optional[np.ndarray] = None
        try:
            evals_zero = spla.eigsh(
                matrix, k=k, sigma=0.0, which="LM",
                return_eigenvectors=False, maxiter=500,
            )
        except Exception:
            evals_zero = None

        # 零附近密度：采样特征值中落在零窄邻域内的占比
        if evals_zero is not None and len(evals_zero) > 0:
            near_zero_frac = float(np.mean(np.abs(evals_zero) < 0.05 * width))
        else:
            all_evals = np.concatenate([evals_min, evals])
            near_zero_frac = float(np.mean(np.abs(all_evals) < 0.05 * width))
        result["density_near_zero"] = near_zero_frac
        if evals_zero is not None:
            result["eigenvalues_near_zero"] = np.sort(evals_zero).tolist()

        # 重特征值线索：特征值间距分布（合并所有采样值）
        sample = np.concatenate([evals_min, evals])
        if evals_zero is not None:
            sample = np.concatenate([sample, evals_zero])
        sorted_evals = np.sort(sample)
        gaps = np.diff(sorted_evals)
        if len(gaps) > 0:
            min_gap = float(np.min(gaps))
            result["min_gap"] = min_gap
            # 归一化间距
            result["degeneracy_hint"] = float(np.mean(gaps < 1e-6 * width))
        else:
            result["min_gap"] = 0.0
            result["degeneracy_hint"] = 0.0
    except Exception:
        result["spectral_range"] = (0.0, float(spla.norm(matrix, ord=1)))
        result["density_near_zero"] = 0.0
        result["degeneracy_hint"] = 0.0
        result["min_gap"] = 0.0

    return result


# ---------------------------------------------------------------------------
# 内存需求估计
# ---------------------------------------------------------------------------
def estimate_memory_requirement(
    matrix: sp.spmatrix,
    n_eigenvalues: int,
    krylov_dim: Optional[int] = None,
) -> Dict[str, float]:
    """估计求解所需内存峰值（字节）。

    内存主要来源：
    1. 稀疏矩阵存储：约 16 * nnz 字节（CSR：data + indices + indptr）。
    2. Krylov 基向量：2 * krylov_dim * n * 8 字节（复/双精度，含重正交化暂存）。
    3. 特征向量存储：n * k * 8 字节。
    4. 位移求逆 LU 因子：取决于填充，无法精确估计（给出警告）。

    Args:
        matrix: 稀疏矩阵。
        n_eigenvalues: 特征值个数。
        krylov_dim: Krylov 子空间维度。

    Returns:
        内存估计字典（字节）：
        - matrix_storage
        - krylov_basis
        - eigenvectors
        - total
    """
    n = matrix.shape[0]
    nnz = matrix.nnz
    k = n_eigenvalues
    if krylov_dim is None:
        krylov_dim = max(2 * k + 30, 60)

    # 稀疏矩阵 CSR 存储：indptr(n+1 int32) + indices(nnz int32) + data(nnz float64)
    matrix_bytes = (n + 1) * 4 + nnz * 4 + nnz * 8
    # 方块 Krylov（完全重正交化暂存）
    krylov_bytes = 2 * krylov_dim * n * 8
    # 特征向量
    eigvec_bytes = n * k * 8
    # 工作区
    workspace = 4 * n * 8

    total = matrix_bytes + krylov_bytes + eigvec_bytes + workspace
    return {
        "matrix_storage": float(matrix_bytes),
        "krylov_basis": float(krylov_bytes),
        "eigenvectors": float(eigvec_bytes),
        "workspace": float(workspace),
        "total": float(total),
        "total_mb": float(total / (1024 * 1024)),
    }


# ---------------------------------------------------------------------------
# 故障场景诊断器
# ---------------------------------------------------------------------------
@dataclass
class FaultDiagnosis:
    """故障场景诊断结果。"""

    scenarios: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenarios": self.scenarios,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
        }


class FaultDiagnoser:
    """故障场景诊断器。

    根据矩阵性质、位移配置与谱分析结果，识别典型故障场景并给出处置建议。
    """

    def __init__(self, matrix: sp.spmatrix, config: Any = None):
        self.matrix = matrix
        self.config = config

    def diagnose(
        self,
        sigma: Optional[float] = None,
        n_eigenvalues: Optional[int] = None,
        memory_limit: Optional[int] = None,
    ) -> FaultDiagnosis:
        """执行故障诊断。

        Args:
            sigma: 目标位移。
            n_eigenvalues: 目标特征值个数。
            memory_limit: 内存上限（字节）。

        Returns:
            FaultDiagnosis。
        """
        d = FaultDiagnosis()
        n = self.matrix.shape[0]
        n_eigenvalues = n_eigenvalues or 6

        # ---------- 谱分析 ----------
        spectrum = analyze_spectrum(self.matrix, n_samples=min(20, n))
        d.metrics["spectrum"] = spectrum

        # ---------- 场景 1：零附近密集谱（低频击穿） ----------
        density_zero = spectrum.get("density_near_zero", 0.0)
        lo, hi = spectrum.get("spectral_range", (0.0, 1.0))
        width = max(hi - lo, 1e-14)

        if density_zero > 0.3:
            d.scenarios.append("zero_dense_spectrum")
            d.warnings.append(
                "检测到零附近密集谱（低频击穿风险）：大量特征值在零附近聚集"
            )
            d.recommendations.append(
                "建议：使用 LOBPCG 块迭代同时收敛多个特征值；"
                "或使用位移求逆 σ 略偏离 0 避免 A 奇异；"
                "增加块大小与子空间维度"
            )

        # ---------- 场景 2：重特征值 ----------
        degeneracy = spectrum.get("degeneracy_hint", 0.0)
        if degeneracy > 0.3:
            d.scenarios.append("degenerate_eigenvalues")
            d.warnings.append("检测到重特征值（多重特征值）线索")
            d.recommendations.append(
                "建议：使用 LOBPCG 块迭代（天然支持多重特征值）；"
                "确保块大小 ≥ 重数；注意特征向量正交化"
            )

        # ---------- 场景 3：近奇异位移 ----------
        if sigma is not None:
            # 估算 σ 距最近特征值距离
            try:
                k = min(10, n)
                M = (self.matrix - sigma * sp.eye(n)).tocsc()
                evals = spla.eigsh(
                    M, k=k, which="SM", return_eigenvectors=False, maxiter=200
                )
                dist = float(np.min(np.abs(evals)))
            except Exception:
                dist = 0.0
            norm_A = float(spla.norm(self.matrix, ord=1)) if self.matrix.nnz > 0 else 1.0
            rel_dist = dist / max(norm_A, 1e-14)

            if rel_dist < 1e-8:
                d.scenarios.append("near_singular_shift")
                d.warnings.append(
                    f"检测到近奇异位移：σ={sigma:.6e} 距最近特征值相对距离 {rel_dist:.2e}"
                )
                d.recommendations.append(
                    "建议：启用近奇异保护；使用 Jacobi-Davidson（校正方程在正交补上求解，"
                    "即使 A-σI 近奇异仍良态）；或微调位移避开特征值"
                )
            elif rel_dist < 1e-5:
                d.warnings.append(
                    f"位移 σ={sigma:.6e} 相对靠近特征值（相对距离 {rel_dist:.2e}），"
                    "注意位移求逆迭代收敛可能变慢"
                )

        # ---------- 场景 4：内存受限 ----------
        mem_est = estimate_memory_requirement(self.matrix, n_eigenvalues)
        d.metrics["memory_estimate_mb"] = mem_est["total_mb"]
        if memory_limit is not None and mem_est["total"] > memory_limit:
            d.scenarios.append("memory_limited")
            d.warnings.append(
                f"估计内存需求 {mem_est['total_mb']:.1f} MB 超过限制"
                f" {memory_limit / (1024*1024):.1f} MB"
            )
            d.recommendations.append(
                "建议：降低 Krylov 子空间维度；使用部分重正交化；"
                "减少特征值个数；将特征向量按需计算"
            )

        # ---------- 场景 5：非正定矩阵 ----------
        # 通过谱范围判断
        if lo < 0:
            d.scenarios.append("indefinite_matrix")
            d.warnings.append(f"矩阵为非正定厄密矩阵（谱范围 [{lo:.4e}, {hi:.4e}]）")
            d.recommendations.append(
                "建议：避免使用 Cholesky 类预处理；使用位移求逆 + 带枢轴 LU 分解；"
                "或 Jacobi-Davidson / Krylov-Schur（非正定安全）"
            )

        # ---------- 场景 6：大规模矩阵（十万阶） ----------
        if n >= 100_000:
            d.scenarios.append("large_scale")
            d.recommendations.append(
                "大规模矩阵：确保使用稀疏存储（禁止稠密化）；"
                "启用多线程 BLAS；预计迭代次数与矩阵条件数有关"
            )

        return d


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------
def diagnose_matrix(
    matrix: sp.spmatrix,
    sigma: Optional[float] = None,
    n_eigenvalues: Optional[int] = None,
    memory_limit: Optional[int] = None,
) -> FaultDiagnosis:
    """便捷诊断函数。"""
    return FaultDiagnoser(matrix).diagnose(sigma, n_eigenvalues, memory_limit)
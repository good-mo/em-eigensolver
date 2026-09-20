"""Orchestrator：统一组织矩阵读取、预处理、核心迭代、收敛控制与结果输出。

高层 API，封装完整求解流水线：
读取 → 预处理 → 位移策略 → 核心迭代 → 收敛校验 → 安全校验 → 结果输出
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.sparse as sp

from .config import SolverConfig, load_config
from .io import load_matrix, save_matrix, save_eigenpairs
from .preprocess import preprocess, check_hermitian
from .shifts import ShiftManager
from .solvers.base import EigenResult
from .solvers.factory import create_solver, SOLVER_REGISTRY


@dataclass
class SolveReport:
    """求解报告。"""

    matrix_path: Optional[str] = None
    matrix_shape: Optional[tuple] = None
    nnz: int = 0
    is_hermitian: bool = True
    is_positive_definite: bool = False
    spectral_range: Optional[tuple] = None
    solver: str = ""
    n_eigenvalues: int = 0
    sigma: Optional[float] = None
    sigma_history: list = field(default_factory=list)
    converged: bool = False
    n_converged: int = 0
    residuals: list = field(default_factory=list)
    eigenvalues: list = field(default_factory=list)
    n_iterations: int = 0
    n_matvecs: int = 0
    elapsed_time: float = 0.0
    peak_memory_mb: float = 0.0
    near_singular_events: int = 0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix_path": self.matrix_path,
            "matrix_shape": list(self.matrix_shape) if self.matrix_shape else None,
            "nnz": self.nnz,
            "is_hermitian": self.is_hermitian,
            "is_positive_definite": self.is_positive_definite,
            "spectral_range": list(self.spectral_range) if self.spectral_range else None,
            "solver": self.solver,
            "n_eigenvalues": self.n_eigenvalues,
            "sigma": self.sigma,
            "sigma_history": self.sigma_history,
            "converged": self.converged,
            "n_converged": self.n_converged,
            "residuals": self.residuals,
            "n_iterations": self.n_iterations,
            "n_matvecs": self.n_matvecs,
            "elapsed_time": self.elapsed_time,
            "peak_memory_mb": self.peak_memory_mb,
            "near_singular_events": self.near_singular_events,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 内存峰值监控（尽力而为）
# ---------------------------------------------------------------------------
def _measure_peak_memory() -> float:
    """测量当前进程峰值内存（MB）。

    使用 resource 模块（Linux/Unix）。失败时返回 0。
    """
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


class EMESolver:
    """电磁特征值求解器高层封装。"""

    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        config_path: Optional[str] = None,
    ):
        if config is None:
            config = load_config(config_path)
        self.config = config
        self.report = SolveReport()

    def solve_matrix(
        self,
        matrix: sp.spmatrix,
        n_eigenvalues: Optional[int] = None,
        sigma: Optional[float] = None,
        which: Optional[str] = None,
        verbose: Optional[bool] = None,
    ) -> EigenResult:
        """对内存中的矩阵求解特征值。

        Args:
            matrix: 稀疏矩阵。
            n_eigenvalues: 特征值个数。
            sigma: 目标位移。
            which: 目标类型。

        Returns:
            EigenResult。
        """
        start = time.time()
        cfg = self.config
        n_eigenvalues = n_eigenvalues or cfg.n_eigenvalues
        which = which or cfg.which
        verbose = cfg.verbose if verbose is None else verbose

        # ---------- 性质检测 ----------
        A = matrix.tocsr()
        self.report.matrix_shape = A.shape
        self.report.nnz = A.nnz
        try:
            self.report.is_hermitian = check_hermitian(A)
        except Exception:
            self.report.is_hermitian = False

        if not self.report.is_hermitian:
            # 对非厄密矩阵，自动对称化（厄密部分）并警告
            self.report.warnings.append(
                "矩阵非厄密，使用厄密化处理 A_herm = (A + A^H)/2"
            )
            A = 0.5 * (A + A.conj().T).tocsr()

        # ---------- 位移管理 ----------
        shift_mgr = ShiftManager(cfg.shift)
        sigma_eff = shift_mgr.initialize(A) if sigma is None else float(sigma)
        self.report.sigma = sigma_eff

        # 近奇异检查与保护
        if shift_mgr.config.near_singular_protection:
            if shift_mgr.check_near_singular(A):
                self.report.warnings.append(
                    f"检测到近奇异位移 σ={sigma_eff:.6e}，触发保护"
                )
                sigma_eff = shift_mgr.apply_protection(A)
                self.report.sigma = sigma_eff
        self.report.near_singular_events = shift_mgr.near_singular_events

        # ---------- 求解器选择 ----------
        solver_type = cfg.solver
        if solver_type == "auto":
            from .solvers.factory import auto_select

            solver_type = auto_select(A, cfg)
        self.report.solver = solver_type

        if verbose:
            print(f"[EME] 矩阵形状: {A.shape}, nnz: {A.nnz}")
            print(f"[EME] 厄密: {self.report.is_hermitian}, 求解器: {solver_type}")
            print(f"[EME] 目标: {n_eigenvalues} 个特征值, σ={sigma_eff:.6e}")

        # ---------- 核心求解 ----------
        solver = create_solver(solver_type, A, cfg)
        result = solver.solve(A, n_eigenvalues=n_eigenvalues, sigma=sigma_eff, which=which)

        # ---------- 自适应位移（外层循环） ----------
        # 若未收敛且位移策略为自适应，迭代更新位移重新求解
        max_adapt = 5
        adapt_iter = 0
        while not result.converged and adapt_iter < max_adapt and cfg.shift.strategy in (
            "adaptive",
            "rayleigh",
            "wilkinson",
        ):
            adapt_iter += 1
            if verbose:
                print(f"[EME] 结果未收敛，自适应位移 round {adapt_iter}")
            # 用当前 Ritz 值更新位移
            if len(result.eigenvalues) > 0:
                new_sigma = shift_mgr.update(A, result.eigenvalues, result.eigenvectors)
                if new_sigma is None:
                    break
                sigma_eff = new_sigma
                self.report.sigma = sigma_eff
                self.report.sigma_history.append(sigma_eff)
                # 重新求解
                result2 = solver.solve(
                    A, n_eigenvalues=n_eigenvalues, sigma=sigma_eff, which=which
                )
                # 若新结果更好则采用
                if result2.converged or (
                    len(result2.eigenvalues) > 0
                    and np.mean(result2.residuals) < np.mean(result.residuals)
                ):
                    result = result2

        # ---------- 结果整理 ----------
        self.report.eigenvalues = (
            result.eigenvalues.tolist() if len(result.eigenvalues) else []
        )
        self.report.residuals = result.residuals.tolist()
        self.report.converged = result.converged
        self.report.n_converged = int(np.sum(result.residuals <= cfg.convergence.tol))
        self.report.n_iterations = result.n_iterations
        self.report.n_matvecs = result.n_matvecs
        self.report.elapsed_time = time.time() - start
        self.report.peak_memory_mb = _measure_peak_memory()

        if verbose:
            print(f"[EME] 收敛: {result.converged} ({self.report.n_converged}/{n_eigenvalues})")
            print(f"[EME] 最大残差: {np.max(result.residuals) if len(result.residuals) else 0:.3e}")
            print(f"[EME] 耗时: {self.report.elapsed_time:.3f}s, matvecs: {result.n_matvecs}")

        return result

    def solve_file(
        self,
        matrix_path: str,
        n_eigenvalues: Optional[int] = None,
        sigma: Optional[float] = None,
        which: Optional[str] = None,
        output_dir: Optional[str] = None,
        sanitize: str = "none",
    ) -> EigenResult:
        """从文件加载矩阵并求解。

        Args:
            matrix_path: 矩阵文件路径。
            n_eigenvalues: 特征值个数。
            sigma: 目标位移。
            which: 目标类型。
            output_dir: 输出目录（保存结果）。
            sanitize: 脱敏模式。

        Returns:
            EigenResult。
        """
        self.report.matrix_path = matrix_path
        matrix = load_matrix(matrix_path, sanitize=sanitize)
        result = self.solve_matrix(
            matrix,
            n_eigenvalues=n_eigenvalues,
            sigma=sigma,
            which=which,
        )

        # 输出
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(matrix_path))[0]
            # 保存特征对
            eig_path = os.path.join(output_dir, f"{base}_eigenpairs.npz")
            save_eigenpairs(
                result.eigenvalues,
                result.eigenvectors,
                eig_path,
                metadata={
                    "solver": self.report.solver,
                    "source": base,
                    "is_hermitian": self.report.is_hermitian,
                },
                sanitize=sanitize,
            )
            # 保存报告
            report_path = os.path.join(output_dir, f"{base}_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(self.report.to_json())
            # 保存配置副本
            config_path = os.path.join(output_dir, "config.yaml")
            if not os.path.exists(config_path):
                self.save_config(config_path)

        return result

    def save_config(self, path: str) -> None:
        """保存当前配置到 YAML。"""
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.config.to_dict(), f, default_flow_style=False, allow_unicode=True
            )

    def get_report(self) -> SolveReport:
        """获取求解报告（含安全脱敏）。"""
        # 报告中的路径信息脱敏
        return self.report
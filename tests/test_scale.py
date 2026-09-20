"""规模回归测试：验证十万阶非正定厄密稀疏矩阵的求解能力。

注意：这些测试可能耗时较长，默认标记为 slow，可通过 `-m slow` 运行。
"""

import time

import numpy as np
import pytest
import scipy.sparse as sp

from em_eigensolver.config import SolverConfig
from em_eigensolver.solvers import KrylovSchurSolver, LOBPCGSolver
from em_eigensolver.samples import generate_sample_matrix
from em_eigensolver.diagnostics import estimate_memory_requirement


@pytest.mark.slow
def test_large_scale_100k_generation():
    """生成十万阶矩阵，验证内存与稀疏性。"""
    A, meta = generate_sample_matrix("indefinite", size=100_000, sigma_offset=2.0, seed=42)
    assert A.shape == (100_000, 100_000)
    # 稀疏性：nnz 应远小于 n^2
    assert A.nnz < 20 * A.shape[0]
    # 内存估计
    mem = estimate_memory_requirement(A, n_eigenvalues=6)
    assert mem["total_mb"] < 2000  # 应远小于 16GB


@pytest.mark.slow
def test_large_scale_100k_solve():
    """十万阶矩阵求解（Krylov-Schur）。"""
    A, meta = generate_sample_matrix("indefinite", size=100_000, sigma_offset=2.0, seed=42)
    cfg = SolverConfig.from_dict({
        "solver": "krylov_schur",
        "n_eigenvalues": 4,
        "which": "target",
        "shift": {"sigma": 2.0, "strategy": "fixed"},
        "convergence": {"tol": 1e-6, "max_iter": 200, "max_restarts": 10},
        "seed": 42,
        "verbose": False,
    })
    solver = KrylovSchurSolver(cfg)
    start = time.time()
    result = solver.solve(A, n_eigenvalues=4, sigma=2.0, which="target")
    elapsed = time.time() - start
    assert len(result.eigenvalues) == 4
    # 残差应达标（允许一定容差）
    assert np.max(result.residuals) < 1e-3
    # 性能：十万阶应在合理时间内完成
    assert elapsed < 120


@pytest.mark.slow
def test_large_scale_100k_lobpcg():
    """十万阶矩阵求解（LOBPCG，块迭代）。"""
    A, meta = generate_sample_matrix("indefinite", size=100_000, sigma_offset=2.0, seed=42)
    cfg = SolverConfig.from_dict({
        "solver": "lobpcg",
        "n_eigenvalues": 4,
        "which": "smallest_magnitude",
        "shift": {"sigma": 0.0, "strategy": "fixed"},
        "convergence": {"tol": 1e-5, "max_iter": 200},
        "seed": 42,
        "verbose": False,
    })
    solver = LOBPCGSolver(cfg)
    start = time.time()
    result = solver.solve(A, n_eigenvalues=4, sigma=0.0, which="smallest_magnitude")
    elapsed = time.time() - start
    assert len(result.eigenvalues) == 4
    assert elapsed < 120


@pytest.mark.slow
def test_large_scale_memory_control():
    """验证大规模求解不稠密化（内存可控）。"""
    A, meta = generate_sample_matrix("indefinite", size=50_000, sigma_offset=2.0, seed=42)
    # 确保矩阵保持稀疏
    assert sp.issparse(A)
    assert A.nnz < 20 * A.shape[0]
    # 内存估计应远小于 16GB
    mem = estimate_memory_requirement(A, n_eigenvalues=6)
    assert mem["total_mb"] < 1000

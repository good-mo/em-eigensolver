"""测试各求解器（Lanczos、Krylov-Schur、Jacobi-Davidson、LOBPCG）。"""

import numpy as np
import pytest
import scipy.sparse as sp

from em_eigensolver.config import SolverConfig
from em_eigensolver.solvers import (
    LanczosSolver,
    KrylovSchurSolver,
    JacobiDavidsonSolver,
    LOBPCGSolver,
    create_solver,
)
from em_eigensolver.samples import generate_sample_matrix
from conftest import make_small_matrix, residual_norm


def _config(solver, n=4, which="target", sigma=None, tol=1e-6):
    return SolverConfig.from_dict({
        "solver": solver,
        "n_eigenvalues": n,
        "which": which,
        "shift": {"sigma": sigma, "strategy": "fixed"},
        "convergence": {"tol": tol, "max_iter": 500, "max_restarts": 30},
        "seed": 42,
        "verbose": False,
    })


def test_lanczos_positive_definite():
    A, true_evals = make_small_matrix(100, type="indefinite")
    cfg = _config("lanczos", n=4, which="smallest_magnitude", sigma=0.0)
    solver = LanczosSolver(cfg)
    result = solver.solve(A, n_eigenvalues=4, sigma=0.0, which="smallest_magnitude")
    assert len(result.eigenvalues) == 4
    # 残差应较小
    assert np.max(result.residuals) < 1e-4


def test_krylov_schur_target():
    A, true_evals = make_small_matrix(100, type="indefinite")
    # 目标：最接近 2.0 的特征值
    cfg = _config("krylov_schur", n=4, which="target", sigma=2.0)
    solver = KrylovSchurSolver(cfg)
    result = solver.solve(A, n_eigenvalues=4, sigma=2.0, which="target")
    assert len(result.eigenvalues) == 4
    # 特征值应接近真实值
    assert np.max(result.residuals) < 1e-3


def test_jacobi_davidson():
    A, true_evals = make_small_matrix(100, type="indefinite")
    cfg = _config("jacobi_davidson", n=3, which="target", sigma=2.0)
    solver = JacobiDavidsonSolver(cfg)
    result = solver.solve(A, n_eigenvalues=3, sigma=2.0, which="target")
    assert len(result.eigenvalues) == 3
    assert np.max(result.residuals) < 1e-3


def test_lobpcg():
    A, true_evals = make_small_matrix(100, type="indefinite")
    cfg = _config("lobpcg", n=4, which="smallest_magnitude", sigma=0.0)
    solver = LOBPCGSolver(cfg)
    result = solver.solve(A, n_eigenvalues=4, sigma=0.0, which="smallest_magnitude")
    assert len(result.eigenvalues) == 4
    assert np.max(result.residuals) < 1e-3


def test_create_solver_auto():
    A, _ = make_small_matrix(100)
    cfg = _config("auto", n=4, which="target", sigma=2.0)
    solver = create_solver("auto", A, cfg)
    assert solver is not None


def test_create_solver_invalid():
    with pytest.raises(ValueError):
        create_solver("unknown", None, SolverConfig())


def test_solver_registry():
    from em_eigensolver.solvers import SOLVER_REGISTRY
    assert "lanczos" in SOLVER_REGISTRY
    assert "krylov_schur" in SOLVER_REGISTRY
    assert "jacobi_davidson" in SOLVER_REGISTRY
    assert "lobpcg" in SOLVER_REGISTRY


def test_eigenresult_sort():
    from em_eigensolver.solvers.base import EigenResult
    r = EigenResult(
        eigenvalues=np.array([3.0, 1.0, 2.0]),
        eigenvectors=np.eye(3),
        residuals=np.array([0.1, 0.2, 0.3]),
        converged=True,
        n_iterations=1,
        n_matvecs=1,
        elapsed_time=0.0,
    )
    r.sort("ascending")
    assert np.allclose(r.eigenvalues, [1.0, 2.0, 3.0])

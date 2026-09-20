"""测试引擎（Orchestrator）与端到端流程。"""

import os
import tempfile

import numpy as np
import scipy.sparse as sp

from em_eigensolver.engine import EMESolver
from em_eigensolver.config import SolverConfig
from em_eigensolver.samples import generate_sample_matrix
from em_eigensolver.io import save_matrix


def test_solve_matrix_end_to_end():
    A, meta = generate_sample_matrix("indefinite", size=100, sigma_offset=2.0, seed=42)
    cfg = SolverConfig.from_dict({
        "solver": "krylov_schur",
        "n_eigenvalues": 4,
        "which": "target",
        "shift": {"sigma": 2.0, "strategy": "fixed"},
        "convergence": {"tol": 1e-6, "max_iter": 500},
        "seed": 42,
        "verbose": False,
    })
    solver = EMESolver(cfg)
    result = solver.solve_matrix(A, n_eigenvalues=4, sigma=2.0, which="target")
    assert len(result.eigenvalues) == 4
    assert solver.report.matrix_shape == A.shape
    assert solver.report.solver == "krylov_schur"


def test_solve_file_end_to_end():
    A, meta = generate_sample_matrix("indefinite", size=100, sigma_offset=2.0, seed=42)
    with tempfile.TemporaryDirectory() as d:
        matrix_path = os.path.join(d, "test.npz")
        save_matrix(A, matrix_path)
        out_dir = os.path.join(d, "out")
        cfg = SolverConfig.from_dict({
            "solver": "krylov_schur",
            "n_eigenvalues": 3,
            "which": "target",
            "shift": {"sigma": 2.0, "strategy": "fixed"},
            "convergence": {"tol": 1e-6, "max_iter": 500},
            "seed": 42,
            "verbose": False,
        })
        solver = EMESolver(cfg)
        result = solver.solve_file(matrix_path, n_eigenvalues=3, sigma=2.0, which="target", output_dir=out_dir)
        assert len(result.eigenvalues) == 3
        # 检查输出文件
        assert os.path.exists(os.path.join(out_dir, "test_eigenpairs.npz"))
        assert os.path.exists(os.path.join(out_dir, "test_report.json"))


def test_solve_non_hermitian_warning():
    rng = np.random.default_rng(0)
    A = sp.random(50, 50, density=0.1, random_state=rng, format="csr")
    cfg = SolverConfig.from_dict({
        "solver": "krylov_schur",
        "n_eigenvalues": 2,
        "which": "target",
        "shift": {"sigma": 0.0, "strategy": "fixed"},
        "convergence": {"tol": 1e-4, "max_iter": 200},
        "seed": 42,
        "verbose": False,
    })
    solver = EMESolver(cfg)
    result = solver.solve_matrix(A, n_eigenvalues=2, sigma=0.0, which="target")
    # 非厄密矩阵应触发警告
    assert any("非厄密" in w for w in solver.report.warnings)


def test_report_to_dict():
    A, meta = generate_sample_matrix("indefinite", size=50, sigma_offset=2.0, seed=42)
    cfg = SolverConfig.from_dict({
        "solver": "lobpcg",
        "n_eigenvalues": 2,
        "which": "smallest_magnitude",
        "shift": {"sigma": 0.0, "strategy": "fixed"},
        "convergence": {"tol": 1e-4, "max_iter": 200},
        "seed": 42,
        "verbose": False,
    })
    solver = EMESolver(cfg)
    solver.solve_matrix(A, n_eigenvalues=2, sigma=0.0, which="smallest_magnitude")
    d = solver.report.to_dict()
    assert "solver" in d
    assert "converged" in d
    assert "elapsed_time" in d

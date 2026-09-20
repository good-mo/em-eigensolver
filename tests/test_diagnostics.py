"""测试诊断模块。"""

import numpy as np
import scipy.sparse as sp

from em_eigensolver.diagnostics import (
    analyze_spectrum,
    estimate_memory_requirement,
    diagnose_matrix,
    ConvergenceMonitor,
)
from em_eigensolver.samples import generate_sample_matrix
from conftest import make_hermitian_positive


def test_analyze_spectrum():
    A = make_hermitian_positive(50)
    spec = analyze_spectrum(A, n_samples=10)
    assert "spectral_range" in spec
    lo, hi = spec["spectral_range"]
    assert lo <= hi


def test_estimate_memory():
    A = make_hermitian_positive(100)
    mem = estimate_memory_requirement(A, n_eigenvalues=6)
    assert mem["total"] > 0
    assert mem["total_mb"] > 0


def test_diagnose_positive():
    A = make_hermitian_positive(50)
    d = diagnose_matrix(A, sigma=5.0, n_eigenvalues=4)
    assert d is not None
    assert hasattr(d, "scenarios")


def test_diagnose_dense_zero():
    A, meta = generate_sample_matrix("dense_zero", 200, sigma_offset=0.0, seed=42)
    d = diagnose_matrix(A, sigma=0.0, n_eigenvalues=4)
    assert "zero_dense_spectrum" in d.scenarios


def test_diagnose_indefinite():
    A, meta = generate_sample_matrix("indefinite", 200, sigma_offset=2.0, seed=42)
    d = diagnose_matrix(A, sigma=2.0, n_eigenvalues=4)
    assert "indefinite_matrix" in d.scenarios


def test_diagnose_memory_limited():
    A = make_hermitian_positive(100)
    d = diagnose_matrix(A, n_eigenvalues=6, memory_limit=1000)  # 1KB 限制
    assert "memory_limited" in d.scenarios


def test_convergence_monitor():
    m = ConvergenceMonitor(window=5, tol=1e-6)
    # 模拟收敛
    for i in range(10):
        r = 1.0 / (i + 1)
        if m.update(r):
            break
    assert m.iterations > 0
    assert m.best_residual < 1.0


def test_convergence_stagnation():
    m = ConvergenceMonitor(window=5, tol=1e-8)
    # 模拟停滞（残差不变）
    for _ in range(10):
        m.update(0.5)
    assert m.is_stagnant

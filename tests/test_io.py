"""测试矩阵 IO 模块。"""

import os
import tempfile

import numpy as np
import scipy.sparse as sp

from em_eigensolver.io import (
    load_matrix,
    save_matrix,
    save_eigenpairs,
    load_eigenpairs,
    restore_matrix,
    DataSanitizer,
)


def _make_matrix(n=50):
    rng = np.random.default_rng(0)
    A = sp.random(n, n, density=0.1, random_state=rng, format="csr")
    A = (A + A.T) / 2  # 对称
    return A


def test_save_load_npz():
    A = _make_matrix()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "A.npz")
        save_matrix(A, path)
        B = load_matrix(path)
        assert B.shape == A.shape
        assert abs(B - A).max() < 1e-12


def test_save_load_mtx():
    A = _make_matrix()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "A.mtx")
        save_matrix(A, path)
        B = load_matrix(path)
        assert B.shape == A.shape
        assert abs(B - A).max() < 1e-10


def test_load_missing_file():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_matrix("/nonexistent.npz")


def test_save_load_eigenpairs():
    evals = np.array([1.0, 2.0, 3.0])
    evecs = np.eye(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "eig.npz")
        save_eigenpairs(evals, evecs, path, metadata={"solver": "test"})
        e2, v2, meta = load_eigenpairs(path)
        assert np.allclose(e2, evals)
        assert np.allclose(v2, evecs)
        assert meta["solver"] == "test"


def test_restore_matrix():
    A = _make_matrix()
    upper = restore_matrix(A, "upper")
    lower = restore_matrix(A, "lower")
    full = restore_matrix(A, "full")
    assert upper.shape == A.shape
    assert lower.shape == A.shape
    assert full.shape == A.shape


def test_sanitize_metadata():
    meta = {"password": "secret", "name": "cavity", "design": "x"}
    out = DataSanitizer.sanitize_metadata(meta)
    assert out["password"] == "[REDACTED]"
    assert out["name"] == "cavity"
    assert out["design"] == "[REDACTED]"


def test_sanitize_matrix_round():
    A = _make_matrix()
    B = DataSanitizer.sanitize_matrix(A, "round")
    assert B.shape == A.shape

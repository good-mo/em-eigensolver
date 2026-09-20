"""测试样例矩阵生成模块。"""

import numpy as np
import scipy.sparse as sp

from em_eigensolver.samples import (
    generate_sample_matrix,
    make_benchmark,
    make_sparse_with_givens,
)


def test_generate_cavity():
    A, meta = generate_sample_matrix("cavity", size=100, sigma_offset=12.0)
    assert A.shape[0] >= 100
    assert A.shape[0] == A.shape[1]
    assert "true_eigenvalues" in meta
    assert A.nnz > 0


def test_generate_photon():
    A, meta = generate_sample_matrix("photon", size=100, sigma_offset=12.0)
    assert A.shape[0] == A.shape[1]
    assert A.nnz > 0


def test_generate_pml():
    A, meta = generate_sample_matrix("pml", size=100, sigma_offset=12.0)
    assert A.shape[0] == A.shape[1]
    assert A.nnz > 0


def test_generate_dense_zero():
    A, meta = generate_sample_matrix("dense_zero", size=200, sigma_offset=0.0)
    assert A.shape[0] == 200
    evals = np.array(meta["true_eigenvalues"])
    # 大量特征值在零附近
    assert np.sum(np.abs(evals) < 1.0) > 0


def test_generate_degenerate():
    A, meta = generate_sample_matrix("degenerate", size=200, sigma_offset=0.0)
    evals = np.array(meta["true_eigenvalues"])
    # 存在重复特征值
    assert len(np.unique(np.round(evals, 6))) < len(evals)


def test_generate_indefinite():
    A, meta = generate_sample_matrix("indefinite", size=200, sigma_offset=2.0)
    evals = np.array(meta["true_eigenvalues"])
    # 谱跨越正负
    assert np.any(evals < 0) and np.any(evals > 0)


def test_generate_hermitian():
    A, meta = generate_sample_matrix("indefinite", size=100, sigma_offset=2.0)
    # 检查厄密性
    diff = (A - A.T.conj()).tocsr()
    assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-10


def test_make_benchmark():
    A, true_evals = make_benchmark(200)
    assert A.shape[0] == 200
    assert len(true_evals) == 10


def test_make_sparse_with_givens():
    evals = np.linspace(-5, 5, 50)
    A, rotations = make_sparse_with_givens(evals, n_rotations=30, seed=42)
    assert A.shape == (50, 50)
    assert len(rotations) == 30
    # 特征值应保持（相似变换）
    # 用稠密特征值验证（小矩阵）
    dense_evals = np.linalg.eigvalsh(A.toarray())
    assert np.allclose(np.sort(dense_evals), np.sort(evals), atol=1e-8)

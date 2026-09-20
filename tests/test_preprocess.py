"""测试预处理模块。"""

import numpy as np
import scipy.sparse as sp

from em_eigensolver.preprocess import (
    preprocess,
    check_hermitian,
    jacobi_scaling,
    reorder_matrix,
    estimate_spectral_range,
)
from conftest import make_hermitian_positive, make_hermitian_indefinite


def test_check_hermitian_positive():
    A = make_hermitian_positive(50)
    assert check_hermitian(A)


def test_check_hermitian_indefinite():
    A = make_hermitian_indefinite(50)
    assert check_hermitian(A)


def test_check_hermitian_non_symmetric():
    rng = np.random.default_rng(0)
    A = sp.random(50, 50, density=0.1, random_state=rng, format="csr")
    assert not check_hermitian(A)


def test_jacobi_scaling():
    A = make_hermitian_positive(50)
    A_hat, d = jacobi_scaling(A)
    assert A_hat.shape == A.shape
    assert len(d) == 50


def test_reorder_matrix():
    A = make_hermitian_positive(50)
    A_re, perm, inv_perm = reorder_matrix(A, "rcm")
    assert A_re.shape == A.shape
    assert len(perm) == 50
    assert len(inv_perm) == 50


def test_estimate_spectral_range():
    A = make_hermitian_positive(50)
    lo, hi = estimate_spectral_range(A, n_iters=10)
    assert lo <= hi
    assert lo > 0  # 正定矩阵最小特征值 > 0


def test_preprocess_positive():
    A = make_hermitian_positive(50)
    A_hat, info = preprocess(A, {"scale": True, "sort": True})
    assert info.is_hermitian
    assert info.is_positive_definite
    assert A_hat.shape == A.shape


def test_preprocess_indefinite():
    A = make_hermitian_indefinite(50)
    A_hat, info = preprocess(A, {"scale": True, "sort": True})
    assert info.is_hermitian
    assert not info.is_positive_definite
    assert A_hat.shape == A.shape


def test_restore_eigenvector():
    from em_eigensolver.preprocess import PreprocessingInfo
    # 符号翻转缩放：d_i = ±1，v_orig = v_hat / d（d_i=±1 时乘除等价）
    info = PreprocessingInfo(scaled=True, scale_vector=np.array([1.0, -1.0]))
    v = np.array([1.0, 1.0])
    restored = info.restore_eigenvector(v)
    assert np.allclose(restored, [1.0, -1.0])

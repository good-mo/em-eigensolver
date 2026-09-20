"""测试位移策略模块。"""

import numpy as np
import scipy.sparse as sp

from em_eigensolver.shifts import ShiftManager
from em_eigensolver.config import ShiftConfig
from conftest import make_hermitian_positive


def test_shift_fixed():
    cfg = ShiftConfig(sigma=5.0, strategy="fixed")
    mgr = ShiftManager(cfg)
    A = make_hermitian_positive(50)
    sigma = mgr.initialize(A)
    assert sigma == 5.0
    # fixed 策略不更新
    new = mgr.update(A, np.array([1.0, 2.0]))
    assert new is None


def test_shift_adaptive_initialize():
    cfg = ShiftConfig(strategy="adaptive", target_band=[10.0, 20.0])
    mgr = ShiftManager(cfg)
    A = make_hermitian_positive(50)
    sigma = mgr.initialize(A)
    assert sigma == 15.0  # 频段中心


def test_shift_adaptive_update():
    cfg = ShiftConfig(strategy="adaptive", sigma=2.0, damping=0.5)
    mgr = ShiftManager(cfg)
    A = make_hermitian_positive(50)
    mgr.initialize(A)
    new = mgr.update(A, np.array([1.0, 2.0, 3.0]))
    assert new is not None
    assert len(mgr.history) >= 2


def test_shift_near_singular_detection():
    cfg = ShiftConfig(sigma=0.0, strategy="fixed", near_singular_protection=True)
    mgr = ShiftManager(cfg)
    A = make_hermitian_positive(50)
    mgr.initialize(A)
    # 对正定矩阵，σ=0 远离特征值，不应触发
    assert not mgr.check_near_singular(A)


def test_shift_apply_protection():
    cfg = ShiftConfig(sigma=1.0, strategy="fixed", near_singular_protection=True)
    mgr = ShiftManager(cfg)
    A = make_hermitian_positive(50)
    mgr.initialize(A)
    safe = mgr.apply_protection(A)
    assert safe != 1.0  # 位移被微调


def test_shift_rayleigh_quotient():
    cfg = ShiftConfig(strategy="adaptive", adaptive_mode="rayleigh_quotient", sigma=2.0)
    mgr = ShiftManager(cfg)
    A = make_hermitian_positive(50)
    mgr.initialize(A)
    # 构造一个近似特征向量
    v = np.ones(50)
    v = v / np.linalg.norm(v)
    new = mgr.update(A, np.array([2.0]), np.column_stack([v]))
    assert new is not None

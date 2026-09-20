"""测试配置系统。"""

import os
import tempfile

import pytest

from em_eigensolver.config import SolverConfig, load_config, default_config


def test_default_config():
    cfg = default_config()
    assert cfg.solver == "auto"
    assert cfg.n_eigenvalues == 6
    assert cfg.which == "target"
    assert cfg.shift.strategy == "adaptive"
    assert cfg.convergence.tol == 1e-8
    assert cfg.parallel.backend == "auto"
    cfg.validate()


def test_config_override():
    cfg = SolverConfig.from_dict({
        "solver": "lobpcg",
        "n_eigenvalues": 10,
        "shift": {"strategy": "fixed", "sigma": 5.0},
        "convergence": {"tol": 1e-6},
    })
    assert cfg.solver == "lobpcg"
    assert cfg.n_eigenvalues == 10
    assert cfg.shift.strategy == "fixed"
    assert cfg.shift.sigma == 5.0
    assert cfg.convergence.tol == 1e-6


def test_config_invalid_solver():
    with pytest.raises(ValueError):
        SolverConfig.from_dict({"solver": "unknown"})


def test_config_invalid_shift():
    with pytest.raises(ValueError):
        SolverConfig.from_dict({"shift": {"strategy": "bad"}})


def test_load_config_from_yaml():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("solver: lanczos\nn_eigenvalues: 4\n")
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg.solver == "lanczos"
        assert cfg.n_eigenvalues == 4
    finally:
        os.unlink(path)


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_config_to_dict():
    cfg = default_config()
    d = cfg.to_dict()
    assert "solver" in d
    assert "shift" in d
    assert "convergence" in d

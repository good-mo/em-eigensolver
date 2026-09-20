"""测试 CLI 命令。"""

import os
import tempfile

import pytest

from em_eigensolver.cli import main


def test_cli_config():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "cfg.yaml")
        rc = main(["config", out])
        assert rc == 0
        assert os.path.exists(out)


def test_cli_generate():
    with tempfile.TemporaryDirectory() as d:
        rc = main(["generate", d, "--type", "indefinite", "--size", "100", "--seed", "42"])
        assert rc == 0
        files = os.listdir(d)
        assert any(f.endswith(".npz") for f in files)


def test_cli_diagnose():
    with tempfile.TemporaryDirectory() as d:
        main(["generate", d, "--type", "indefinite", "--size", "100", "--seed", "42"])
        matrix = [f for f in os.listdir(d) if f.endswith(".npz")][0]
        rc = main(["diagnose", os.path.join(d, matrix), "--sigma", "2.0"])
        assert rc == 0


def test_cli_audit():
    with tempfile.TemporaryDirectory() as d:
        # 干净源码
        with open(os.path.join(d, "clean.py"), "w") as f:
            f.write("x = 1\n")
        rc = main(["audit", d])
        assert rc == 0


def test_cli_solve():
    with tempfile.TemporaryDirectory() as d:
        main(["generate", d, "--type", "indefinite", "--size", "100", "--seed", "42"])
        matrix = [f for f in os.listdir(d) if f.endswith(".npz")][0]
        out = os.path.join(d, "out")
        rc = main([
            "solve", os.path.join(d, matrix),
            "--n", "3", "--sigma", "2.0", "--solver", "krylov_schur",
            "--output", out, "--quiet",
        ])
        assert rc == 0
        assert os.path.exists(os.path.join(out, "test_eigenpairs.npz")) or \
               any(f.endswith("_eigenpairs.npz") for f in os.listdir(out))

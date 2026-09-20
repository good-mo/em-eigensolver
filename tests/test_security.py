"""测试安全模块。"""

import os
import tempfile

import numpy as np
import scipy.sparse as sp

from em_eigensolver.security import (
    SecurityAuditor,
    DataSanitizer,
    AccessControl,
    audit_security,
    fingerprint_matrix,
)


def test_audit_clean_source():
    with tempfile.TemporaryDirectory() as d:
        # 写一个干净的源文件
        with open(os.path.join(d, "clean.py"), "w") as f:
            f.write("x = 1 + 2\n")
        audit = SecurityAuditor(d).audit_source(d)
        assert audit.passed


def test_audit_hardcoded_answer():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "bad.py"), "w") as f:
            f.write("eigenvalue = [1.234, 5.678]\n")
        audit = SecurityAuditor(d).audit_source(d)
        assert not audit.passed
        assert len(audit.issues) > 0


def test_audit_reference_read():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "bad.py"), "w") as f:
            f.write("data = np.load('answer.npz')\n")
        audit = SecurityAuditor(d).audit_source(d)
        assert not audit.passed


def test_sanitize_metadata():
    meta = {"password": "x", "name": "cavity", "secret_key": "y"}
    out = DataSanitizer.sanitize_metadata(meta)
    assert out["password"] == "[REDACTED]"
    assert out["secret_key"] == "[REDACTED]"
    assert out["name"] == "cavity"


def test_sanitize_matrix():
    A = sp.eye(5, format="csr")
    B = DataSanitizer.sanitize_matrix(A, "round")
    assert B.shape == A.shape


def test_access_control():
    ac = AccessControl(allowed_roles=["engineer"], allow_export=True)
    assert ac.check_access("engineer", "solve")
    assert ac.check_access("engineer", "export")
    assert not ac.check_access("guest", "solve")
    assert not ac.check_access("engineer", "read_matrix") if not ac.allow_raw_matrix else True


def test_fingerprint():
    A = sp.eye(5, format="csr")
    fp1 = fingerprint_matrix(A)
    fp2 = fingerprint_matrix(A)
    assert fp1 == fp2
    assert len(fp1) == 16

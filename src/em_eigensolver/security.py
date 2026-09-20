"""安全设计模块：脱敏、防硬编码、访问控制说明。

针对赛题要求：
- 禁止硬编码答案、读取参考结果或针对公开用例调参。
- 涉及矩阵数据或计算结果时，实现基本的敏感信息保护措施。

本模块提供：
1. SecurityAuditor：安全审计器，检测硬编码答案、参考结果读取、针对用例调参。
2. DataSanitizer：数据脱敏（矩阵值、元数据、特征对）。
3. AccessControl：访问控制说明与策略。
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# 安全审计器
# ---------------------------------------------------------------------------
@dataclass
class SecurityAudit:
    """安全审计结果。"""

    passed: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "warnings": self.warnings,
            "checks": self.checks,
        }


class SecurityAuditor:
    """安全审计器。

    检测以下违规行为：
    1. 硬编码答案：代码中直接写入已知特征值/特征向量。
    2. 读取参考结果：代码中读取预先计算好的结果文件。
    3. 针对公开用例调参：根据矩阵文件名/规模硬编码特定参数。
    """

    # 硬编码特征值/答案的常见模式
    _HARDCODED_PATTERNS = [
        r"eigenvalue\s*=\s*\[?\s*[-+]?\d+\.\d+",
        r"answer\s*=\s*\[?\s*[-+]?\d+\.\d+",
        r"expected\s*=\s*\[?\s*[-+]?\d+\.\d+",
        r"ground_truth\s*=\s*\[?\s*[-+]?\d+\.\d+",
        r"reference_result\s*=\s*\[?\s*[-+]?\d+\.\d+",
    ]

    # 读取参考结果文件的模式
    _REFERENCE_READ_PATTERNS = [
        r"load.*(?:answer|result|ground_truth|reference|expected)",
        r"np\.load.*(?:answer|result|ground_truth|reference|expected)",
        r"open\(.*(?:answer|result|ground_truth|reference|expected)",
    ]

    # 针对用例调参的模式（根据文件名/规模硬编码）
    _CASE_TUNING_PATTERNS = [
        r"if\s+.*(?:matrix|file|name|n)\s*==\s*['\"].*['\"]",
        r"if\s+.*(?:n|size|dim)\s*==\s*\d{4,}",
    ]

    def __init__(self, source_dir: Optional[str] = None):
        self.source_dir = source_dir

    def audit_source(self, source_dir: Optional[str] = None) -> SecurityAudit:
        """审计源代码目录，检测违规模式。

        Args:
            source_dir: 源代码目录。None 则用初始化时的目录。

        Returns:
            SecurityAudit。
        """
        source_dir = source_dir or self.source_dir
        audit = SecurityAudit()
        if source_dir is None or not os.path.isdir(source_dir):
            audit.warnings.append("未指定源代码目录，跳过源码审计")
            return audit

        # 收集所有 .py 文件
        py_files = []
        for root, _, files in os.walk(source_dir):
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))

        hardcoded_hits = []
        ref_read_hits = []
        case_tuning_hits = []

        for path in py_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            for pat in self._HARDCODED_PATTERNS:
                if re.search(pat, content):
                    hardcoded_hits.append((path, pat))
            for pat in self._REFERENCE_READ_PATTERNS:
                if re.search(pat, content):
                    ref_read_hits.append((path, pat))
            for pat in self._CASE_TUNING_PATTERNS:
                if re.search(pat, content):
                    case_tuning_hits.append((path, pat))

        audit.checks["hardcoded_answers"] = len(hardcoded_hits)
        audit.checks["reference_result_reads"] = len(ref_read_hits)
        audit.checks["case_tuning"] = len(case_tuning_hits)

        if hardcoded_hits:
            audit.passed = False
            audit.issues.append(f"检测到疑似硬编码答案: {hardcoded_hits[:5]}")
        if ref_read_hits:
            audit.passed = False
            audit.issues.append(f"检测到疑似读取参考结果: {ref_read_hits[:5]}")
        if case_tuning_hits:
            audit.warnings.append(f"检测到疑似针对用例调参: {case_tuning_hits[:5]}")

        return audit

    def audit_config(self, config: Any) -> SecurityAudit:
        """审计配置，确保没有硬编码答案。"""
        audit = SecurityAudit()
        # 检查配置中是否有可疑的固定特征值
        cfg_dict = config.to_dict() if hasattr(config, "to_dict") else {}
        for key, value in cfg_dict.items():
            if isinstance(value, (list, tuple)) and len(value) > 0:
                if all(isinstance(v, (int, float)) for v in value) and len(value) <= 20:
                    # 可能是特征值列表，检查是否异常
                    audit.warnings.append(
                        f"配置字段 '{key}' 包含数值列表，请确认非硬编码答案"
                    )
        return audit


# ---------------------------------------------------------------------------
# 数据脱敏
# ---------------------------------------------------------------------------
class DataSanitizer:
    """数据脱敏工具。

    提供矩阵值、元数据、特征对的脱敏方法。
    """

    SENSITIVE_KEYS = {
        "password", "secret", "token", "key", "auth", "credential",
        "private", "confidential", "design", "geometry", "material",
        "serial", "license", "customer", "project",
    }

    @classmethod
    def sanitize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """过滤元数据中的敏感字段。"""
        result = {}
        for k, v in metadata.items():
            kl = str(k).lower()
            if any(s in kl for s in cls.SENSITIVE_KEYS):
                result[k] = "[REDACTED]"
            else:
                result[k] = v
        return result

    @classmethod
    def sanitize_matrix(cls, matrix: sp.spmatrix, mode: str = "none") -> sp.spmatrix:
        """对矩阵值做脱敏。

        Args:
            matrix: 输入稀疏矩阵。
            mode: 'none' | 'round' | 'stat' | 'hash'。

        Returns:
            脱敏后的矩阵。
        """
        if mode == "none":
            return matrix
        m = matrix.tocsr()
        if mode == "round":
            data = np.round(m.data, decimals=6)
            return m._with_data(data)
        if mode == "stat":
            mean = float(np.mean(m.data))
            data = np.full_like(m.data, mean, dtype=m.dtype)
            return m._with_data(data)
        if mode == "hash":
            # 对非零值做哈希脱敏（保留符号与量级）
            data = np.sign(m.data) * np.log1p(np.abs(m.data))
            return m._with_data(data)
        raise ValueError(f"未知脱敏模式: {mode}")

    @classmethod
    def sanitize_eigenpairs(
        cls,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        mode: str = "none",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对特征对做脱敏。"""
        if mode == "none":
            return eigenvalues, eigenvectors
        if mode == "round":
            return (
                np.round(eigenvalues, decimals=6),
                np.round(eigenvectors, decimals=6),
            )
        if mode == "stat":
            # 仅保留统计特征
            return (
                np.array([np.mean(eigenvalues)]),
                np.array([np.mean(eigenvectors, axis=1)]),
            )
        raise ValueError(f"未知脱敏模式: {mode}")


# ---------------------------------------------------------------------------
# 访问控制
# ---------------------------------------------------------------------------
@dataclass
class AccessControl:
    """访问控制策略说明。

    记录数据访问策略，用于敏感信息保护说明。
    """

    #: 允许访问数据的角色
    allowed_roles: List[str] = field(default_factory=lambda: ["engineer", "analyst"])
    #: 是否允许导出结果
    allow_export: bool = True
    #: 是否允许访问原始矩阵
    allow_raw_matrix: bool = True
    #: 数据保留策略
    retention_policy: str = "local_only"
    #: 脱敏级别
    sanitize_level: str = "none"

    def check_access(self, role: str, action: str) -> bool:
        """检查角色是否有权执行操作。

        Args:
            role: 角色名。
            action: 操作名（'read_matrix', 'export', 'solve'）。

        Returns:
            True 如果允许。
        """
        if role not in self.allowed_roles:
            return False
        if action == "export" and not self.allow_export:
            return False
        if action == "read_matrix" and not self.allow_raw_matrix:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_roles": self.allowed_roles,
            "allow_export": self.allow_export,
            "allow_raw_matrix": self.allow_raw_matrix,
            "retention_policy": self.retention_policy,
            "sanitize_level": self.sanitize_level,
        }


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------
def audit_security(source_dir: Optional[str] = None) -> SecurityAudit:
    """便捷安全审计入口。"""
    return SecurityAuditor(source_dir).audit_source(source_dir)


def fingerprint_matrix(matrix: sp.spmatrix) -> str:
    """计算矩阵指纹（用于可复现性校验，不泄露数据）。"""
    m = matrix.tocsr()
    h = hashlib.sha256()
    h.update(m.data.tobytes())
    h.update(m.indices.tobytes())
    h.update(m.indptr.tobytes())
    h.update(np.array(m.shape).tobytes())
    return h.hexdigest()[:16]
"""矩阵与特征对读取/写出模块。

支持 numpy / scipy 稀疏格式输入输出，实现高效稀疏存储，按需恢复三角/完整矩阵。
包含敏感信息保护（脱敏、字段过滤、访问控制说明）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy import io as sio

# 支持的稀疏格式
_SPARSE_FORMATS = ("csr", "csc", "coo", "lil", "dok", "bsr", "dia")


# ---------------------------------------------------------------------------
# 敏感信息保护
# ---------------------------------------------------------------------------
class DataSanitizer:
    """敏感信息保护工具。

    电磁仿真矩阵可能包含设计参数、几何信息等敏感数据。本类提供：
    - 脱敏：对矩阵值做统计脱敏（仅保留统计特征）
    - 字段过滤：过滤元数据中的敏感字段
    - 访问控制说明：记录数据访问策略
    """

    SENSITIVE_KEYS = {
        "password", "secret", "token", "key", "auth", "credential",
        "private", "confidential", "design", "geometry", "material",
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
            mode: 'none' 不脱敏；'round' 舍入；'stat' 仅保留统计特征（破坏性）。
        """
        if mode == "none":
            return matrix
        if mode == "round":
            data = np.round(matrix.data, decimals=6)
            return matrix.copy().tocsr()._with_data(data)
        if mode == "stat":
            # 破坏性脱敏：仅保留均值/方差，用于演示
            mean = float(np.mean(matrix.data))
            std = float(np.std(matrix.data))
            data = np.full_like(matrix.data, mean, dtype=matrix.dtype)
            return matrix.copy().tocsr()._with_data(data)
        raise ValueError(f"未知脱敏模式: {mode}")


# ---------------------------------------------------------------------------
# 矩阵读取
# ---------------------------------------------------------------------------
def load_matrix(
    path: str,
    format: Optional[str] = None,
    sanitize: str = "none",
) -> sp.spmatrix:
    """从文件加载稀疏矩阵。

    支持格式：
    - .npz / .npy (numpy)
    - .mtx (Matrix Market)
    - .csr / .csc / .coo (scipy 稀疏二进制)
    - .npz 内含 'data','indices','indptr' 的 scipy 稀疏

    Args:
        path: 文件路径。
        format: 显式指定格式（覆盖扩展名推断）。
        sanitize: 脱敏模式 'none'|'round'|'stat'。

    Returns:
        scipy.sparse 稀疏矩阵（CSR 格式）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"矩阵文件不存在: {path}")

    ext = format or os.path.splitext(path)[1].lower().lstrip(".")

    if ext in ("npz", "npy"):
        matrix = _load_numpy_sparse(path)
    elif ext == "mtx":
        matrix = sio.mmread(path)
    elif ext in _SPARSE_FORMATS:
        matrix = _load_scipy_sparse(path, ext)
    else:
        raise ValueError(f"不支持的矩阵格式: {ext}")

    matrix = matrix.tocsr()
    matrix = DataSanitizer.sanitize_matrix(matrix, sanitize)
    return matrix


def _load_numpy_sparse(path: str) -> sp.spmatrix:
    """从 numpy .npz/.npy 加载稀疏矩阵。"""
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        keys = set(data.files)
        # scipy 稀疏格式
        if {"data", "indices", "indptr"}.issubset(keys):
            shape = tuple(data["shape"]) if "shape" in keys else None
            m = sp.csr_matrix(
                (data["data"], data["indices"], data["indptr"]),
                shape=shape,
            )
            return m
        # 稠密数组
        if "matrix" in keys:
            return sp.csr_matrix(data["matrix"])
        raise ValueError("npz 文件缺少可识别的矩阵数据")
    else:
        # 单个数组
        return sp.csr_matrix(data)


def _load_scipy_sparse(path: str, fmt: str) -> sp.spmatrix:
    """从 scipy 稀疏二进制格式加载。"""
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        keys = set(data.files)
        if {"data", "indices", "indptr"}.issubset(keys):
            shape = tuple(data["shape"]) if "shape" in keys else None
            return sp.csr_matrix(
                (data["data"], data["indices"], data["indptr"]),
                shape=shape,
            )
    raise ValueError(f"无法从 {path} 解析稀疏矩阵")


# ---------------------------------------------------------------------------
# 矩阵写出
# ---------------------------------------------------------------------------
def save_matrix(
    matrix: sp.spmatrix,
    path: str,
    format: Optional[str] = None,
    sanitize: str = "none",
) -> None:
    """保存稀疏矩阵到文件。

    Args:
        matrix: 稀疏矩阵。
        path: 输出路径。
        format: 输出格式（覆盖扩展名推断）。
        sanitize: 脱敏模式。
    """
    matrix = DataSanitizer.sanitize_matrix(matrix, sanitize)
    ext = format or os.path.splitext(path)[1].lower().lstrip(".")

    if ext in ("npz", "npy"):
        _save_numpy_sparse(matrix, path)
    elif ext == "mtx":
        sio.mmwrite(path, matrix)
    elif ext in _SPARSE_FORMATS:
        _save_scipy_sparse(matrix, path, ext)
    else:
        raise ValueError(f"不支持的输出格式: {ext}")


def _save_numpy_sparse(matrix: sp.spmatrix, path: str) -> None:
    """保存为 numpy .npz（scipy 稀疏格式）。"""
    m = matrix.tocsr()
    np.savez(
        path,
        data=m.data,
        indices=m.indices,
        indptr=m.indptr,
        shape=np.array(m.shape),
    )


def _save_scipy_sparse(matrix: sp.spmatrix, path: str, fmt: str) -> None:
    """保存为 scipy 稀疏二进制格式。"""
    m = matrix.tocsr()
    np.savez(
        path,
        data=m.data,
        indices=m.indices,
        indptr=m.indptr,
        shape=np.array(m.shape),
    )


# ---------------------------------------------------------------------------
# 特征对读取/写出
# ---------------------------------------------------------------------------
def save_eigenpairs(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
    sanitize: str = "none",
) -> None:
    """保存特征值/特征向量。

    Args:
        eigenvalues: 特征值数组 (k,)。
        eigenvectors: 特征向量矩阵 (n, k)。
        path: 输出 .npz 路径。
        metadata: 附加元数据（会被脱敏）。
        sanitize: 脱敏模式。
    """
    if metadata is not None:
        metadata = DataSanitizer.sanitize_metadata(metadata)
    np.savez(
        path,
        eigenvalues=np.asarray(eigenvalues),
        eigenvectors=np.asarray(eigenvectors),
        metadata=np.array([str(metadata)]) if metadata else np.array([]),
    )


def load_eigenpairs(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """加载特征值/特征向量。

    Returns:
        (eigenvalues, eigenvectors, metadata)
    """
    data = np.load(path, allow_pickle=True)
    eigenvalues = data["eigenvalues"]
    eigenvectors = data["eigenvectors"]
    metadata = None
    if "metadata" in data and data["metadata"].size > 0:
        try:
            import ast
            metadata = ast.literal_eval(str(data["metadata"][0]))
        except Exception:
            metadata = None
    return eigenvalues, eigenvectors, metadata


# ---------------------------------------------------------------------------
# 按需恢复三角/完整矩阵
# ---------------------------------------------------------------------------
def restore_matrix(
    matrix: sp.spmatrix,
    mode: str = "full",
) -> sp.spmatrix:
    """按需恢复三角/完整矩阵。

    Args:
        matrix: 输入稀疏矩阵。
        mode: 'full' 完整矩阵；'upper' 上三角；'lower' 下三角。

    Returns:
        恢复后的稀疏矩阵。
    """
    if mode == "full":
        return matrix.tocsr()
    if mode == "upper":
        return sp.triu(matrix).tocsr()
    if mode == "lower":
        return sp.tril(matrix).tocsr()
    raise ValueError(f"未知恢复模式: {mode}")

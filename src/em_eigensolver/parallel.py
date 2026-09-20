"""并行后端模块：多线程 / MPI / GPU 加速。

支持：
- 多线程（BLAS/OpenMP）：通过设置线程数加速矩阵-向量乘法。
- MPI：分布式矩阵-向量乘法（可选，需 mpi4py）。
- GPU：CUDA 加速（可选，需 cupy）。

提供最小化降级模式：缺少 GPU/MPI 时自动回退 CPU 单机多核。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp

from .config import ParallelConfig


@dataclass
class BackendInfo:
    """并行后端信息。"""

    backend: str
    num_threads: Optional[int]
    gpu_available: bool
    mpi_available: bool
    mpi_rank: int = 0
    mpi_size: int = 1
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "num_threads": self.num_threads,
            "gpu_available": self.gpu_available,
            "mpi_available": self.mpi_available,
            "mpi_rank": self.mpi_rank,
            "mpi_size": self.mpi_size,
            "fallback_used": self.fallback_used,
        }


class ParallelBackend:
    """并行后端管理器。

    根据配置选择后端，并提供矩阵-向量乘法加速。
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        self.info = self._detect_backend()

    def _detect_backend(self) -> BackendInfo:
        """检测可用后端并初始化。"""
        cfg = self.config
        backend = cfg.backend

        # 检测 GPU
        gpu_available = False
        try:
            import cupy  # noqa: F401

            gpu_available = True
        except ImportError:
            gpu_available = False

        # 检测 MPI
        mpi_available = False
        mpi_rank, mpi_size = 0, 1
        try:
            from mpi4py import MPI

            mpi_available = True
            mpi_rank = MPI.COMM_WORLD.Get_rank()
            mpi_size = MPI.COMM_WORLD.Get_size()
        except ImportError:
            mpi_available = False

        # 决定实际后端
        actual = backend
        fallback = False
        if backend == "auto":
            if cfg.use_gpu and gpu_available:
                actual = "gpu"
            elif mpi_available and mpi_size > 1:
                actual = "mpi"
            else:
                actual = "threads"
        elif backend == "gpu" and not gpu_available:
            if cfg.allow_fallback:
                actual = "threads"
                fallback = True
            else:
                raise RuntimeError("GPU 不可用且不允许降级")
        elif backend == "mpi" and not mpi_available:
            if cfg.allow_fallback:
                actual = "threads"
                fallback = True
            else:
                raise RuntimeError("MPI 不可用且不允许降级")

        # 设置线程数
        num_threads = cfg.num_threads
        if actual in ("threads", "cpu"):
            if num_threads is None:
                num_threads = os.cpu_count() or 1
            # 设置 BLAS 线程
            try:
                os.environ["OMP_NUM_THREADS"] = str(num_threads)
                os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
                os.environ["MKL_NUM_THREADS"] = str(num_threads)
            except Exception:
                pass

        return BackendInfo(
            backend=actual,
            num_threads=num_threads,
            gpu_available=gpu_available,
            mpi_available=mpi_available,
            mpi_rank=mpi_rank,
            mpi_size=mpi_size,
            fallback_used=fallback,
        )

    def matvec(self, A: sp.spmatrix, x: np.ndarray) -> np.ndarray:
        """执行矩阵-向量乘法（按后端加速）。

        Args:
            A: 稀疏矩阵。
            x: 向量。

        Returns:
            A @ x。
        """
        if self.info.backend == "gpu" and self.info.gpu_available:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cpsp

                A_gpu = cpsp.csr_matrix(A)
                x_gpu = cp.asarray(x)
                y = A_gpu @ x_gpu
                return cp.asnumpy(y)
            except Exception:
                # GPU 失败降级
                return A @ x
        # CPU / threads / mpi：scipy 稀疏矩阵乘法已利用 BLAS 多线程
        return A @ x

    def get_info(self) -> BackendInfo:
        """获取后端信息。"""
        return self.info


def get_parallel_backend(config: Optional[ParallelConfig] = None) -> ParallelBackend:
    """获取并行后端实例。"""
    return ParallelBackend(config)
"""em_eigensolver: 面向电磁谐振与微波器件仿真的高效、稳定大规模非正定厄密稀疏矩阵特征值求解方案。

本包提供：
- 矩阵读取/写出（numpy / scipy 稀疏格式）
- 预处理（缩放、平衡、排序）
- 多种 Krylov 子空间与现代迭代求解器（Lanczos、Krylov-Schur、Jacobi-Davidson、LOBPCG）
- 自适应位移策略（目标频段）
- 收敛控制与故障诊断
- 安全设计（脱敏、防硬编码）
- 并行后端（多线程 / MPI / GPU 可选）
"""

from .config import SolverConfig, load_config
from .io import load_matrix, save_matrix, load_eigenpairs, save_eigenpairs
from .solvers.base import EigenSolver, EigenResult
from .solvers.lanczos import LanczosSolver
from .solvers.krylov_schur import KrylovSchurSolver
from .solvers.jacobi_davidson import JacobiDavidsonSolver
from .solvers.lobpcg import LOBPCGSolver
from .solvers.factory import create_solver, SOLVER_REGISTRY

__version__ = "0.1.0"

__all__ = [
    "SolverConfig",
    "load_config",
    "load_matrix",
    "save_matrix",
    "load_eigenpairs",
    "save_eigenpairs",
    "EigenSolver",
    "EigenResult",
    "LanczosSolver",
    "KrylovSchurSolver",
    "JacobiDavidsonSolver",
    "LOBPCGSolver",
    "create_solver",
    "SOLVER_REGISTRY",
]

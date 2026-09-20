"""求解器子包。"""

from .base import EigenSolver, EigenResult
from .lanczos import LanczosSolver
from .krylov_schur import KrylovSchurSolver
from .jacobi_davidson import JacobiDavidsonSolver
from .lobpcg import LOBPCGSolver
from .factory import create_solver, SOLVER_REGISTRY

__all__ = [
    "EigenSolver",
    "EigenResult",
    "LanczosSolver",
    "KrylovSchurSolver",
    "JacobiDavidsonSolver",
    "LOBPCGSolver",
    "create_solver",
    "SOLVER_REGISTRY",
]

"""求解器工厂：根据配置创建合适的求解器实例。

支持自动选择（auto），根据矩阵性质与目标自动挑选最合适的算法。
"""

from __future__ import annotations

from typing import Dict, Optional, Type

import scipy.sparse as sp

from .base import EigenSolver
from .lanczos import LanczosSolver
from .krylov_schur import KrylovSchurSolver
from .jacobi_davidson import JacobiDavidsonSolver
from .lobpcg import LOBPCGSolver
from ..config import SolverConfig

# 求解器注册表
SOLVER_REGISTRY: Dict[str, Type[EigenSolver]] = {
    "lanczos": LanczosSolver,
    "krylov_schur": KrylovSchurSolver,
    "jacobi_davidson": JacobiDavidsonSolver,
    "lobpcg": LOBPCGSolver,
}


def create_solver(
    solver_type: Optional[str] = None,
    matrix: Optional[sp.spmatrix] = None,
    config: Optional[SolverConfig] = None,
) -> EigenSolver:
    """创建求解器实例。

    Args:
        solver_type: 求解器类型。None 则用 config.solver 或自动选择。
        matrix: 输入矩阵（用于自动选择）。
        config: 求解配置。

    Returns:
        EigenSolver 实例。

    Raises:
        ValueError: 未知求解器类型。
    """
    config = config or SolverConfig()
    stype = solver_type or config.solver

    if stype == "auto":
        stype = auto_select(matrix, config)

    if stype not in SOLVER_REGISTRY:
        raise ValueError(
            f"未知求解器: {stype}，可用: {list(SOLVER_REGISTRY.keys())}"
        )

    return SOLVER_REGISTRY[stype](config)


def auto_select(
    matrix: Optional[sp.spmatrix],
    config: Optional[SolverConfig] = None,
) -> str:
    """自动选择求解器。

    选择策略：
    - 目标频段/目标位移（target）：Krylov-Schur（现代稳定算法）
    - 最小特征值（低频击穿场景）：Jacobi-Davidson（对非正定最稳健）
    - 最大特征值：LOBPCG（块迭代，收敛快）
    - 默认：Krylov-Schur

    Args:
        matrix: 输入矩阵。
        config: 求解配置。

    Returns:
        求解器类型字符串。
    """
    config = config or SolverConfig()
    which = config.which
    sigma = config.shift.sigma

    if which == "target" or which == "closest":
        # 近奇异位移风险下使用最稳健算法
        if sigma is not None:
            return "krylov_schur"
        return "krylov_schur"

    if which == "smallest_magnitude":
        # 求零附近特征值（低频击穿场景）：JD 对非正定最稳健
        return "jacobi_davidson"

    if which == "largest_magnitude":
        return "lobpcg"

    return "krylov_schur"
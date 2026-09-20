"""配置系统：定义求解器配置、位移策略、收敛控制与并行后端参数。

所有配置均可通过 YAML / 字典 / 命令行覆盖，禁止硬编码答案。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# 位移策略配置
# ---------------------------------------------------------------------------
@dataclass
class ShiftConfig:
    """位移策略配置。

    位移 σ 对应目标频率（电磁谐振点）。当 σ 接近特征值时 A - σI 近奇异，
    需要自适应位移策略避免数值崩溃。
    """

    #: 目标位移（目标频率）。None 表示自动选择（如取谱中心）。
    sigma: Optional[float] = None

    #: 目标频段 [lo, hi]，用于捕获该频段内的谐振模式。
    target_band: Optional[List[float]] = None

    #: 位移策略：'fixed' | 'adaptive' | 'rayleigh' | 'wilkinson'
    strategy: str = "adaptive"

    #: 自适应位移的更新方式：'rayleigh_quotient' | 'harmonic' | 'spectrum'
    adaptive_mode: str = "rayleigh_quotient"

    #: 自适应位移更新阻尼因子（0~1），避免位移剧烈震荡。
    damping: float = 0.5

    #: 位移更新最小步长，防止位移停滞。
    min_step: float = 1e-6

    #: 位移更新最大步长，防止位移发散。
    max_step: float = 1e3

    #: 近奇异保护阈值：当 |A - σI| 的估计条件数超过该值时触发保护。
    near_singular_threshold: float = 1e12

    #: 是否启用近奇异保护（对 A - σI 做正则化/位移微调）。
    near_singular_protection: bool = True

    def validate(self) -> None:
        if self.strategy not in ("fixed", "adaptive", "rayleigh", "wilkinson"):
            raise ValueError(f"未知位移策略: {self.strategy}")
        if self.adaptive_mode not in ("rayleigh_quotient", "harmonic", "spectrum"):
            raise ValueError(f"未知自适应位移模式: {self.adaptive_mode}")
        if self.target_band is not None:
            if len(self.target_band) != 2 or self.target_band[0] > self.target_band[1]:
                raise ValueError(f"目标频段必须为 [lo, hi] 且 lo<=hi: {self.target_band}")


# ---------------------------------------------------------------------------
# 收敛控制配置
# ---------------------------------------------------------------------------
@dataclass
class ConvergenceConfig:
    """收敛控制配置。"""

    #: 相对残差阈值 ||A x - λ x|| / (||A|| ||x||) <= tol
    tol: float = 1e-8

    #: 特征值相对变化阈值（用于判定特征值稳定）。
    tol_eigval: float = 1e-10

    #: 最大迭代次数（外迭代）。
    max_iter: int = 1000

    #: 最大内迭代次数（如 Jacobi-Davidson 内迭代）。
    max_inner_iter: int = 30

    #: 最大重启次数。
    max_restarts: int = 50

    #: 收敛后额外验证的残差检查次数。
    verify_steps: int = 2

    #: 是否启用自适应迭代策略（根据收敛速度调整）。
    adaptive: bool = True

    #: 收敛停滞检测窗口。
    stagnation_window: int = 10

    #: 停滞阈值：连续窗口内残差下降比例低于该值视为停滞。
    stagnation_tol: float = 1e-3

    def validate(self) -> None:
        if not (0 < self.tol < 1):
            raise ValueError(f"残差阈值必须在 (0,1): {self.tol}")
        if self.max_iter <= 0 or self.max_restarts <= 0:
            raise ValueError("迭代次数必须为正")


# ---------------------------------------------------------------------------
# 并行后端配置
# ---------------------------------------------------------------------------
@dataclass
class ParallelConfig:
    """并行/加速后端配置。

    支持多线程（OpenMP/BLAS）、MPI、GPU（CUDA）。默认单机多核 CPU。
    缺少 GPU 时自动降级到 CPU 模式（最小化降级模式）。
    """

    #: 后端类型：'cpu' | 'threads' | 'mpi' | 'gpu' | 'auto'
    backend: str = "auto"

    #: 线程数（threads 后端）。None 表示使用 BLAS 默认。
    num_threads: Optional[int] = None

    #: 是否启用 GPU（若可用）。
    use_gpu: bool = False

    #: GPU 设备编号。
    gpu_device: int = 0

    #: 内存峰值上限（字节）。None 表示不限制。
    memory_limit: Optional[int] = None

    #: 是否允许降级（GPU 不可用时回退 CPU）。
    allow_fallback: bool = True

    def validate(self) -> None:
        if self.backend not in ("cpu", "threads", "mpi", "gpu", "auto"):
            raise ValueError(f"未知并行后端: {self.backend}")


# ---------------------------------------------------------------------------
# 求解器配置
# ---------------------------------------------------------------------------
@dataclass
class SolverConfig:
    """求解器总配置。"""

    #: 求解器类型：'lanczos' | 'krylov_schur' | 'jacobi_davidson' | 'lobpcg' | 'auto'
    solver: str = "auto"

    #: 需要求解的特征值个数（目标谐振模式数）。
    n_eigenvalues: int = 6

    #: 需要求解的特征值个数（目标谐振模式数）。
    which: str = "target"  # 'target' | 'smallest_magnitude' | 'largest_magnitude' | 'closest'

    #: 位移配置。
    shift: ShiftConfig = field(default_factory=ShiftConfig)

    #: 收敛控制配置。
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)

    #: 并行后端配置。
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    #: 预处理配置。
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "scale": True,          # 对角缩放
        "balance": False,       # 平衡（对非对称矩阵）
        "sort": True,           # 按特征值排序
        "shift_invert": False,  # 是否使用位移求逆
    })

    #: 是否启用安全模式（脱敏、防硬编码检查）。
    security: bool = True

    #: 随机种子（可复现性）。
    seed: Optional[int] = 42

    #: 是否输出诊断信息。
    verbose: bool = True

    #: 日志级别。
    log_level: str = "INFO"

    def validate(self) -> None:
        if self.solver not in ("lanczos", "krylov_schur", "jacobi_davidson", "lobpcg", "auto"):
            raise ValueError(f"未知求解器: {self.solver}")
        if self.n_eigenvalues <= 0:
            raise ValueError("n_eigenvalues 必须为正")
        if self.which not in ("target", "smallest_magnitude", "largest_magnitude", "closest"):
            raise ValueError(f"未知 which 参数: {self.which}")
        self.shift.validate()
        self.convergence.validate()
        self.parallel.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolverConfig":
        """从字典构建配置，支持嵌套覆盖。"""
        cfg = cls()
        for key, value in data.items():
            if not hasattr(cfg, key):
                continue
            current = getattr(cfg, key)
            if isinstance(current, (ShiftConfig, ConvergenceConfig, ParallelConfig)):
                if isinstance(value, dict):
                    setattr(cfg, key, type(current).from_dict(value))
                else:
                    setattr(cfg, key, value)
            else:
                setattr(cfg, key, value)
        cfg.validate()
        return cfg


# 为嵌套 dataclass 添加 from_dict 类方法
def _add_from_dict(cls):
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        obj = cls()
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        obj.validate()
        return obj
    cls.from_dict = from_dict
    return cls


ShiftConfig = _add_from_dict(ShiftConfig)
ConvergenceConfig = _add_from_dict(ConvergenceConfig)
ParallelConfig = _add_from_dict(ParallelConfig)


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_config(path: Optional[str] = None, **overrides) -> SolverConfig:
    """从 YAML 文件加载配置，并应用命令行/关键字覆盖。

    Args:
        path: YAML 配置文件路径。None 表示使用默认配置。
        **overrides: 关键字覆盖，如 solver='lobpcg', n_eigenvalues=10。

    Returns:
        SolverConfig 实例。
    """
    data: Dict[str, Any] = {}
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # 应用覆盖
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    cfg = SolverConfig.from_dict(data)
    return cfg


def default_config() -> SolverConfig:
    """返回默认配置。"""
    return SolverConfig()

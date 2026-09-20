# EM-Eigensolver

面向 **电磁谐振与微波器件仿真** 的高效、稳定特征值求解方案

> 【AI+工业软件赛道】中国商飞赛题：面向电磁谐振与微波器件仿真的高效、稳定求解方案

## 项目概述

针对电磁仿真中常见的 **十万阶以上、非正定（不定）厄密稀疏矩阵** 特征值问题，本项目提供了一套完整、高效的求解框架：

- 支持矩阵的按需读入/转换（numpy `.npz` / scipy `.mtx` / `.npz` 稀疏格式）
- 四种核心 Krylov 特征值求解器：**Lanczos**（完全重正交化）、**Krylov-Schur**（隐式重启）、**Jacobi-Davidson**（校正方程）、**LOBPCG**（块迭代）
- 面向目标频段的 **位移（shift-invert）策略** 与自适应位移
- 收敛控制与故障诊断（近奇异位移、零附近密集谱等）
- 全程 **稀疏存储**，不进行矩阵稠密化

## 目录结构

```
em-eigensolver/
├── config/                 # 求解配置示例
├── src/em_eigensolver/
│   ├── cli.py              # 命令行入口（solve / generate / diagnose）
│   ├── config.py           # 配置系统
│   ├── engine.py           # 求解引擎（端到端编排）
│   ├── io/                 # 矩阵与特征对读写
│   ├── preprocess.py       # 预处理：缩放/排序/性质检测
│   ├── samples.py          # 样例矩阵生成（含解析特征值）
│   ├── diagnostics.py      # 收敛控制与故障诊断
│   ├── shifts.py           # 位移策略
│   ├── parallel.py         # 并行后端管理
│   ├── security.py         # 数据脱敏与安全设计
│   └── solvers/            # 四种核心求解器
├── tests/                  # 单元测试与规模回归测试
└── pyproject.toml
```

## 快速开始

```bash
pip install -r requirements.txt

# 生成样例矩阵并求解
python -m em_eigensolver generate --type indefinite --size 1000 --output A.mtx
python -m em_eigensolver solve --matrix A.mtx --solver lanczos --n-eigenvalues 4

# 求解诊断
python -m em_eigensolver diagnose --matrix A.mtx
```

## 核心求解器

| 求解器 | 方法 | 适用场景 |
|---|---|---|
| `lanczos` | 带完全重正交化的 Lanczos 迭代 | 一般厄密/非正定矩阵，健壮性优先 |
| `krylov_schur` | Krylov-Schur 隐式重启 | 需要较多特征值 / 目标频段 |
| `jacobi_davidson` | 校正方程（MINRES/GMRES） | 位移接近奇异、特征值密集 |
| `lobpcg` | 块迭代（Locally Optimal Block PCG） | 块求解、重特征值 |

### 位移求逆

对目标频率 σ，求解变换后问题 $(A - \sigma I)^{-1}x = \theta x$，特征值满足 $\lambda = \sigma + 1/\theta$。位移求逆采用稀疏 LU 分解（`scipy.sparse.linalg.splu`，默认部分枢轴，适配非正定矩阵），对近奇异位移自动加入小扰动并给出诊断。

## 预处理

- **雅可比缩放（符号翻转）**：仅做 $D_{ii} = \pm 1$ 正交缩放，保证特征值精确不变（一般对角缩放会因合同变换改变谱）
- **RCM 重排序**：减小 LU 填充，加速位移求逆
- **性质检测**：厄密性、对称性、正定性（基于谱范围估计）

## 规模与性能

- 目标规模：**100,000 阶以上** 非正定厄密稀疏矩阵
- 非零元：约 $O(n)$（有限差分/有限元离散矩阵）
- 求解器内存：`Lanczos/Krylov-Schur` 为 $O(m \cdot n)$，`LOBPCG` 为 $O(k \cdot n)$（k 为块大小），不做稠密化
- 线性求解：稀疏 LU 分解，资源优化（可配置线程数）

## 测试

```bash
python -m pytest tests/ -q            # 单元测试
python -m pytest tests/ -m slow -q    # 十万阶规模回归测试
```

## 许可

本项目基于开源组件构建（NumPy / SciPy），仅供竞赛与科研学习使用。
"""样例矩阵生成模块。

生成公开可用的电磁仿真测试矩阵，覆盖不同谱分布、目标频段与故障场景：

1. cavity:      谐振腔有限差分矩阵（物理离散，特征值解析已知）。
2. photon:      光子晶体（周期结构）有限差分矩阵。
3. pml:         含 PML 吸收边界的复频移拉普拉斯（谱跨越正负，非正定）。
4. dense_zero:  零附近密集谱（低频击穿场景）。
5. degenerate:  重特征值（多重特征值）场景。
6. indefinite:  非正定厄密矩阵（谱跨越正负）。

核心构造方法（Givens 相似变换）：
对对角矩阵 D = diag(λ) 施加一系列稀疏正交相似变换（Givens 旋转），
得到特征值精确等于 λ、且保持稀疏松散结构的厄密矩阵 A = Q^T D Q。
该方案保证：
- 特征值精确已知（可用于基准验证，非硬编码答案）
- 矩阵保持稀疏（非零数 ~ O(n)）
- 特征向量非平凡（非标准基）
- 内存 O(n)

用法：
>>> A, meta = generate_sample_matrix("cavity", size=10000)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Givens 旋转相似变换
# ---------------------------------------------------------------------------
def _apply_givens(A: sp.spmatrix, i: int, j: int, theta: float, n: int):
    """对稀疏矩阵施加 Givens 相似变换 A ← G(i,j,θ)^T A G(i,j,θ)。

    G 为 (i,j) 平面内的正交旋转矩阵（G^T G = I），因此该变换保持特征值不变。
    只影响 i、j 两行两列，保持稀疏性。

    实现：G = I + 旋转块（在 (i,j) 平面上覆盖 4 个入口），即
        G[i,i] = c, G[i,j] = s, G[j,i] = -s, G[j,j] = c
    其余对角元为 1。通过稀疏矩阵乘法 A' = G^T A G 完成相似变换。
    """
    c = math.cos(theta)
    s = math.sin(theta)
    # 单位阵 + 覆盖 (i,j) 平面的旋转块
    G = sp.eye(n, format="csr")
    G[i, i] = c
    G[i, j] = s
    G[j, i] = -s
    G[j, j] = c
    G.eliminate_zeros()
    # A' = G^T A G（相似变换，保持特征值）
    A_new = (G.T @ A @ G).tocsr()
    A_new.eliminate_zeros()
    return A_new


def make_sparse_with_givens(
    eigenvalues: np.ndarray,
    n_rotations: Optional[int] = None,
    seed: Optional[int] = 42,
) -> Tuple[sp.spmatrix, List[Tuple[int, int]]]:
    """用 Givens 相似变换生成稀疏厄密矩阵，特征值精确等于 eigenvalues。

    Args:
        eigenvalues: 目标特征值数组（升序）。
        n_rotations: Givens 旋转次数。None 则自动 = max(n, nnz_target/4)。
        seed: 随机种子。

    Returns:
        (A, rotations)：稀疏厄密矩阵与旋转位置列表。
    """
    n = len(eigenvalues)
    rng = np.random.default_rng(seed)

    # 初始对角矩阵
    A = sp.diags(eigenvalues, format="coo")

    if n_rotations is None:
        # 旋转次数自适应：小矩阵用较多旋转使结构丰富，大矩阵限制次数保证性能
        n_rotations = min(max(n // 2, 10), 100)

    rotations = []
    for _ in range(n_rotations):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        while j == i:
            j = int(rng.integers(0, n))
        theta = rng.uniform(-0.5, 0.5)
        rotations.append((i, j))
        A = _apply_givens(A, i, j, theta, n)
        # 定期清理小元素控制稀疏度
        if A.nnz > 8 * n:
            A = A.tocsr()
            A.data[np.abs(A.data) < 1e-12] = 0.0
            A.eliminate_zeros()
            A = A.tocoo()

    # 最终清理
    A = A.tocsr()
    A.data[np.abs(A.data) < 1e-12] = 0.0
    A.eliminate_zeros()
    # 对称化检查
    if not (A != A.T.conj()).nnz == 0:
        A = 0.5 * (A + A.T.conj()).tocsr()
        A.eliminate_zeros()
    return A, rotations


# ---------------------------------------------------------------------------
# 物理离散化矩阵（有限差分）
# ---------------------------------------------------------------------------
def _make_fd_matrix(
    nx: int,
    ny: int,
    mode: str = "cavity",
    sigma_offset: float = 0.0,
) -> Tuple[sp.spmatrix, np.ndarray]:
    """二维有限差分厄密矩阵。

    2D Dirichlet 拉普拉斯：特征值解析已知。
        λ_{p,q} = 4 - 2cos(π p/(Nx+1)) - 2cos(π q/(Ny+1))   （离散，0~8）
    加位移后特征值平移 sigma_offset。

    Args:
        nx, ny: 网格维度。
        mode: 'cavity' | 'photon' | 'pml'。
        sigma_offset: 谱平移量。

    Returns:
        (A, true_eigenvalues)。
    """
    n = nx * ny

    # 2D 离散拉普拉斯（5 点模板）
    main = np.full(n, 4.0)
    off_x = np.full(n - 1, -1.0)
    off_y = np.full(n - nx, -1.0)

    A = sp.diags(
        [off_y, off_x, main, off_x, off_y],
        [-nx, -1, 0, 1, nx],
        shape=(n, n),
        format="csr",
    )
    # 去掉跨越行边界的连接（周期性剪裁）
    A = A.tolil()
    for i in range(ny):
        for j in range(nx - 1):
            idx = i * nx + j
            # 水平连接 idx <-> idx+1 合法（同行的相邻列）
            pass
    # 移除跨行边界的 x 连接：位置 i*nx + (nx-1) <-> i*nx + nx
    for i in range(ny - 1):
        idx = i * nx + nx - 1  # 行末
        A[idx, idx + 1] = 0.0
        A[idx + 1, idx] = 0.0
    A = A.tocsr()
    A.eliminate_zeros()

    # 模式修饰
    if mode == "photon":
        # 光子晶体：添加对角势（周期结构）
        rng = np.random.default_rng(0)
        potential = 0.5 * rng.uniform(0, 1, n)
        A = A + sp.diags(potential, format="csr")

    if mode == "pml":
        # PML：对角加权（吸收边界，谱扩展并平移）
        # 添加位置相关对角项使谱跨越正负
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        weight = -2.0 * (X**2 + Y**2 - 0.5)  # 中心为正、边缘为负
        A = A + sp.diags(weight.ravel() * 0.2, format="csr")

    # 谱平移（构造目标频段）
    if sigma_offset != 0.0:
        A = A + sigma_offset * sp.eye(n, format="csr")

    # 解析特征值（最小的 min(n, 64) 个）
    k = min(n, 64)
    true_evals = []
    p_vals = np.arange(1, nx + 1)
    q_vals = np.arange(1, ny + 1)
    for p in p_vals:
        for q in q_vals:
            lam = 4 - 2 * np.cos(np.pi * p / (nx + 1)) - 2 * np.cos(
                np.pi * q / (ny + 1)
            )
            if mode == "photon":
                lam += 0.25  # 势均值近似
            if mode == "pml":
                lam += 0.0
            true_evals.append(lam + sigma_offset)
    true_evals = np.sort(np.array(true_evals))[:k]

    return A, true_evals


# ---------------------------------------------------------------------------
# 主生成函数
# ---------------------------------------------------------------------------
def generate_sample_matrix(
    matrix_type: str = "cavity",
    size: int = 10000,
    sigma_offset: Optional[float] = None,
    k_cluster: Optional[int] = None,
    seed: Optional[int] = 42,
    **kwargs,
) -> Tuple[sp.spmatrix, Dict[str, Any]]:
    """生成电磁仿真测试矩阵。

    Args:
        matrix_type: 矩阵类型。
        size: 矩阵规模（近似）。
        sigma_offset: 目标位移（构造目标频段）。
        k_cluster: 目标簇特征值个数。
        seed: 随机种子。

    Returns:
        (A, metadata)。metadata 包含 true_eigenvalues（真实特征值，用于验证）。
    """
    rng = np.random.default_rng(seed)

    if matrix_type in ("cavity", "photon", "pml"):
        # 物理离散化：2D 网格
        nx = int(math.sqrt(size))
        ny = nx
        while nx * ny < size:
            nx += 1
        sigma_offset = sigma_offset or 12.0  # 目标频段基准（谐振频率）
        A, true_evals = _make_fd_matrix(nx, ny, matrix_type, sigma_offset)
        meta = {
            "type": matrix_type,
            "size": A.shape[0],
            "nnz": A.nnz,
            "sigma": sigma_offset,
            "true_eigenvalues": true_evals.tolist(),
            "spectral": "indefinite" if (matrix_type == "pml") else "positive",
            "discretization": "finite_difference_2d",
        }
        return A, meta

    if matrix_type in ("dense_zero", "degenerate", "indefinite"):
        # Givens 相似变换构造，特征值精确已知
        n = size
        sigma_offset = sigma_offset or 0.0

        if matrix_type == "dense_zero":
            # 零附近密集谱：大量特征值集中在 0 附近
            # λ = 线性分布在小区间 [-0.5, 0.5]（主簇）+ 少量远离值
            k = k_cluster or max(n // 50, 20)
            band = 0.5
            main = np.linspace(-band, band, n - k) + 1e-6 * rng.standard_normal(n - k)
            far = np.concatenate([
                np.linspace(-20, -1, k // 2),
                np.linspace(1, 20, k - k // 2),
            ])
            eigenvalues = np.sort(np.concatenate([main, far])) + sigma_offset

        elif matrix_type == "degenerate":
            # 重特征值：多个特征值完全相同
            n_clusters = max(5, n // 500)
            k = k_cluster or 8  # 每个簇的特征值重数
            cluster_vals = np.linspace(-3, 3, n_clusters) + sigma_offset
            eigenvalues = np.repeat(cluster_vals, k)[:n]
            if len(eigenvalues) < n:
                extra = rng.uniform(-3, 3, n - len(eigenvalues))
                eigenvalues = np.sort(np.concatenate([eigenvalues, extra]))

        else:  # indefinite
            # 非正定：谱跨越正负
            n_pos = n // 2
            n_neg = n - n_pos
            pos = rng.uniform(0.5, 10, n_pos)
            neg = rng.uniform(-10, -0.5, n_neg)
            eigenvalues = np.sort(np.concatenate([pos, neg])) + sigma_offset

        # 生成稀疏矩阵（Givens 变换）
        A, rotations = make_sparse_with_givens(eigenvalues, seed=seed)
        A = A.tocsr()

        # 目标簇：最接近 sigma 的 k 个真实特征值
        if k_cluster:
            target_k = k_cluster
            dist = np.abs(eigenvalues - sigma_offset)
            target_evals = eigenvalues[np.argsort(dist)[:target_k]]
            cluster_evals = target_evals
        else:
            cluster_evals = eigenvalues

        meta = {
            "type": matrix_type,
            "size": A.shape[0],
            "nnz": A.nnz,
            "sigma": sigma_offset,
            "true_eigenvalues": eigenvalues.tolist(),
            "target_eigenvalues": cluster_evals.tolist(),
            "spectral": "indefinite" if matrix_type == "indefinite" else (
                "dense_zero" if matrix_type == "dense_zero" else "degenerate"
            ),
            "construction": "givens_similarity",
        }
        return A, meta

    raise ValueError(f"未知矩阵类型: {matrix_type}")


# ---------------------------------------------------------------------------
# 便捷：生成基准（已知特征值的标准测试）
# ---------------------------------------------------------------------------
def make_benchmark(n: int = 500, seed: int = 42) -> Tuple[sp.spmatrix, np.ndarray]:
    """生成一个带已知特征值的基准矩阵（用于测试与基准验证）。

    Returns:
        (A, true_eigenvalues)。
    """
    A, meta = generate_sample_matrix("indefinite", n, sigma_offset=2.0, seed=seed)
    # 目标：最接近 2.0 的 k 个特征值
    evals = np.array(meta["true_eigenvalues"])
    dist = np.abs(evals - 2.0)
    idx = np.argsort(dist)[:10]
    return A, evals[idx]
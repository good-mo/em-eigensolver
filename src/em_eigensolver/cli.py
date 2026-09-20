"""命令行入口：em-eig。

用法：
    em-eig solve <matrix_file> [options]      求解特征值
    em-eig generate <output_dir> [options]   生成样例矩阵
    em-eig diagnose <matrix_file> [options]   诊断矩阵与故障场景
    em-eig audit <source_dir>                 安全审计
    em-eig config [output_file]               输出默认配置

示例：
    em-eig solve cavity.npz --n 6 --sigma 12.5 --solver auto
    em-eig generate ./examples --max-size 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

# 允许从源码直接运行（不安装）
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from .config import load_config, SolverConfig, default_config
from .engine import EMESolver
from .io import load_matrix, save_matrix
from .samples import generate_sample_matrix
from .diagnostics import diagnose_matrix, analyze_spectrum
from .security import audit_security


def _parse_common(parser: argparse.ArgumentParser) -> None:
    """添加通用参数。"""
    parser.add_argument("--config", type=str, default=None, help="配置文件路径 (YAML)")
    parser.add_argument("--n", "--n-eigenvalues", type=int, default=None, dest="n",
                        help="需要求解的特征值个数")
    parser.add_argument("--sigma", type=float, default=None, help="目标位移/频率")
    parser.add_argument("--which", type=str, default=None,
                        choices=["target", "smallest_magnitude", "largest_magnitude", "closest"],
                        help="目标类型")
    parser.add_argument("--solver", type=str, default=None,
                        choices=["auto", "lanczos", "krylov_schur", "jacobi_davidson", "lobpcg"],
                        help="求解器类型")
    parser.add_argument("--tol", type=float, default=None, help="残差阈值")
    parser.add_argument("--target-band", type=str, default=None,
                        help="目标频段，如 '10,20'")
    parser.add_argument("--shift-strategy", type=str, default=None,
                        choices=["fixed", "adaptive", "rayleigh", "wilkinson"],
                        help="位移策略")
    parser.add_argument("--threads", type=int, default=None, help="线程数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--verbose", action="store_true", default=None,
                        help="输出详细信息")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="安静模式")


def _build_overrides(args: argparse.Namespace) -> dict:
    """从 argparse 构建配置覆盖。"""
    overrides = {}
    if args.config:
        overrides["_config_path"] = args.config
    if args.n is not None:
        overrides["n_eigenvalues"] = args.n
    if args.solver is not None:
        overrides["solver"] = args.solver
    if args.tol is not None:
        overrides.setdefault("convergence", {})["tol"] = args.tol
    if args.threads is not None:
        overrides.setdefault("parallel", {})["num_threads"] = args.threads
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.verbose:
        overrides["verbose"] = True
    if args.target_band is not None:
        band = [float(x) for x in args.target_band.split(",")]
        overrides.setdefault("shift", {})["target_band"] = band
    if args.shift_strategy is not None:
        overrides.setdefault("shift", {})["strategy"] = args.shift_strategy
    return overrides


def cmd_solve(args: argparse.Namespace) -> int:
    """执行求解命令。"""
    overrides = _build_overrides(args)
    config_path = overrides.pop("_config_path", None)
    try:
        config = load_config(config_path, **overrides)
    except Exception as e:
        print(f"[ERROR] 配置加载失败: {e}", file=sys.stderr)
        return 1

    solver = EMESolver(config)
    output_dir = args.output if hasattr(args, "output") and args.output else None

    try:
        result = solver.solve_file(
            args.matrix,
            n_eigenvalues=args.n,
            sigma=args.sigma,
            which=args.which,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"[ERROR] 求解失败: {e}", file=sys.stderr)
        return 1

    # 输出结果摘要
    print("=" * 60)
    print("电磁特征值求解结果")
    print("=" * 60)
    print(f"矩阵:        {args.matrix}")
    print(f"形状:        {solver.report.matrix_shape}")
    print(f"nnz:         {solver.report.nnz}")
    print(f"厄密:        {solver.report.is_hermitian}")
    print(f"求解器:      {solver.report.solver}")
    print(f"位移 σ:      {solver.report.sigma}")
    print(f"收敛:        {solver.report.converged} "
          f"({solver.report.n_converged}/{len(solver.report.eigenvalues)})")
    print(f"迭代次数:    {solver.report.n_iterations}")
    print(f"矩阵乘次数:  {solver.report.n_matvecs}")
    print(f"耗时:        {solver.report.elapsed_time:.3f}s")
    print(f"峰值内存:    {solver.report.peak_memory_mb:.1f} MB")
    if solver.report.warnings:
        print("警告:")
        for w in solver.report.warnings:
            print(f"  - {w}")
    print("-" * 60)
    print("特征值（按升序）:")
    evals = sorted(solver.report.eigenvalues)
    resids = solver.report.residuals
    order = [solver.report.eigenvalues.index(v) for v in evals]
    for i, v in enumerate(evals):
        r = solver.report.residuals[order[i]]
        mark = "✓" if r <= solver.config.convergence.tol else "!"
        print(f"  [{mark}] λ_{i} = {v:+.10e}  (残差 {r:.2e})")

    if output_dir:
        print(f"\n结果已保存到: {output_dir}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """生成样例矩阵。"""
    matrix_type = args.type or "cavity"
    size = args.size or 10000
    k_eigs = args.k
    sigma_shift = args.sigma

    print(f"生成样例矩阵: type={matrix_type}, size={size}, k={k_eigs}, sigma={sigma_shift}")
    os.makedirs(args.output, exist_ok=True)

    try:
        A, meta = generate_sample_matrix(
            matrix_type=matrix_type,
            size=size,
            sigma_offset=sigma_shift,
            k_cluster=k_eigs,
            seed=args.seed or 42,
        )
    except Exception as e:
        print(f"[ERROR] 生成失败: {e}", file=sys.stderr)
        return 1

    path = os.path.join(args.output, f"{matrix_type}_{size}.npz")
    save_matrix(A, path)
    print(f"矩阵已保存: {path} (nnz={A.nnz}, shape={A.shape})")
    print(f"元数据: {meta}")

    # 保存元数据
    meta_path = os.path.join(args.output, f"{matrix_type}_{size}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"元数据已保存: {meta_path}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """诊断矩阵。"""
    matrix = load_matrix(args.matrix)
    result = diagnose_matrix(
        matrix,
        sigma=args.sigma,
        n_eigenvalues=args.n,
        memory_limit=args.memory,
    )
    print("=" * 60)
    print("故障场景诊断")
    print("=" * 60)
    print(f"矩阵:   {args.matrix} 形状 {matrix.shape}  nnz {matrix.nnz}")
    if result.scenarios:
        print(f"识别场景: {result.scenarios}")
    for w in result.warnings:
        print(f"警告: {w}")
    for r in result.recommendations:
        print(f"建议: {r}")
    print(f"指标: {json.dumps({k: v for k, v in result.metrics.items() if k != 'spectrum'}, indent=2)}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """安全审计。"""
    audit = audit_security(args.source)
    print("=" * 60)
    print("安全审计")
    print("=" * 60)
    print(f"通过: {audit.passed}")
    print(f"检查项: {audit.checks}")
    for issue in audit.issues:
        print(f"[FAIL] {issue}")
    for w in audit.warnings:
        print(f"[WARN] {w}")
    return 0 if audit.passed else 1


def cmd_config(args: argparse.Namespace) -> int:
    """输出默认配置。"""
    import yaml

    cfg = default_config()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)
        print(f"配置已写入: {args.output}")
    else:
        print(yaml.dump(cfg.to_dict(), default_flow_style=False, allow_unicode=True))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """主入口。"""
    parser = argparse.ArgumentParser(
        prog="em-eig",
        description="面向电磁谐振仿真的非正定厄密稀疏矩阵特征值求解器",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # solve
    p_solve = sub.add_parser("solve", help="求解矩阵特征值")
    p_solve.add_argument("matrix", help="矩阵文件路径")
    _parse_common(p_solve)
    p_solve.add_argument("-o", "--output", default=None, help="输出目录")
    p_solve.set_defaults(func=cmd_solve)

    # generate
    p_gen = sub.add_parser("generate", help="生成样例矩阵")
    p_gen.add_argument("output", help="输出目录")
    p_gen.add_argument("--type", default=None,
                       choices=["cavity", "photon", "pml", "dense_zero", "degenerate", "indefinite"],
                       help="矩阵类型")
    p_gen.add_argument("--size", type=int, default=None, help="矩阵规模 (默认 10000)")
    p_gen.add_argument("--k", type=int, default=None, help="目标簇特征值数")
    p_gen.add_argument("--sigma", type=float, default=None, help="目标位移（构造目标频段）")
    p_gen.add_argument("--seed", type=int, default=None, help="随机种子")
    p_gen.set_defaults(func=cmd_generate)

    # diagnose
    p_diag = sub.add_parser("diagnose", help="诊断矩阵与故障场景")
    p_diag.add_argument("matrix", help="矩阵文件路径")
    p_diag.add_argument("--sigma", type=float, default=None, help="目标位移")
    p_diag.add_argument("--n", type=int, default=None, help="特征值个数")
    p_diag.add_argument("--memory", type=int, default=None, help="内存上限（字节）")
    p_diag.set_defaults(func=cmd_diagnose)

    # audit
    p_audit = sub.add_parser("audit", help="安全审计")
    p_audit.add_argument("source", help="源代码目录")
    p_audit.set_defaults(func=cmd_audit)

    # config
    p_cfg = sub.add_parser("config", help="输出默认配置")
    p_cfg.add_argument("output", nargs="?", default=None, help="配置文件路径")
    p_cfg.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
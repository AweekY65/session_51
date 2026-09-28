"""数学工具函数：角度规范化与协方差合法性校验."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """将角度规范化到 [-pi, pi)，支持标量或数组（向量化）。

    跨越 +-pi 的航向差应使用本函数处理，避免出现 2*pi 量级的错误跳变。
    约定 -pi 与 +pi 等价，统一取 -pi（numpy 取模的自然结果）。
    """
    a = np.asarray(angle, dtype=float)
    return ((a + np.pi) % (2.0 * np.pi) - np.pi).item() if a.ndim == 0 \
        else (a + np.pi) % (2.0 * np.pi) - np.pi


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """wrap_angle 的语义化别名。"""
    return wrap_angle(angle)


def validate_covariance(
    P: np.ndarray,
    dim: int | None = None,
    name: str = "covariance",
    eps: float = 1e-12,
) -> np.ndarray:
    """校验矩阵是否为合法协方差：维度正确、有限、对称、半正定。

    参数
    ----
    P : 待校验矩阵
    dim : 期望维度（方阵边长）；None 表示只检查方阵
    name : 报错时使用的名称，便于定位
    eps : 对称性容差，以及最小特征值判定容差

    异常
    ----
    EKFError: 维度错误 / 含 NaN 或 Inf / 非对称 / 非半正定
    """
    # 延迟导入避免循环依赖
    from .ekf import EKFError

    P = np.asarray(P, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise EKFError(
            f"{name} 必须是二维方阵，实际 shape={P.shape}"
        )
    if dim is not None and P.shape[0] != dim:
        raise EKFError(
            f"{name} 维度错误：期望 {dim}x{dim}，实际 {P.shape[0]}x{P.shape[1]}"
        )
    if not np.all(np.isfinite(P)):
        raise EKFError(f"{name} 含 NaN 或 Inf，属于非法 covariance")

    asym = np.max(np.abs(P - P.T))
    if asym > eps * max(1.0, np.max(np.abs(P))):
        raise EKFError(f"{name} 非对称：最大不对称量 {asym:.3e} 超过容差 {eps:.0e}")

    # 对称化后做特征值检查（eigvalsh 要求对称矩阵）
    Psym = 0.5 * (P + P.T)
    eigvals = np.linalg.eigvalsh(Psym)
    if np.min(eigvals) < -eps * max(1.0, np.max(np.abs(eigvals))):
        raise EKFError(
            f"{name} 非正定（最小特征值 {np.min(eigvals):.3e} < 0）"
        )

    return Psym

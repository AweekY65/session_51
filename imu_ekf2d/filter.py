"""二维 IMU 扩展卡尔曼滤波器。

状态向量 (8 维)
----------------
``x = [px, py, vx, vy, yaw, bax, bay, bg]``

============  ============================================
分量          含义
============  ============================================
px, py        世界系下二维位置 [m]
vx, vy        世界系下二维速度 [m/s]
yaw           航向角 [rad]，恒规范化到 (-pi, pi]
bax, bay      机体系加速度零偏 [m/s^2]
bg            角速度零偏 [rad/s]
============  ============================================

所有矩阵使用 numpy 计算，整个模块只依赖 Python 标准库与 numpy，
不依赖 ROS、GPS、硬件驱动或任何网络服务。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# 状态索引
# ---------------------------------------------------------------------------
STATE_DIM = 8
PX, PY, VX, VY, YAW, BAX, BAY, BG = range(8)


def wrap_pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """把角度规范化到 (-pi, pi]，支持标量与 numpy 数组。"""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def validate_covariance(
    matrix: np.ndarray,
    name: str = "covariance",
    dim: Optional[int] = None,
    positive_semidefinite: bool = False,
    eps: float = 1e-10,
) -> np.ndarray:
    """校验协方差矩阵。

    检查：二维、方阵、维度匹配（如给定）、元素全有限、对称、
    正定（默认）或半正定。非法时抛出 :class:`ValueError`。
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"{name} 必须是二维方阵，实际形状为 {m.shape}")
    if dim is not None and m.shape[0] != dim:
        raise ValueError(f"{name} 维度应为 {dim}，实际为 {m.shape[0]}")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    if not np.allclose(m, m.T, atol=1e-8, rtol=1e-7):
        raise ValueError(f"{name} 不是对称矩阵")

    sym = 0.5 * (m + m.T)
    eigvals = np.linalg.eigvalsh(sym)
    tol = eps * max(1.0, float(np.max(np.abs(eigvals))))
    if positive_semidefinite:
        if float(np.min(eigvals)) < -tol:
            raise ValueError(
                f"{name} 不是半正定矩阵，最小特征值={float(np.min(eigvals)):.3e}"
            )
    else:
        if float(np.min(eigvals)) <= tol:
            raise ValueError(
                f"{name} 不是正定矩阵，最小特征值={float(np.min(eigvals)):.3e}"
            )
    return sym


# 常用卡方分布分位数表（近似值），按自由度索引，概率 0.95 / 0.99。
_CHI2_QUANTILES = {
    0.95: {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
           6: 12.592, 7: 14.067, 8: 15.507},
    0.99: {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086,
           6: 16.812, 7: 18.475, 8: 20.090},
}


def _chi2_ppf(p: float, dof: int) -> float:
    """卡方分布分位数。优先使用内置表，否则用 Wilson-Hilferty 近似。"""
    table = _CHI2_QUANTILES.get(p)
    if table is not None and dof in table:
        return table[dof]
    # Wilson-Hilferty 变换（对 dof >= 1 足够准确）
    z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263, 0.999: 3.0902}.get(p)
    if z is None:
        # 普通正态分位数回退：用 0.99
        z = 2.3263
    h = 2.0 / (9.0 * dof)
    return dof * (1.0 - h + z * math.sqrt(h)) ** 3


@dataclass
class NoiseConfig:
    """可配置的过程噪声参数（全部为连续时间功率谱密度 PSD）。

    ================  =================================  =================
    参数              含义                               单位
    ================  =================================  =================
    acc_psd           加速度计白噪声 PSD (各轴同向)      m^2/s^3
    gyro_psd          陀螺白噪声 PSD                     rad^2/s
    acc_bias_psd      加速度零偏随机游走 PSD (各轴同向)  (m/s^2)^2/s
    gyro_bias_psd     陀螺零偏随机游走 PSD               rad^2/s^3
    ================  =================================  =================
    """

    acc_psd: float = 0.05
    gyro_psd: float = 1e-4
    acc_bias_psd: float = 1e-4
    gyro_bias_psd: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("acc_psd", "gyro_psd", "acc_bias_psd", "gyro_bias_psd"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"过程噪声 {name} 必须是非负有限数，得到 {value}")


@dataclass
class UpdateResult:
    """一次观测更新的结果，便于测试与诊断。"""

    accepted: bool
    innovation: np.ndarray
    S: np.ndarray
    K: np.ndarray
    mahalanobis: float
    gate_threshold: float
    predicted_measurement: np.ndarray
    reason: str = ""

    @property
    def innov_norm(self) -> float:
        return float(np.linalg.norm(self.innovation))


def _rotation_2d(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


# 默认初始协方差，比较宽松。
_DEFAULT_P0 = np.diag([1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.04])


class EKF2D:
    """二维 IMU 姿态与运动状态 EKF。"""

    def __init__(
        self,
        initial_state: Optional[Sequence[float]] = None,
        initial_covariance: Optional[np.ndarray] = None,
        noise_config: Optional[NoiseConfig] = None,
    ) -> None:
        if initial_state is None:
            self.x = np.zeros(STATE_DIM, dtype=float)
        else:
            self.x = np.array(initial_state, dtype=float).reshape(-1)
            if self.x.shape[0] != STATE_DIM:
                raise ValueError(
                    f"初始状态维度应为 {STATE_DIM}，实际为 {self.x.shape[0]}"
                )
            if not np.all(np.isfinite(self.x)):
                raise ValueError("初始状态包含 NaN 或 Inf")
        self.x[YAW] = wrap_pi(self.x[YAW])

        if initial_covariance is None:
            self.P = _DEFAULT_P0.copy()
        else:
            self.P = validate_covariance(
                initial_covariance, "initial_covariance", STATE_DIM,
                positive_semidefinite=True,
            ).copy()

        self.noise = noise_config if noise_config is not None else NoiseConfig()

    # ------------------------------------------------------------------
    # 便捷访问
    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[PX:PY + 1].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[VX:VY + 1].copy()

    @property
    def yaw(self) -> float:
        return float(self.x[YAW])

    @property
    def accel_bias(self) -> np.ndarray:
        return self.x[BAX:BAY + 1].copy()

    @property
    def gyro_bias(self) -> float:
        return float(self.x[BG])

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def predict(self, ax: float, ay: float, wz: float, dt: float) -> None:
        """根据一组机体系 IMU 样本执行 EKF 预测。

        参数
        ----
        ax, ay : 机体系比力（加速度）测量 [m/s^2]
        wz     : 机体系 z 轴角速度测量 [rad/s]
        dt     : 距上次预测的时间步长 [s]，必须为正有限值
        """
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt 必须是正数，得到 {dt}")
        for name, val in (("ax", ax), ("ay", ay), ("wz", wz)):
            if not math.isfinite(val):
                raise ValueError(f"{name} 必须是有限数")

        x, P = self.x, self.P
        q = float(x[YAW])
        Rb = _rotation_2d(q)

        a_meas = np.array([ax, ay], dtype=float)
        a_unbiased = a_meas - x[BAX:BAY + 1]
        a_world = Rb @ a_unbiased
        wz_unbiased = wz - x[BG]

        # ---- 状态传播 f(x, u) ----
        F = np.eye(STATE_DIM)
        # 位置
        x[PX:PY + 1] += x[VX:VY + 1] * dt + 0.5 * a_world * dt * dt
        F[PX:PY + 1, VX:VY + 1] = np.eye(2) * dt
        J = np.array([[0.0, -1.0], [1.0, 0.0]])
        d_a_world_d_yaw = J @ a_world
        F[PX:PY + 1, YAW] = 0.5 * dt * dt * d_a_world_d_yaw
        F[PX:PY + 1, BAX:BAY + 1] = -0.5 * dt * dt * Rb
        # 速度
        x[VX:VY + 1] += a_world * dt
        F[VX:VY + 1, YAW] = dt * d_a_world_d_yaw
        F[VX:VY + 1, BAX:BAY + 1] = -dt * Rb
        # 航向
        x[YAW] = wrap_pi(q + wz_unbiased * dt)
        F[YAW, BG] = -dt

        # ---- 过程噪声 Q = L N L^T ----
        n = self.noise
        # 离散步噪声协方差（白噪声按 1/dt 离散化，随机游走按 dt）
        N = np.diag([
            n.acc_psd / dt, n.acc_psd / dt,
            n.gyro_psd / dt,
            n.acc_bias_psd * dt, n.acc_bias_psd * dt,
            n.gyro_bias_psd * dt,
        ])
        L = np.zeros((STATE_DIM, 6))
        # 加速度白噪声列 (n_ax, n_ay)
        L[PX:PY + 1, 0:2] = 0.5 * dt * dt * Rb
        L[VX:VY + 1, 0:2] = dt * Rb
        L[YAW, 2] = dt                                  # 陀螺白噪声
        # 加速度零偏随机游走（同时传播到速度/位置，保留交叉相关）
        L[PX:PY + 1, 3:5] = 0.5 * dt * dt * Rb
        L[VX:VY + 1, 3:5] = dt * Rb
        L[BAX, 3] = 1.0
        L[BAY, 4] = 1.0
        # 陀螺零偏随机游走（当前步即影响航向积分）
        L[YAW, 5] = dt
        L[BG, 5] = 1.0
        Q = L @ N @ L.T

        P_new = F @ P @ F.T + Q
        # 对称化抵消浮点误差后校验半正定
        P_new = 0.5 * (P_new + P_new.T)
        self.P = validate_covariance(
            P_new, "predicted P", STATE_DIM, positive_semidefinite=True
        )

    # ------------------------------------------------------------------
    # 观测更新
    # ------------------------------------------------------------------
    def update(
        self,
        z: Sequence[float],
        H: np.ndarray,
        R: np.ndarray,
        angle_components: Iterable[int] = (),
        gate_probability: Optional[float] = 0.99,
        gate_sigma: Optional[float] = None,
    ) -> UpdateResult:
        """通用线性观测更新：``z = H x + v``，v ~ N(0, R)。

        - ``angle_components`` 中列出的 innovation 行会被规范化到 (-pi, pi]，
          用于航向角观测，跨越 ±pi 不会错误跳变。
        - 通过 Mahalanobis 距离做异常观测门限：``gate_probability`` 给出卡方
          分位数概率，或直接给 ``gate_sigma``（标准差倍数，平方为门限）。
          两者皆为 None 表示关闭门限。被拒绝的观测不改变状态/协方差。
        """
        Hm = np.asarray(H, dtype=float)
        Rm = validate_covariance(np.asarray(R, dtype=float), name="R")
        m = Rm.shape[0]
        if Hm.ndim != 2 or Hm.shape != (m, STATE_DIM):
            raise ValueError(
                f"H 形状应为 {(m, STATE_DIM)}，实际为 {Hm.shape}"
            )
        zv = np.asarray(z, dtype=float).reshape(-1)
        if zv.shape[0] != m:
            raise ValueError(f"观测 z 维度应为 {m}，实际为 {zv.shape[0]}")
        if not np.all(np.isfinite(zv)):
            raise ValueError("观测 z 包含 NaN 或 Inf")

        angle_set = set(int(i) for i in angle_components)
        if any(i < 0 or i >= m for i in angle_set):
            raise ValueError("angle_components 中存在越界的行索引")

        z_pred = Hm @ self.x
        innov = zv - z_pred
        for i in angle_set:
            innov[i] = wrap_pi(innov[i])

        S = Hm @ self.P @ Hm.T + Rm
        S = validate_covariance(S, "S", m, positive_semidefinite=False)

        # Mahalanobis 距离
        S_inv = np.linalg.inv(S)
        mahal_sq = float(innov @ S_inv @ innov)
        mahal = math.sqrt(max(mahal_sq, 0.0))

        if gate_sigma is not None:
            if gate_sigma <= 0:
                raise ValueError("gate_sigma 必须为正数")
            threshold_sq = gate_sigma * gate_sigma
        elif gate_probability is not None:
            if not 0.0 < gate_probability < 1.0:
                raise ValueError("gate_probability 必须位于 (0, 1)")
            threshold_sq = _chi2_ppf(gate_probability, m)
        else:
            threshold_sq = math.inf
        threshold = math.sqrt(threshold_sq)

        if mahal_sq > threshold_sq:
            return UpdateResult(
                accepted=False,
                innovation=innov,
                S=S,
                K=np.full((STATE_DIM, m), np.nan),
                mahalanobis=mahal,
                gate_threshold=threshold,
                predicted_measurement=z_pred,
                reason=(
                    f"Mahalanobis 距离 {mahal:.2f} 超过门限 {threshold:.2f}，"
                    "观测被拒绝"
                ),
            )

        K = self.P @ Hm.T @ S_inv
        self.x = self.x + K @ innov
        self.x[YAW] = wrap_pi(self.x[YAW])

        # Joseph 形式协方差更新，保证对称正定性
        IKH = np.eye(STATE_DIM) - K @ Hm
        P_new = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.P = validate_covariance(
            P_new, "updated P", STATE_DIM, positive_semidefinite=True
        )

        return UpdateResult(
            accepted=True,
            innovation=innov,
            S=S,
            K=K,
            mahalanobis=mahal,
            gate_threshold=threshold,
            predicted_measurement=z_pred,
            reason="accepted",
        )

    def update_position(
        self,
        z_xy: Sequence[float],
        R_xy: np.ndarray,
        gate_probability: Optional[float] = 0.99,
        gate_sigma: Optional[float] = None,
    ) -> UpdateResult:
        """二维位置观测更新，z = [px, py]。"""
        H = np.zeros((2, STATE_DIM))
        H[0, PX] = 1.0
        H[1, PY] = 1.0
        return self.update(
            z_xy, H, R_xy,
            gate_probability=gate_probability, gate_sigma=gate_sigma,
        )

    def update_velocity(
        self,
        z_vxvy: Sequence[float],
        R_v: np.ndarray,
        gate_probability: Optional[float] = 0.99,
        gate_sigma: Optional[float] = None,
    ) -> UpdateResult:
        """二维世界系速度观测更新，z = [vx, vy]。"""
        H = np.zeros((2, STATE_DIM))
        H[0, VX] = 1.0
        H[1, VY] = 1.0
        return self.update(
            z_vxvy, H, R_v,
            gate_probability=gate_probability, gate_sigma=gate_sigma,
        )

    def update_yaw(
        self,
        z_yaw: float,
        R_yaw: Union[float, np.ndarray],
        gate_probability: Optional[float] = 0.99,
        gate_sigma: Optional[float] = None,
    ) -> UpdateResult:
        """航向角观测更新 [rad]，innovation 自动规范化到 (-pi, pi]。"""
        H = np.zeros((1, STATE_DIM))
        H[0, YAW] = 1.0
        Rm = np.atleast_2d(np.asarray(R_yaw, dtype=float))
        return self.update(
            [z_yaw], H, Rm, angle_components=(0,),
            gate_probability=gate_probability, gate_sigma=gate_sigma,
        )

    # ------------------------------------------------------------------
    # 保存 / 恢复
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """导出完整滤波状态（状态向量、协方差、噪声配置）。"""
        return {
            "state_dim": STATE_DIM,
            "x": np.asarray(self.x, dtype=float).tolist(),
            "P": np.asarray(self.P, dtype=float).tolist(),
            "noise": {
                "acc_psd": self.noise.acc_psd,
                "gyro_psd": self.noise.gyro_psd,
                "acc_bias_psd": self.noise.acc_bias_psd,
                "gyro_bias_psd": self.noise.gyro_bias_psd,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EKF2D":
        """从 :meth:`to_dict` 产生的字典恢复滤波器。"""
        if int(data.get("state_dim", STATE_DIM)) != STATE_DIM:
            raise ValueError("状态文件的 state_dim 与当前实现不匹配")
        noise = NoiseConfig(**data["noise"])
        ekf = cls(
            initial_state=np.asarray(data["x"], dtype=float),
            initial_covariance=np.asarray(data["P"], dtype=float),
            noise_config=noise,
        )
        return ekf

    def save(self, path: Union[str, Path]) -> None:
        """把完整滤波状态以 JSON 保存到本地文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EKF2D":
        """从本地 JSON 文件恢复滤波器。"""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

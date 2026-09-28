r"""二维 IMU 扩展卡尔曼滤波器核心实现。

状态向量 x（8 维）::

    x = [ px, py, vx, vy, theta, ba_x, ba_y, b_g ]^T
          +----位置----+ +--速度--+  航向  加速度bias(机体系) 陀螺bias

所有计算均为内存中的 numpy 运算，状态可序列化到本地 JSON 文件，
不依赖 ROS / GPS / 硬件驱动 / 云平台。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .config import EKFConfig
from .math_utils import validate_covariance, wrap_angle

# 状态索引
PX, PY, VX, VY, THETA = 0, 1, 2, 3, 4
BAX, BAY, BG = 5, 6, 7
STATE_DIM = 8


class EKFError(ValueError):
    """滤波器参数或协方差非法时抛出（维度错误 / NaN / 非正定等）。"""


class EKFGateReject(EKFError):
    """观测被 Mahalanobis 门限拒绝时抛出（仅 strict 模式）。

    常规 update 不抛此异常，而是在 UpdateResult.accepted 中标记为 False。
    """

    def __init__(self, message: str, result: "UpdateResult") -> None:
        super().__init__(message)
        self.result = result


@dataclass
class UpdateResult:
    """一次观测更新的完整记录，便于调试与测试断言。"""

    kind: str                # "position" | "velocity" | "heading"
    z: np.ndarray            # 实际观测
    z_pred: np.ndarray       # 观测预测 h(x)
    innovation: np.ndarray   # 规范化后的新息 z - h(x)
    S: np.ndarray            # 新息协方差 H P H^T + R
    K: np.ndarray            # Kalman 增益
    maha: float              # Mahalanobis 距离 sqrt(y^T S^-1 y)
    accepted: bool           # 是否通过门限并真正执行了更新
    threshold: float         # 使用的门限

    def __str__(self) -> str:
        flag = "ACCEPT" if self.accepted else "REJECT"
        return (
            f"[{self.kind:8s}] {flag}  maha={self.maha:6.3f} "
            f"(thr={self.threshold:.3f})  innovation={np.round(self.innovation, 4)}"
        )


class IMU2DEKF:
    """二维平面 IMU EKF。

    参数
    ----
    config: EKFConfig，None 时使用默认噪声参数。
    initial_state: [px, py, vx, vy, theta, bax, bay, bg]，None 时为全零。
    initial_P: 8x8 初始协方差，None 时使用默认对角阵。
    """

    def __init__(
        self,
        config: EKFConfig | None = None,
        initial_state: np.ndarray | None = None,
        initial_P: np.ndarray | None = None,
    ) -> None:
        self.config = config if config is not None else EKFConfig()
        if not isinstance(self.config, EKFConfig):
            raise EKFError("config 必须是 EKFConfig 实例")

        if initial_state is None:
            self.x = np.zeros(STATE_DIM)
        else:
            self.x = np.asarray(initial_state, dtype=float).copy()
            if self.x.shape != (STATE_DIM,):
                raise EKFError(
                    f"初始状态维度错误：期望 ({STATE_DIM},)，实际 {self.x.shape}"
                )
            if not np.all(np.isfinite(self.x)):
                raise EKFError("初始状态含 NaN 或 Inf")

        self.x[THETA] = wrap_angle(self.x[THETA])

        if initial_P is None:
            # 默认：位置 1 m、速度 0.5 m/s、航向 0.5 rad、bias 量级适中
            self.P = np.diag([1.0, 1.0, 0.25, 0.25, 0.25,
                              0.05, 0.05, 0.01])
        else:
            self.P = validate_covariance(
                np.asarray(initial_P, dtype=float),
                dim=STATE_DIM,
                name="initial_P（初始协方差）",
            ).copy()

        self.t: float = 0.0

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[[PX, PY]].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[[VX, VY]].copy()

    @property
    def heading(self) -> float:
        return float(self.x[THETA])

    @property
    def accel_bias(self) -> np.ndarray:
        return self.x[[BAX, BAY]].copy()

    @property
    def gyro_bias(self) -> float:
        return float(self.x[BG])

    def reset_time(self, t: float = 0.0) -> None:
        self.t = float(t)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, dt: float, accel_body: np.ndarray, gyro: float) -> None:
        """IMU 预测步。

        参数
        ----
        dt : 时间步长（秒），必须为正有限值
        accel_body : 机体系比力/加速度测量 [ax, ay]（含 bias 与噪声），
                     约定为 body 坐标，R(theta) 将其旋转到世界系
        gyro : 绕 z 轴角速度测量（rad/s，含 bias 与噪声）
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise EKFError(f"dt 必须为正有限数，实际为 {dt}")
        try:
            a_m = np.asarray(accel_body, dtype=float).reshape(2)
        except (ValueError, TypeError) as exc:
            raise EKFError(
                f"accel_body 必须是长度为 2 的序列，实际为 {np.asarray(accel_body).shape}"
            ) from exc
        if not np.all(np.isfinite(a_m)):
            raise EKFError("accel_body 含 NaN 或 Inf")
        if not np.isfinite(gyro):
            raise EKFError("gyro 含 NaN 或 Inf")

        theta = float(self.x[THETA])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        # R 对 theta 的导数
        Rdot = np.array([[-s, -c], [c, -s]])

        v = self.x[[VX, VY]]
        ba = self.x[[BAX, BAY]]
        bg = self.x[BG]

        a_b = a_m - ba                 # 去 bias 的机体系加速度
        a_w = R @ a_b                  # 世界系加速度
        w = gyro - bg                  # 去 bias 的角速度

        # ---- 状态传播 ----
        x = self.x.copy()
        x[[PX, PY]] = self.x[[PX, PY]] + v * dt + 0.5 * a_w * dt * dt
        x[[VX, VY]] = v + a_w * dt
        x[THETA] = wrap_angle(theta + w * dt)
        # bias 状态保持不变（随机游走通过过程噪声体现）
        self.x = x

        # ---- 连续时间误差动力学雅可比 Fc = df/dx（Van Loan 需要连续矩阵）----
        Fc = np.zeros((STATE_DIM, STATE_DIM))
        # 位置行：dp/dt = v
        Fc[PX, VX] = 1.0
        Fc[PY, VY] = 1.0
        Fc[np.ix_([PX, PY], [THETA])] = (Rdot @ a_b).reshape(2, 1)
        Fc[np.ix_([PX, PY], [BAX, BAY])] = -R
        # 速度行：dv/dt = R(theta) (a_m - ba)
        Fc[np.ix_([VX, VY], [THETA])] = (Rdot @ a_b).reshape(2, 1)
        Fc[np.ix_([VX, VY], [BAX, BAY])] = -R
        # 航向行：dtheta/dt = gyro - bg
        Fc[THETA, BG] = -1.0
        # bias 行为 0（随机游走仅通过过程噪声驱动）

        # ---- 过程噪声：连续时间 PSD 经 Van Loan 精确离散化 ----
        # 噪声输入 u = [n_ax(机体系), n_ay, n_g, n_bax, n_bay, n_bg]
        # 连续时间状态方程中的噪声映射 L（8x6）
        cfg = self.config
        L = np.zeros((STATE_DIM, 6))
        L[np.ix_([VX, VY], [0, 1])] = R          # 加速度白噪声经姿态旋转
        L[THETA, 2] = 1.0                         # 陀螺白噪声
        L[BAX, 3] = 1.0                           # acc bias 随机游走
        L[BAY, 4] = 1.0
        L[BG, 5] = 1.0

        Qc = np.diag([
            cfg.acc_noise_psd, cfg.acc_noise_psd,
            cfg.gyro_noise_psd,
            cfg.acc_bias_psd, cfg.acc_bias_psd,
            cfg.gyro_bias_psd,
        ])

        Phi, Qd = self._van_loan(Fc, L @ Qc @ L.T, dt)

        # P = Phi P Phi^T + Qd
        P = Phi @ self.P @ Phi.T + Qd
        # 数值对称化并做正定校验
        P = 0.5 * (P + P.T)
        self.P = validate_covariance(P, dim=STATE_DIM, name="预测后协方差 P")

        self.t += float(dt)

    @staticmethod
    def _van_loan(
        Fc: np.ndarray, Qc_eff: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """连续噪声模型的 Van Loan 离散化。

        输入连续时间状态矩阵 Fc（内部执行 exp(Fc*dt)）与连续时间
        过程噪声 PSD 矩阵 Qc_eff，返回离散状态转移阵 Phi 与离散
        过程噪声协方差 Qd。
        """
        n = Fc.shape[0]
        M = np.zeros((2 * n, 2 * n))
        M[:n, :n] = -Fc
        M[:n, n:] = Qc_eff
        M[n:, n:] = Fc.T
        E = expm(M * dt)
        # Van Loan 分块：
        #   E22 = Phi^T,  E12 = Phi^-1 Qd
        Phi = E[n:, n:].T                  # (Phi^T)^T = Phi
        Qd = Phi @ E[:n, n:]               # Phi (Phi^-1 Qd) = Qd
        return Phi, 0.5 * (Qd + Qd.T)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    @staticmethod
    def _as_obs_vec(z, dim: int, kind: str) -> np.ndarray:
        """把观测转换成长度为 dim 的数组，维度非法时统一抛 EKFError。"""
        try:
            z_arr = np.asarray(z, dtype=float).reshape(dim)
        except (ValueError, TypeError) as exc:
            raise EKFError(
                f"{kind} 观测维度错误：期望长度 {dim}，实际 shape="
                f"{np.asarray(z).shape}"
            ) from exc
        if not np.all(np.isfinite(z_arr)):
            raise EKFError(f"{kind} 观测值含 NaN 或 Inf")
        return z_arr

    def update(
        self,
        kind: str,
        z,
        R: np.ndarray | None = None,
        *,
        strict: bool = False,
    ) -> UpdateResult:
        """通用 EKF 观测更新。

        参数
        ----
        kind : "position"（2D 位置）、"velocity"（2D 速度）或
               "heading"（标量航向，自动做角度规范化）
        z    : 观测值，位置/速度为长度 2 的序列，航向为标量（rad）
        R    : 本次观测的测量噪声协方差；None 时使用 config 默认值。
               必须为与观测维度匹配的对称（半）正定矩阵。
        strict : True 时观测被门限拒绝会抛出 EKFGateReject；
                 False（默认）时仅在结果中标记 accepted=False，状态不变。

        返回
        ----
        UpdateResult，包含 innovation、S、Kalman 增益、Mahalanobis 距离等。
        """
        if kind == "position":
            idx = [PX, PY]
            H = np.zeros((2, STATE_DIM)); H[:, idx] = np.eye(2)
            R_eff = self.config.R_pos if R is None else R
            z_arr = self._as_obs_vec(z, 2, kind)
            wrap = False
        elif kind == "velocity":
            idx = [VX, VY]
            H = np.zeros((2, STATE_DIM)); H[:, idx] = np.eye(2)
            R_eff = self.config.R_vel if R is None else R
            z_arr = self._as_obs_vec(z, 2, kind)
            wrap = False
        elif kind == "heading":
            idx = [THETA]
            H = np.zeros((1, STATE_DIM)); H[0, THETA] = 1.0
            if R is None:
                R_eff = np.array([[self.config.R_heading]])
            else:
                R_arr = np.asarray(R, dtype=float)
                if R_arr.ndim == 0:
                    R_arr = R_arr.reshape(1, 1)   # 允许标量 R
                R_eff = R_arr
            z_scalar = float(self._as_obs_vec(z, 1, kind)[0])
            z_arr = np.array([wrap_angle(z_scalar)])
            wrap = True
        else:
            raise EKFError(
                f"未知观测类型 {kind!r}，支持 'position'/'velocity'/'heading'"
            )

        m = H.shape[0]
        R_eff = validate_covariance(
            np.asarray(R_eff, dtype=float), dim=m, name=f"R（{kind} 测量噪声）"
        )

        z_pred = self.x[idx].copy()
        innovation = z_arr - z_pred
        if wrap:
            # 关键：航向新息规范化到 [-pi, pi)，跨越 ±pi 不产生 2*pi 跳变
            innovation = np.array([wrap_angle(float(innovation[0]))])

        S = H @ self.P @ H.T + R_eff
        # 新息协方差必须正定，否则无法求逆，属于非法 covariance
        S = validate_covariance(S, dim=m, name=f"S（{kind} 新息协方差）", eps=1e-10)
        # validate_covariance 允许半正定；求逆要求严格正定
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError as exc:
            raise EKFError(f"{kind} 新息协方差 S 不可逆：{exc}") from exc

        maha_sq = float(innovation @ S_inv @ innovation)
        maha = float(np.sqrt(max(maha_sq, 0.0)))

        K = self.P @ H.T @ S_inv
        result = UpdateResult(
            kind=kind,
            z=z_arr,
            z_pred=z_pred,
            innovation=innovation,
            S=S,
            K=K,
            maha=maha,
            accepted=True,
            threshold=self.config.gate_threshold,
        )

        # ---- 异常观测 gating ----
        if self.config.enable_gating and maha > self.config.gate_threshold:
            result.accepted = False
            if strict:
                raise EKFGateReject(
                    f"{kind} 观测被门限拒绝：maha={maha:.3f} > "
                    f"threshold={self.config.gate_threshold:.3f}",
                    result,
                )
            return result

        # ---- 状态与协方差更新（Joseph 形式，保证数值稳定与对称正定）----
        I_KH = np.eye(STATE_DIM) - K @ H
        self.x = self.x + K @ innovation
        self.x[THETA] = wrap_angle(float(self.x[THETA]))
        self.P = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P = validate_covariance(
            self.P, dim=STATE_DIM, name="更新后协方差 P"
        )
        # Joseph 更新后 K/S 保持本次计算值；同步 result.K 即同一数组
        return result

    # ------------------------------------------------------------------
    # 状态保存 / 恢复（本地 JSON 文件）
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """导出完整滤波状态（含配置），可 JSON 序列化。"""
        cfg = self.config
        return {
            "version": 1,
            "t": self.t,
            "x": self.x.tolist(),
            "P": self.P.tolist(),
            "config": {
                "acc_noise_psd": cfg.acc_noise_psd,
                "gyro_noise_psd": cfg.gyro_noise_psd,
                "acc_bias_psd": cfg.acc_bias_psd,
                "gyro_bias_psd": cfg.gyro_bias_psd,
                "gravity": cfg.gravity,
                "R_pos": cfg.R_pos.tolist(),
                "R_vel": cfg.R_vel.tolist(),
                "R_heading": cfg.R_heading,
                "gate_threshold": cfg.gate_threshold,
                "enable_gating": cfg.enable_gating,
            },
        }

    def save(self, path: str) -> None:
        """保存到本地 JSON 文件（不涉及任何网络/外部服务）。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "IMU2DEKF":
        """从 to_dict 产生的字典恢复滤波器。"""
        if not isinstance(data, dict) or data.get("version") != 1:
            raise EKFError("非法的滤波状态快照：缺少 version=1")
        cfg_data = data["config"]
        config = EKFConfig(
            acc_noise_psd=cfg_data["acc_noise_psd"],
            gyro_noise_psd=cfg_data["gyro_noise_psd"],
            acc_bias_psd=cfg_data["acc_bias_psd"],
            gyro_bias_psd=cfg_data["gyro_bias_psd"],
            gravity=cfg_data["gravity"],
            R_pos=np.asarray(cfg_data["R_pos"], dtype=float),
            R_vel=np.asarray(cfg_data["R_vel"], dtype=float),
            gate_threshold=cfg_data["gate_threshold"],
            enable_gating=cfg_data["enable_gating"],
        )
        config.R_heading = float(cfg_data["R_heading"])

        ekf = cls(
            config=config,
            initial_state=np.asarray(data["x"], dtype=float),
            initial_P=np.asarray(data["P"], dtype=float),
        )
        ekf.t = float(data["t"])
        return ekf

    @classmethod
    def load(cls, path: str) -> "IMU2DEKF":
        """从本地 JSON 文件恢复滤波器。"""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

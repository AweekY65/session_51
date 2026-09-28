"""EKF 配置：过程噪声与测量噪声均可配置，并带合法性检查."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import validate_covariance


@dataclass
class EKFConfig:
    """滤波器噪声与门限配置。

    过程噪声采用连续时间 IMU 噪声模型（功率谱密度 PSD），
    在每次 prediction 时按时间步长精确离散化：

    - acc_noise_psd : 加速度计白噪声 PSD，单位 m^2/s^3（即 (m/s^2)/sqrt(Hz) 的平方）
    - gyro_noise_psd: 陀螺仪白噪声 PSD，单位 rad^2/s
    - acc_bias_psd  : 加速度计 bias 随机游走 PSD，单位 m^2/s^5
    - gyro_bias_psd : 陀螺 bias 随机游走 PSD，单位 rad^2/s^3
    - gravity       : 重力加速度模长，单位 m/s^2（二维模型中不直接使用，保留接口）

    测量噪声：
    - R_pos: 位置观测噪声协方差 2x2，单位 m^2
    - R_vel: 速度观测噪声协方差 2x2，单位 m^2/s^2
    - R_heading: 航向观测噪声方差（标量），单位 rad^2

    门限：
    - gate_threshold: Mahalanobis 距离（chi 值）门限。若希望按 chi^2 概率门限
      配置，可用 chi2.ppf 换算后取平方根，例如 2 维、95% 对应
      sqrt(5.991) ≈ 2.448。默认 3.0（约对应 2 维 98.6% / 1 维 99.7%）。
    - enable_gating: False 时不拒绝任何观测（但仍记录距离）。
    """

    acc_noise_psd: float = 0.05
    gyro_noise_psd: float = 0.01
    acc_bias_psd: float = 0.002
    gyro_bias_psd: float = 0.0002
    gravity: float = 9.80665

    R_pos: np.ndarray | None = None
    R_vel: np.ndarray | None = None
    R_heading: float = 0.04  # 0.2 rad, 1 sigma

    gate_threshold: float = 3.0
    enable_gating: bool = True

    def __post_init__(self) -> None:
        if self.R_pos is None:
            self.R_pos = np.diag([0.25, 0.25])  # 0.5 m, 1 sigma
        if self.R_vel is None:
            self.R_vel = np.diag([0.04, 0.04])  # 0.2 m/s, 1 sigma

        self.R_pos = np.asarray(self.R_pos, dtype=float)
        self.R_vel = np.asarray(self.R_vel, dtype=float)

        self._validate_scalars()
        validate_covariance(self.R_pos, dim=2, name="R_pos（位置测量噪声）")
        validate_covariance(self.R_vel, dim=2, name="R_vel（速度测量噪声）")

    def _validate_scalars(self) -> None:
        """延迟导入 EKFError 以避免循环依赖。"""
        from .ekf import EKFError

        for name, val in [
            ("acc_noise_psd", self.acc_noise_psd),
            ("gyro_noise_psd", self.gyro_noise_psd),
            ("acc_bias_psd", self.acc_bias_psd),
            ("gyro_bias_psd", self.gyro_bias_psd),
        ]:
            if not np.isfinite(val) or val < 0.0:
                raise EKFError(f"{name} 必须为非负有限数，实际为 {val}")
        if not np.isfinite(self.gate_threshold) or self.gate_threshold <= 0.0:
            raise EKFError(
                f"gate_threshold 必须为正数，实际为 {self.gate_threshold}"
            )
        if not np.isfinite(self.R_heading) or self.R_heading <= 0.0:
            raise EKFError(
                f"R_heading 必须为正数，实际为 {self.R_heading}"
            )

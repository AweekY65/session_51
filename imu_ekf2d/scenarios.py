"""自动化测试使用的标准场景：代码生成轨迹、带噪 IMU 与观测。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .filter import NoiseConfig, STATE_DIM
from .simulator import SimConfig, SimResult, TrajSegment, generate_trajectory


@dataclass
class Scenario:
    name: str
    sim: SimResult
    initial_state: np.ndarray
    initial_covariance: Optional[np.ndarray]
    noise_config: NoiseConfig
    gate_probability: Optional[float]
    steady_after: float


def _initial_covariance(
    pos: float = 1.0,
    vel: float = 1.0,
    yaw: float = 0.25,
    abias: float = 0.25,
    gbias: float = 0.04,
) -> np.ndarray:
    return np.diag([pos, pos, vel, vel, yaw, abias, abias, gbias])


def make_static() -> Scenario:
    """场景 1：全程静止，存在小幅 IMU 白噪声。"""
    cfg = SimConfig(
        seed=101,
        acc_noise_std=0.05,
        gyro_noise_std=0.005,
        use_velocity_obs=True,
        use_yaw_obs=True,
        yaw_noise_std=0.02,
    )
    sim = generate_trajectory([TrajSegment(duration=20.0)], cfg)
    x0 = np.zeros(STATE_DIM)
    return Scenario(
        "静止 (static)",
        sim, x0,
        _initial_covariance(pos=0.5, vel=0.5, yaw=0.1, abias=0.1,
                            gbias=0.01),
        NoiseConfig(acc_psd=0.02, gyro_psd=2.5e-5,
                    acc_bias_psd=1e-5, gyro_bias_psd=1e-7),
        gate_probability=0.99,
        steady_after=2.0,
    )


def make_constant_velocity() -> Scenario:
    """场景 2：匀速直线运动 (2 m/s 沿世界系 x 轴)。"""
    cfg = SimConfig(
        seed=202,
        acc_noise_std=0.05,
        gyro_noise_std=0.005,
        use_yaw_obs=True,
        yaw_noise_std=0.02,
    )
    sim = generate_trajectory(
        [TrajSegment(duration=20.0)],
        cfg,
        initial_state=np.array([0, 0, 2.0, 0.0, 0.0, 0, 0, 0]),
    )
    # 初始位置/航向有偏差，初速度未知，滤波应收敛到真值
    x0 = np.array([0.5, -0.5, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    return Scenario(
        "匀速直线 (constant velocity)",
        sim, x0,
        _initial_covariance(),
        NoiseConfig(acc_psd=0.05, gyro_psd=2.5e-5),
        gate_probability=0.99,
        steady_after=3.0,
    )


def make_turn() -> Scenario:
    """场景 3：机体系恒定向心加速度 + 恒定角速度，走圆弧。"""
    radius = 5.0
    speed = 2.0
    wz = speed / radius           # 0.4 rad/s
    lat_acc = speed * wz          # 0.8 m/s^2，机体系 y 向
    cfg = SimConfig(
        seed=303,
        acc_noise_std=0.05,
        gyro_noise_std=0.005,
        use_velocity_obs=True,
    )
    sim = generate_trajectory(
        [TrajSegment(duration=25.0, ax=0.0, ay=lat_acc, wz=wz,
                     frame="body")],
        cfg,
        initial_state=np.array([0, 0, speed, 0.0, 0.0, 0, 0, 0]),
    )
    x0 = np.array([0.3, -0.3, 1.5, 0.3, 0.05, 0.0, 0.0, 0.0])
    return Scenario(
        "匀速转弯 (circular turn)",
        sim, x0,
        _initial_covariance(),
        NoiseConfig(acc_psd=0.05, gyro_psd=2.5e-5,
                    acc_bias_psd=1e-4, gyro_bias_psd=1e-6),
        gate_probability=0.99,
        steady_after=5.0,
    )


def make_bias() -> Scenario:
    """场景 4：含真实加速度/陀螺零偏，分段做直线 + 转弯以激励 bias 可观。"""
    cfg = SimConfig(
        seed=404,
        acc_noise_std=0.05,
        gyro_noise_std=0.005,
        use_velocity_obs=True,
        use_yaw_obs=True,
        bax=0.15,
        bay=-0.10,
        bg=0.03,
        yaw_noise_std=0.015,
    )
    segments = [
        TrajSegment(duration=12.0, ax=0.0, ay=0.0, wz=0.0),
        TrajSegment(duration=13.0, ax=0.0, ay=0.8, wz=0.4, frame="body"),
        TrajSegment(duration=15.0, ax=0.0, ay=0.0, wz=0.0),
    ]
    sim = generate_trajectory(
        segments,
        cfg,
        initial_state=np.array([0, 0, 2.0, 0.0, 0.0, 0, 0, 0]),
    )
    x0 = np.zeros(STATE_DIM)
    x0[2] = 1.5  # 初始速度也略有偏差；bias 初值为 0（未知）
    return Scenario(
        "IMU 零偏估计 (bias estimation)",
        sim, x0,
        # bias 初始不确定度给大一些，表示未知零偏
        _initial_covariance(abias=0.25**2 * 10, gbias=0.1),
        NoiseConfig(acc_psd=0.05, gyro_psd=2.5e-5,
                    acc_bias_psd=2e-5, gyro_bias_psd=1e-7),
        gate_probability=0.99,
        steady_after=20.0,
    )


def make_yaw_wrap() -> Scenario:
    """场景 5：累计旋转超过 ±pi，反复跨界，验证航向规范化。"""
    cfg = SimConfig(
        seed=505,
        acc_noise_std=0.03,
        gyro_noise_std=0.004,
        use_velocity_obs=False,
        use_yaw_obs=True,
        yaw_noise_std=0.02,
        pos_noise_std=0.15,
    )
    segments = [
        TrajSegment(duration=12.0, ax=0.0, ay=0.0, wz=0.8, frame="body"),
        TrajSegment(duration=12.0, ax=0.0, ay=0.0, wz=-0.8, frame="body"),
    ]
    # 从接近 +pi 的初始航向开始，立即跨界
    yaw0 = np.pi - 0.2
    sim = generate_trajectory(
        segments,
        cfg,
        initial_state=np.array([0, 0, 0.0, 0.0, yaw0, 0, 0, 0]),
    )
    x0 = np.zeros(STATE_DIM)
    x0[4] = yaw0
    return Scenario(
        "航向角跨界 (yaw wrap)",
        sim, x0,
        _initial_covariance(pos=0.2, vel=0.2, yaw=0.01, abias=0.05,
                            gbias=0.01),
        NoiseConfig(acc_psd=0.02, gyro_psd=1.6e-5,
                    acc_bias_psd=1e-5, gyro_bias_psd=1e-7),
        gate_probability=0.99,
        steady_after=2.0,
    )


def make_outlier() -> Scenario:
    """场景 6：注入大偏移位置/速度观测，gating 必须拒绝。"""
    cfg = SimConfig(
        seed=606,
        acc_noise_std=0.05,
        gyro_noise_std=0.005,
        use_yaw_obs=True,
        yaw_noise_std=0.02,
    )
    outlier_times = [4.0, 8.0, 12.0, 16.0]
    outlier_kinds = ["position", "velocity", "position", "velocity"]
    sim = generate_trajectory(
        [TrajSegment(duration=20.0)],
        cfg,
        initial_state=np.array([0, 0, 1.0, 0.5, 0.0, 0, 0, 0]),
        outlier_times=outlier_times,
        outlier_kinds=outlier_kinds,
        outlier_magnitude=10.0,
    )
    x0 = np.zeros(STATE_DIM)
    x0[2], x0[3] = 1.0, 0.5
    return Scenario(
        "异常观测门限 (Mahalanobis gating)",
        sim, x0,
        _initial_covariance(pos=0.25, vel=0.25),
        NoiseConfig(acc_psd=0.05, gyro_psd=2.5e-5),
        gate_probability=0.99,
        steady_after=2.0,
    )


def make_save_restore() -> Scenario:
    """场景 7：转弯 + 零偏轨迹，用于保存/恢复一致性（复用 bias 风格数据）。"""
    cfg = SimConfig(
        seed=707,
        acc_noise_std=0.05,
        gyro_noise_std=0.005,
        use_velocity_obs=True,
        use_yaw_obs=True,
        yaw_noise_std=0.02,
        bax=0.10,
        bay=0.05,
        bg=0.02,
    )
    segments = [
        TrajSegment(duration=10.0, ax=0.0, ay=0.0, wz=0.0),
        TrajSegment(duration=12.0, ax=0.0, ay=0.6, wz=0.3, frame="body"),
    ]
    sim = generate_trajectory(
        segments,
        cfg,
        initial_state=np.array([0, 0, 1.5, 0.0, 0.0, 0, 0, 0]),
    )
    x0 = np.zeros(STATE_DIM)
    x0[2] = 1.5
    return Scenario(
        "状态保存与恢复 (save / restore)",
        sim, x0,
        _initial_covariance(abias=0.2, gbias=0.05),
        NoiseConfig(acc_psd=0.05, gyro_psd=2.5e-5,
                    acc_bias_psd=2e-5, gyro_bias_psd=1e-7),
        gate_probability=0.99,
        steady_after=8.0,
    )


ALL_SCENARIOS = [
    make_static,
    make_constant_velocity,
    make_turn,
    make_bias,
    make_yaw_wrap,
    make_outlier,
    make_save_restore,
]

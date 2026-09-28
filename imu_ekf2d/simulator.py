"""代码生成的真实轨迹 + 带噪 IMU 样本与观测。

全部数据在内存中生成，不接入任何真实硬件、GPS 或网络服务。

轨迹由若干 :class:`TrajSegment` 拼接，每个分段在机体系（``frame='body'``）
或世界系（``frame='world'``）下给定常值加速度与常值角速度，用小步长
RK/中点积分得到地面真值，再按采样率生成：

* IMU 样本：机体系 ``[ax, ay, wz]`` = 真值 + 常值零偏 + 高斯白噪声；
* 观测：位置 / 速度 / 航向角，各自独立的采样率与高斯噪声，可注入离群值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np

from .filter import BAX, BAY, BG, PX, PY, VX, VY, YAW, STATE_DIM, wrap_pi


FrameName = Literal["body", "world"]


@dataclass
class TrajSegment:
    """一段常值加速度 + 常值角速度运动。

    ax, ay : 分段坐标系下的加速度 [m/s^2]
    wz     : 角速度 [rad/s]（平面运动时机体系/世界系相同）
    frame  : 'body' 表示加速度在机体系，'world' 表示在世界系
    """

    duration: float
    ax: float = 0.0
    ay: float = 0.0
    wz: float = 0.0
    frame: FrameName = "world"


@dataclass
class Observation:
    """一次标量/向量观测。"""

    t: float
    kind: str           # 'position' | 'velocity' | 'yaw'
    z: np.ndarray
    R: np.ndarray
    is_outlier: bool = False


@dataclass
class SimConfig:
    """仿真采样与噪声配置。"""

    imu_rate: float = 100.0          # IMU 采样率 [Hz]
    obs_rate: float = 10.0           # 位置观测采样率 [Hz]
    acc_noise_std: float = 0.05      # 每个 IMU 样本加速度白噪声标准差 [m/s^2]
    gyro_noise_std: float = 0.01     # 每个 IMU 样本角速度白噪声标准差 [rad/s]
    pos_noise_std: float = 0.2       # 位置观测噪声标准差 [m]
    vel_noise_std: float = 0.1       # 速度观测噪声标准差 [m/s]
    yaw_noise_std: float = 0.02      # 航向观测噪声标准差 [rad]
    use_velocity_obs: bool = True
    use_yaw_obs: bool = False
    bax: float = 0.0                 # 真实加速度零偏 [m/s^2]
    bay: float = 0.0
    bg: float = 0.0                  # 真实陀螺零偏 [rad/s]
    seed: int = 0


@dataclass
class SimResult:
    """仿真产物。"""

    t_gt: np.ndarray                  # 真值时间戳
    x_gt: np.ndarray                  # 真值状态 (M, 8)，bias 列为真值常值
    imu: List[tuple]                  # [(t, ax_m, ay_m, wz_m), ...]
    observations: List[Observation]
    config: SimConfig
    segments: List[TrajSegment] = field(default_factory=list)


def generate_trajectory(
    segments: List[TrajSegment],
    config: Optional[SimConfig] = None,
    initial_state: Optional[np.ndarray] = None,
    outlier_times: Optional[List[float]] = None,
    outlier_kinds: Optional[List[str]] = None,
    outlier_magnitude: float = 10.0,
) -> SimResult:
    """根据分段描述生成真值轨迹、带噪 IMU 样本与观测。

    参数
    ----
    segments : 轨迹分段列表
    config   : 采样/噪声配置
    initial_state : 8 维初始真值 [px,py,vx,vy,yaw,bax,bay,bg]，
                    其中 bias 项会被 ``config`` 中的真值 bias 覆盖
    outlier_times : 在这些（近似）时刻注入离群观测
    outlier_kinds : 与时间一一对应，指定离群观测类型（默认 'position'）
    outlier_magnitude : 离群偏移幅度（位置 m / 速度 m/s / 航向 rad）
    """
    cfg = config if config is not None else SimConfig()
    rng = np.random.default_rng(cfg.seed)
    outlier_times = outlier_times or []
    outlier_kinds = outlier_kinds or ["position"] * len(outlier_times)
    if len(outlier_kinds) != len(outlier_times):
        raise ValueError("outlier_times 与 outlier_kinds 长度必须一致")

    if initial_state is None:
        x0 = np.zeros(STATE_DIM)
    else:
        x0 = np.asarray(initial_state, dtype=float).copy()
    x0[BAX], x0[BAY], x0[BG] = cfg.bax, cfg.bay, cfg.bg
    x0[YAW] = wrap_pi(x0[YAW])

    total_duration = sum(seg.duration for seg in segments)
    dt_imu = 1.0 / cfg.imu_rate
    # 积分子步长：远小于 IMU 步长，保证真值轨迹平滑
    h = dt_imu / 10.0

    # ---- 用小步长积分真值 ----
    gt_times: List[float] = []
    gt_states: List[np.ndarray] = []
    x = x0.copy()
    n_sub = int(round(total_duration / h))
    gt_times.append(0.0)
    gt_states.append(x.copy())

    seg_ends = np.cumsum([seg.duration for seg in segments])

    def seg_at(time: float) -> TrajSegment:
        idx = int(np.searchsorted(seg_ends, time, side="right"))
        return segments[min(idx, len(segments) - 1)]

    for k in range(1, n_sub + 1):
        time = min(k * h, total_duration)
        seg = seg_at(time)
        a = np.array([seg.ax, seg.ay])
        if seg.frame == "body":
            # 中点方向，二阶精度
            q_mid = x[YAW] + 0.5 * seg.wz * h
            a_world = _R(q_mid) @ a
        else:
            a_world = a
        x[PX:PY + 1] += x[VX:VY + 1] * h + 0.5 * a_world * h * h
        x[VX:VY + 1] += a_world * h
        x[YAW] = wrap_pi(x[YAW] + seg.wz * h)
        gt_times.append(time)
        gt_states.append(x.copy())

    gt_times = np.asarray(gt_times)
    gt_states = np.asarray(gt_states)

    # ---- 在 IMU 时刻采样真值，生成带噪 IMU 样本 ----
    imu: List[tuple] = []
    n_imu = int(round(total_duration * cfg.imu_rate))
    # 用整数计数生成时间戳，保证 IMU/观测落在同一个二进制有理数网格上
    imu_times = np.arange(n_imu + 1) / cfg.imu_rate
    idx = np.clip(
        np.round(imu_times / h).astype(int), 0, len(gt_times) - 1
    )
    for s, imu_t in enumerate(imu_times):
        # 真值机体系加速度/角速度：从分段描述解析得到（分段常值），
        # 与真值积分使用同样的分段归属规则
        seg = seg_at(min(imu_t, total_duration - 1e-12))
        q = float(gt_states[idx[s]][YAW])
        a_body = (
            np.array([seg.ax, seg.ay])
            if seg.frame == "body"
            else _R(q).T @ np.array([seg.ax, seg.ay])
        )
        ax_true, ay_true, wz_true = (
            float(a_body[0]), float(a_body[1]), seg.wz
        )

        ax_m = ax_true + cfg.bax + rng.normal(0.0, cfg.acc_noise_std)
        ay_m = ay_true + cfg.bay + rng.normal(0.0, cfg.acc_noise_std)
        wz_m = wz_true + cfg.bg + rng.normal(0.0, cfg.gyro_noise_std)
        imu.append((float(imu_t), float(ax_m), float(ay_m), float(wz_m)))

    # ---- 生成观测 ----
    observations: List[Observation] = []
    dt_obs = 1.0 / cfg.obs_rate
    n_obs = int(round(total_duration * cfg.obs_rate))
    obs_times = np.arange(1, n_obs + 1) / cfg.obs_rate
    obs_idx = np.clip(
        np.round(obs_times / h).astype(int), 0, len(gt_states) - 1
    )
    R_pos = np.diag([cfg.pos_noise_std**2] * 2)
    R_vel = np.diag([cfg.vel_noise_std**2] * 2)
    R_yaw = np.array([[cfg.yaw_noise_std**2]])

    for o, obs_t in enumerate(obs_times):
        st = gt_states[obs_idx[o]]
        tt = float(obs_t)
        # 位置观测
        z_pos = st[PX:PY + 1] + rng.normal(0.0, cfg.pos_noise_std, size=2)
        is_out = _match_outlier_time(tt, outlier_times, dt_obs, "position",
                                     outlier_kinds)
        if is_out:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            z_pos = z_pos + outlier_magnitude * np.array(
                [np.cos(angle), np.sin(angle)]
            )
        observations.append(Observation(tt, "position", z_pos, R_pos, is_out))

        if cfg.use_velocity_obs:
            z_vel = st[VX:VY + 1] + rng.normal(0.0, cfg.vel_noise_std, size=2)
            is_out = _match_outlier_time(tt, outlier_times, dt_obs, "velocity",
                                         outlier_kinds)
            if is_out:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                z_vel = z_vel + outlier_magnitude * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
            observations.append(Observation(tt, "velocity", z_vel, R_vel,
                                            is_out))

        if cfg.use_yaw_obs:
            z_yaw = wrap_pi(
                st[YAW] + rng.normal(0.0, cfg.yaw_noise_std)
            )
            is_out = _match_outlier_time(tt, outlier_times, dt_obs, "yaw",
                                         outlier_kinds)
            if is_out:
                z_yaw = wrap_pi(z_yaw + outlier_magnitude)
            observations.append(
                Observation(tt, "yaw", np.atleast_1d(z_yaw), R_yaw, is_out)
            )

    observations.sort(key=lambda ob: ob.t)
    return SimResult(
        t_gt=gt_times,
        x_gt=gt_states,
        imu=imu,
        observations=observations,
        config=cfg,
        segments=list(segments),
    )


def _match_outlier_time(
    t: float,
    outlier_times: List[float],
    dt_obs: float,
    kind: str,
    outlier_kinds: List[str],
) -> bool:
    for ot, ok in zip(outlier_times, outlier_kinds):
        if ok == kind and abs(t - ot) < 0.5 * dt_obs + 1e-9:
            return True
    return False


def _R(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])

"""把 :class:`SimResult` 喂给 EKF 并统计估计误差。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .filter import (
    BAX,
    BAY,
    BG,
    EKF2D,
    NoiseConfig,
    STATE_DIM,
    VX,
    VY,
    YAW,
    PX,
    PY,
    wrap_pi,
)
from .simulator import Observation, SimResult


@dataclass
class UpdateRecord:
    """一次观测更新的记录（含真值 outlier 标记，供测试核对）。"""

    t: float
    kind: str
    accepted: bool
    is_outlier: bool
    mahalanobis: float
    gate_threshold: float
    innovation: np.ndarray


@dataclass
class RunResult:
    times: np.ndarray
    estimates: np.ndarray       # (N, 8)
    covariances: np.ndarray     # (N, 8, 8)
    records: List[UpdateRecord] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return sum(1 for r in self.records if r.accepted)

    @property
    def rejected_count(self) -> int:
        return sum(1 for r in self.records if not r.accepted)

    @property
    def outliers_total(self) -> int:
        return sum(1 for r in self.records if r.is_outlier)

    @property
    def outliers_rejected(self) -> int:
        return sum(
            1 for r in self.records if r.is_outlier and not r.accepted
        )

    @property
    def inliers_rejected(self) -> int:
        return sum(
            1 for r in self.records if not r.is_outlier and not r.accepted
        )


def apply_observation(
    ekf: EKF2D, obs: Observation, gate_probability: Optional[float]
) -> UpdateRecord:
    """根据观测类型调用对应的 update。"""
    if obs.kind == "position":
        res = ekf.update_position(
            obs.z, obs.R, gate_probability=gate_probability
        )
    elif obs.kind == "velocity":
        res = ekf.update_velocity(
            obs.z, obs.R, gate_probability=gate_probability
        )
    elif obs.kind == "yaw":
        res = ekf.update_yaw(
            float(obs.z[0]), obs.R, gate_probability=gate_probability
        )
    else:
        raise ValueError(f"未知观测类型: {obs.kind}")
    return UpdateRecord(
        t=obs.t,
        kind=obs.kind,
        accepted=res.accepted,
        is_outlier=obs.is_outlier,
        mahalanobis=res.mahalanobis,
        gate_threshold=res.gate_threshold,
        innovation=res.innovation,
    )


def advance(
    ekf: EKF2D,
    sim: SimResult,
    gate_probability: Optional[float] = 0.99,
    imu_start: int = 0,
    obs_start: int = 0,
) -> RunResult:
    """从给定的 IMU/观测索引开始处理（用于保存/恢复后继续滤波）。

    ``imu_start``/``obs_start`` 是样本列表中的下标。dt 仍由时间戳差值
    确定，因此调用方恢复滤波器后无需保存额外的“上一时刻”信息。
    """
    times: List[float] = []
    estimates: List[np.ndarray] = []
    covariances: List[np.ndarray] = []
    records: List[UpdateRecord] = []

    obs_list = list(sim.observations)
    obs_ptr = max(0, obs_start)
    imu_items = sim.imu[max(0, imu_start):]
    if not imu_items:
        return RunResult(
            times=np.asarray(times),
            estimates=np.asarray(estimates),
            covariances=np.asarray(covariances),
            records=records,
        )
    # 第一段 dt 相对于前一条 IMU 时间戳（或 0）
    first_t = imu_items[0][0]
    if imu_start >= 1:
        prev_t = sim.imu[imu_start - 1][0]
    else:
        prev_t = 0.0
    assert first_t >= prev_t

    for imu_t, ax_m, ay_m, wz_m in imu_items:
        dt = imu_t - prev_t
        if dt > 0.0:
            ekf.predict(ax_m, ay_m, wz_m, dt)
        while obs_ptr < len(obs_list) and obs_list[obs_ptr].t <= imu_t + 1e-9:
            records.append(
                apply_observation(ekf, obs_list[obs_ptr], gate_probability)
            )
            obs_ptr += 1
        times.append(imu_t)
        estimates.append(ekf.x.copy())
        covariances.append(ekf.P.copy())
        prev_t = imu_t

    return RunResult(
        times=np.asarray(times),
        estimates=np.asarray(estimates),
        covariances=np.asarray(covariances),
        records=records,
    )


def run_filter(
    sim: SimResult,
    ekf: Optional[EKF2D] = None,
    gate_probability: Optional[float] = 0.99,
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
    noise_config: Optional[NoiseConfig] = None,
) -> RunResult:
    """按时间顺序处理全部 IMU 预测与观测更新。

    若不提供 ``ekf``，则用其它初始参数新建滤波器。
    """
    if ekf is None:
        ekf = EKF2D(
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            noise_config=noise_config,
        )

    # 记录初始时刻（t=0，还没做预测）
    times: List[float] = [0.0]
    estimates: List[np.ndarray] = [ekf.x.copy()]
    covariances: List[np.ndarray] = [ekf.P.copy()]

    rest = advance(ekf, sim, gate_probability=gate_probability, imu_start=0)
    times.extend(rest.times.tolist())
    estimates.extend(list(rest.estimates))
    covariances.extend(list(rest.covariances))

    return RunResult(
        times=np.asarray(times),
        estimates=np.asarray(estimates),
        covariances=np.asarray(covariances),
        records=rest.records,
    )


@dataclass
class Metrics:
    rmse_pos: float           # [m]
    rmse_vel: float           # [m/s]
    rmse_yaw: float           # [rad]，已角度规范化
    rmse_bax: float           # [m/s^2]
    rmse_bay: float           # [m/s^2]
    rmse_bg: float            # [rad/s]
    final_pos: float
    final_vel: float
    final_yaw: float
    max_pos: float
    max_vel: float
    n_samples: int


def _interp_gt(sim: SimResult, times: np.ndarray) -> np.ndarray:
    """把 8 维真值插值到滤波器记录的时刻（yaw 列单独处理）。"""
    out = np.zeros((len(times), STATE_DIM))
    for col in range(STATE_DIM):
        if col == YAW:
            # 对航向用复数表示插值，避免跨 ±pi 出错
            cx = np.interp(times, sim.t_gt, np.cos(sim.x_gt[:, col]))
            sx = np.interp(times, sim.t_gt, np.sin(sim.x_gt[:, col]))
            out[:, col] = np.arctan2(sx, cx)
        else:
            out[:, col] = np.interp(times, sim.t_gt, sim.x_gt[:, col])
    return out


def compute_metrics(
    sim: SimResult, run: RunResult, steady_after: float = 0.0
) -> Metrics:
    """计算 RMSE / 末端误差 / 最大误差。

    ``steady_after`` 给定滤波器收敛时间（秒），只统计该时刻之后的样本，
    便于区分初始误差与稳态精度。
    """
    gt = _interp_gt(sim, run.times)
    mask = run.times >= steady_after

    err_pos = np.linalg.norm(
        run.estimates[mask, PX:PY + 1] - gt[mask, PX:PY + 1], axis=1
    )
    err_vel = np.linalg.norm(
        run.estimates[mask, VX:VY + 1] - gt[mask, VX:VY + 1], axis=1
    )
    err_yaw = np.abs(
        wrap_pi(run.estimates[mask, YAW] - gt[mask, YAW])
    )
    err_bax = np.abs(run.estimates[mask, BAX] - gt[mask, BAX])
    err_bay = np.abs(run.estimates[mask, BAY] - gt[mask, BAY])
    err_bg = np.abs(run.estimates[mask, BG] - gt[mask, BG])

    return Metrics(
        rmse_pos=float(np.sqrt(np.mean(err_pos**2))),
        rmse_vel=float(np.sqrt(np.mean(err_vel**2))),
        rmse_yaw=float(np.sqrt(np.mean(err_yaw**2))),
        rmse_bax=float(np.sqrt(np.mean(err_bax**2))),
        rmse_bay=float(np.sqrt(np.mean(err_bay**2))),
        rmse_bg=float(np.sqrt(np.mean(err_bg**2))),
        final_pos=float(err_pos[-1]),
        final_vel=float(err_vel[-1]),
        final_yaw=float(err_yaw[-1]),
        max_pos=float(np.max(err_pos)),
        max_vel=float(np.max(err_vel)),
        n_samples=int(mask.sum()),
    )


def format_metrics(name: str, metrics: Metrics, run: RunResult) -> str:
    """生成人类可读的误差报告。"""
    lines = [
        f"[{name}]",
        f"  样本数            : {metrics.n_samples}",
        f"  位置 RMSE         : {metrics.rmse_pos*1000:8.2f} mm"
        f"   (末端 {metrics.final_pos*1000:7.2f} mm, "
        f"最大 {metrics.max_pos*1000:7.2f} mm)",
        f"  速度 RMSE         : {metrics.rmse_vel*1000:8.2f} mm/s"
        f" (末端 {metrics.final_vel*1000:7.2f} mm/s, "
        f"最大 {metrics.max_vel*1000:7.2f} mm/s)",
        f"  航向 RMSE         : {metrics.rmse_yaw*1000:8.2f} mrad"
        f" (末端 {metrics.final_yaw*1000:7.2f} mrad)",
        f"  acc bias RMSE     : x={metrics.rmse_bax*1000:7.2f} "
        f"y={metrics.rmse_bay*1000:7.2f} mm/s^2",
        f"  gyro bias RMSE    : {metrics.rmse_bg*1000:8.3f} mrad/s",
        f"  观测接受/拒绝     : {run.accepted_count}/{run.rejected_count}"
        f" (真值离群 {run.outliers_total} 个, 拒绝 {run.outliers_rejected} 个,"
        f" 正常观测误拒 {run.inliers_rejected} 个)",
    ]
    return "\n".join(lines)

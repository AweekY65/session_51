"""将仿真场景送入 EKF 并计算估计误差的辅助工具（本地评估，无外部依赖）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ekf import IMU2DEKF, UpdateResult
from .math_utils import wrap_angle
from .simulator import ScenarioData


@dataclass
class RunMetrics:
    t: np.ndarray                 # (N,)
    est: np.ndarray               # (N, 8) 估计状态
    truth: np.ndarray             # (N, 8) 真值状态
    update_results: list          # 每次观测更新的 UpdateResult
    rejected: int                 # 被门限拒绝的观测数
    injected_outliers: int = 0

    def _err(self, col: int) -> np.ndarray:
        return self.est[:, col] - self.truth[:, col]

    @property
    def pos_rmse(self) -> float:
        e = np.concatenate([self._err(0), self._err(1)])
        return float(np.sqrt(np.mean(e ** 2)))

    @property
    def vel_rmse(self) -> float:
        e = np.concatenate([self._err(2), self._err(3)])
        return float(np.sqrt(np.mean(e ** 2)))

    @property
    def heading_rmse(self) -> float:
        e = wrap_angle(self.est[:, 4] - self.truth[:, 4])
        return float(np.sqrt(np.mean(e ** 2)))

    @property
    def accel_bias_rmse(self) -> float:
        e = np.concatenate([self._err(5), self._err(6)])
        return float(np.sqrt(np.mean(e ** 2)))

    @property
    def gyro_bias_err(self) -> float:
        return float(np.sqrt(np.mean(self._err(7) ** 2)))

    @property
    def final_pos_err(self) -> float:
        return float(np.hypot(self._err(0)[-1], self._err(1)[-1]))

    def summary(self) -> dict:
        return {
            "pos_rmse_m": self.pos_rmse,
            "vel_rmse_mps": self.vel_rmse,
            "heading_rmse_rad": self.heading_rmse,
            "accel_bias_rmse": self.accel_bias_rmse,
            "gyro_bias_rmse": self.gyro_bias_err,
            "rejected_obs": self.rejected,
            "injected_outliers": self.injected_outliers,
        }


def run_scenario(
    data: ScenarioData,
    ekf: IMU2DEKF,
    *,
    record: bool = True,
    use_obs: bool = True,
) -> RunMetrics:
    """按时间顺序对场景执行 predict/update，记录估计与真值。

    观测在其时间戳不晚于当前 IMU 时刻时执行；位置/速度/航向观测
    的测量噪声可由 EKFConfig 配置。
    """
    obs_sorted = sorted(data.obs, key=lambda o: o.t)
    obs_i = 0
    n_obs = len(obs_sorted)
    results: list[UpdateResult] = []
    rejected = 0

    times = [0.0]
    ests = [ekf.x.copy()]
    truth0 = data.truth[0].array
    truths = [truth0]

    for sample in data.imu:
        ekf.predict(sample.dt, sample.accel_body, sample.gyro)
        t_now = sample.t

        while obs_i < n_obs and obs_sorted[obs_i].t <= t_now + 1e-9:
            o = obs_sorted[obs_i]
            if use_obs:
                r = ekf.update(o.kind, o.z)
                results.append(r)
                if not r.accepted:
                    rejected += 1
            obs_i += 1

        if record:
            times.append(t_now)
            ests.append(ekf.x.copy())
            truths.append(_truth_at(data, t_now))

    return RunMetrics(
        t=np.asarray(times),
        est=np.asarray(ests),
        truth=np.asarray(truths),
        update_results=results,
        rejected=rejected,
        injected_outliers=sum(1 for o in data.obs if o.outlier),
    )


def _truth_at(data: ScenarioData, t: float) -> np.ndarray:
    """取不晚于 t 的最近真值采样（IMU 时刻与 truth 时刻对齐，直接索引）。"""
    ts = np.asarray([s.t for s in data.truth])
    i = int(np.searchsorted(ts, t, side="right")) - 1
    i = max(0, min(i, len(data.truth) - 1))
    return data.truth[i].array

"""pytest 公共夹具与辅助：构造仿真器、按真值初值初始化滤波器。"""

import numpy as np
import pytest

from imu_ekf2d import EKFConfig, IMU2DEKF, TrajectorySimulator
from imu_ekf2d.runner import RunMetrics, run_scenario
from imu_ekf2d.simulator import ScenarioData


SEED = 20260928


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def simulator(rng) -> TrajectorySimulator:
    return TrajectorySimulator(rng=rng)


def make_ekf_from_truth(
    data: ScenarioData, config: EKFConfig | None = None
) -> IMU2DEKF:
    """用场景第一个真值初始化滤波器（bias 初值仍为 0，协方差保持默认）。"""
    t0 = data.truth[0]
    x0 = np.array([
        t0.px, t0.py, t0.vx, t0.vy, t0.theta,
        0.0, 0.0, 0.0,  # bias 真值对滤波器未知
    ])
    return IMU2DEKF(config=config, initial_state=x0)


def run(data: ScenarioData, config: EKFConfig | None = None) -> tuple[IMU2DEKF, RunMetrics]:
    ekf = make_ekf_from_truth(data, config)
    metrics = run_scenario(data, ekf)
    return ekf, metrics

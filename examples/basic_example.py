#!/usr/bin/env python3
"""最小可运行示例：代码生成一段转弯轨迹并跑 EKF，保存/恢复后继续。

运行::

    python3 examples/basic_example.py

所有数据都在内存中生成，状态快照写到本地 ./ekf_snapshot.json。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imu_ekf2d import EKF2D, NoiseConfig
from imu_ekf2d.runner import advance, compute_metrics, format_metrics, run_filter
from imu_ekf2d.scenarios import make_turn
from imu_ekf2d.simulator import SimResult


def main() -> int:
    scenario = make_turn()
    sim = scenario.sim

    ekf = EKF2D(
        initial_state=scenario.initial_state,
        initial_covariance=scenario.initial_covariance,
        noise_config=scenario.noise_config,
    )
    run = run_filter(sim, ekf=ekf,
                     gate_probability=scenario.gate_probability)
    metrics = compute_metrics(
        sim, run, steady_after=scenario.steady_after
    )
    print(format_metrics(scenario.name, metrics, run))

    # 演示保存/恢复：在轨迹一半处存盘，恢复后继续，结果与连续运行一致
    cut = int(len(sim.imu) // 2)
    head = SimResult(
        t_gt=sim.t_gt, x_gt=sim.x_gt,
        imu=sim.imu[:cut],
        observations=[o for o in sim.observations
                      if o.t < sim.imu[cut][0] - 1e-9],
        config=sim.config, segments=sim.segments,
    )
    ekf_head = EKF2D(
        initial_state=scenario.initial_state,
        initial_covariance=scenario.initial_covariance,
        noise_config=scenario.noise_config,
    )
    advance(ekf_head, head, gate_probability=0.99)

    snapshot_path = Path(__file__).resolve().parent / "ekf_snapshot.json"
    ekf_head.save(snapshot_path)
    print(f"\n滤波状态已保存到: {snapshot_path}")

    n_obs_cut = sum(
        1 for o in sim.observations if o.t < sim.imu[cut][0] - 1e-9
    )
    ekf_restored = EKF2D.load(snapshot_path)
    tail = advance(
        ekf_restored, sim, gate_probability=0.99,
        imu_start=cut, obs_start=n_obs_cut,
    )
    cont = run_filter(
        sim,
        gate_probability=0.99,
        initial_state=scenario.initial_state,
        initial_covariance=scenario.initial_covariance,
        noise_config=scenario.noise_config,
    )
    max_diff = float(np.max(np.abs(
        tail.estimates - cont.estimates[cut + 1:]
    )))
    print(f"保存-恢复继续与连续运行的最大状态差: {max_diff:.3e}")
    assert max_diff < 1e-12
    return 0


if __name__ == "__main__":
    sys.exit(main())

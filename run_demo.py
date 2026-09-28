#!/usr/bin/env python3
"""终端演示：在代码生成的本地轨迹上运行 2D IMU EKF，打印各场景估计误差。

运行::

    python3 run_demo.py

全程不访问网络 / ROS / GPS / 硬件；所有数据均在内存中生成，
状态快照演示会写入本地 JSON 文件 ekf_snapshot_demo.json。
"""

from __future__ import annotations

import os
import sys

import numpy as np

from imu_ekf2d import IMU2DEKF, TrajectorySimulator, wrap_angle
from imu_ekf2d.runner import run_scenario

SEED = 20260928
SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "ekf_snapshot_demo.json")


def ekf_from_truth(data) -> IMU2DEKF:
    t0 = data.truth[0]
    x0 = [t0.px, t0.py, t0.vx, t0.vy, t0.theta, 0.0, 0.0, 0.0]
    return IMU2DEKF(initial_state=np.asarray(x0))


def print_header() -> None:
    print("=" * 92)
    print("本地二维 IMU EKF 演示（纯内存/本地文件，无 ROS/GPS/硬件/云依赖）")
    print("=" * 92)
    print(f"{'场景':<14}{'位置RMSE(m)':>12}{'速度RMSE(m/s)':>14}"
          f"{'航向RMSE(deg)':>14}{'accBiasRMSE':>14}{'gyroBiasRMSE':>14}")
    print("-" * 92)


def report(name: str, m) -> dict:
    s = m.summary()
    print(f"{name:<14}{s['pos_rmse_m']:>12.4f}{s['vel_rmse_mps']:>14.4f}"
          f"{np.degrees(s['heading_rmse_rad']):>14.3f}"
          f"{s['accel_bias_rmse']:>14.4f}{s['gyro_bias_rmse']:>14.5f}")
    return s


def main() -> int:
    rng = np.random.default_rng(SEED)
    sim = TrajectorySimulator(rng=rng)

    print_header()

    # 1-5：标准运动场景
    scenarios = [
        ("静止", sim.stationary(duration=20.0)),
        ("匀速直线", sim.constant_velocity(duration=20.0, speed=2.0)),
        ("匀加速直线", sim.accel_straight(duration=20.0, accel=0.4)),
        ("匀速转弯", sim.turning(duration=24.0, speed=1.5, yaw_rate=0.35)),
        ("IMU bias", sim.bias_drift(duration=30.0, ba=(0.15, -0.10), bg=0.03)),
        ("航向跨±π", sim.angle_wrap(
            start_heading=np.pi - 0.4, yaw_rate=0.5, duration=8.0)),
    ]
    for name, data in scenarios:
        ekf = ekf_from_truth(data)
        m = run_scenario(data, ekf)
        s = report(name, m)
        if name == "IMU bias":
            ba_err = m.est[-1, 5:7] - np.array([0.15, -0.10])
            print(f"  └─ 末态 bias 估计: ba={np.round(m.est[-1, 5:7], 4)} "
                  f"(误差 {np.round(ba_err, 4)}), "
                  f"bg={m.est[-1, 7]:.4f} (误差 {m.est[-1, 7] - 0.03:+.4f})")
        if name == "航向跨±π":
            worst = max(
                abs(r.innovation[0]) for r in m.update_results
                if r.kind == "heading" and r.accepted)
            print(f"  └─ 穿越 ±π 期间接受的航向观测最大新息 = {worst:.4f} rad"
                  f"（无 2π 跳变）")

    # 6：异常观测 gating
    print("-" * 92)
    rng2 = np.random.default_rng(101)
    data = TrajectorySimulator(rng=np.random.default_rng(102)).constant_velocity(
        duration=20.0, speed=1.5)
    n_out = TrajectorySimulator.inject_position_outliers(
        data, rng=rng2, fraction=0.3, magnitude=20.0)
    ekf_g = ekf_from_truth(data)
    m_g = run_scenario(data, ekf_g)
    s_g = report("异常观测+门限", m_g)
    n_normal_acc = sum(
        1 for r in m_g.update_results if r.accepted and r.kind == "position")
    print(f"  └─ 注入 {n_out} 个位置离群值（20 m 偏移），"
          f"门限拒绝 {s_g['rejected_obs']} 个，正常位置观测接受 {n_normal_acc} 个；"
          f"位置 RMSE 仍为 {s_g['pos_rmse_m']:.3f} m")

    # 对照：关闭 gating
    from imu_ekf2d import EKFConfig
    data2 = TrajectorySimulator(rng=np.random.default_rng(102)).constant_velocity(
        duration=20.0, speed=1.5)
    TrajectorySimulator.inject_position_outliers(
        data2, rng=np.random.default_rng(101), fraction=0.3, magnitude=20.0)
    ekf_ng = IMU2DEKF(
        config=EKFConfig(enable_gating=False),
        initial_state=np.asarray([
            data2.truth[0].px, data2.truth[0].py,
            data2.truth[0].vx, data2.truth[0].vy,
            data2.truth[0].theta, 0.0, 0.0, 0.0]))
    m_ng = run_scenario(data2, ekf_ng)
    print(f"  └─ 对照（关闭门限）：位置 RMSE 恶化到 {m_ng.pos_rmse:.3f} m")

    # 7：保存 / 恢复一致性
    print("-" * 92)
    data = sim.turning(duration=24.0, speed=1.5, yaw_rate=0.3)
    obs_sorted = sorted(data.obs, key=lambda o: o.t)
    split = 227

    ekf_full = ekf_from_truth(data)
    oi = 0
    for sample in data.imu:
        ekf_full.predict(sample.dt, sample.accel_body, sample.gyro)
        while oi < len(obs_sorted) and obs_sorted[oi].t <= sample.t + 1e-9:
            o = obs_sorted[oi]
            ekf_full.update(o.kind, o.z)
            oi += 1

    ekf_a = ekf_from_truth(data)
    oi = 0
    for sample in data.imu[:split]:
        ekf_a.predict(sample.dt, sample.accel_body, sample.gyro)
        while oi < len(obs_sorted) and obs_sorted[oi].t <= sample.t + 1e-9:
            o = obs_sorted[oi]
            ekf_a.update(o.kind, o.z)
            oi += 1
    ekf_a.save(SNAPSHOT_PATH)
    size = os.path.getsize(SNAPSHOT_PATH)

    ekf_b = IMU2DEKF.load(SNAPSHOT_PATH)
    for sample in data.imu[split:]:
        ekf_b.predict(sample.dt, sample.accel_body, sample.gyro)
        while oi < len(obs_sorted) and obs_sorted[oi].t <= sample.t + 1e-9:
            o = obs_sorted[oi]
            ekf_b.update(o.kind, o.z)
            oi += 1

    dx = np.max(np.abs(ekf_b.x - ekf_full.x))
    dP = np.max(np.abs(ekf_b.P - ekf_full.P))
    print(f"状态保存/恢复：t={ekf_a.t:.2f}s 处写入本地文件 "
          f"{os.path.basename(SNAPSHOT_PATH)}（{size} 字节）")
    print(f"  └─ 恢复后续跑 vs 连续运行：状态最大差 = {dx:.2e}，"
          f"协方差最大差 = {dP:.2e}（应 ≈ 0）")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())

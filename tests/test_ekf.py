#!/usr/bin/env python3
"""imu_ekf2d 自动化测试。

直接运行（无需安装）::

    python3 tests/test_ekf.py

或用 unittest::

    python3 -m unittest -v tests/test_ekf.py

脚本先逐个场景打印估计误差（位置/速度/航向/零偏 RMSE、末端误差、
异常观测门限统计），再运行全部单元测试，任一断言失败时进程返回非零。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imu_ekf2d import (
    BAX,
    BAY,
    BG,
    EKF2D,
    NoiseConfig,
    PX,
    PY,
    STATE_DIM,
    VX,
    YAW,
    validate_covariance,
    wrap_pi,
)
from imu_ekf2d.runner import (
    compute_metrics,
    format_metrics,
    run_filter,
)
from imu_ekf2d.scenarios import (
    ALL_SCENARIOS,
    make_bias,
    make_constant_velocity,
    make_outlier,
    make_save_restore,
    make_static,
    make_turn,
    make_yaw_wrap,
)


# ---------------------------------------------------------------------------
# 场景级测试（代码生成真实轨迹 + 带噪 IMU/观测，输出估计误差）
# ---------------------------------------------------------------------------
def _run_scenario(scenario):
    ekf = EKF2D(
        initial_state=scenario.initial_state,
        initial_covariance=scenario.initial_covariance,
        noise_config=scenario.noise_config,
    )
    run = run_filter(
        scenario.sim,
        ekf=ekf,
        gate_probability=scenario.gate_probability,
    )
    metrics = compute_metrics(
        scenario.sim, run, steady_after=scenario.steady_after
    )
    print("\n" + format_metrics(scenario.name, metrics, run))
    return ekf, run, metrics


class TestScenarios(unittest.TestCase):
    def test_01_static(self):
        _, run, m = _run_scenario(make_static())
        self.assertLess(m.rmse_pos, 0.20, "静止场景位置 RMSE 过大")
        self.assertLess(m.rmse_vel, 0.08, "静止场景速度 RMSE 过大")
        self.assertLess(m.rmse_yaw, 0.05, "静止场景航向 RMSE 过大")
        self.assertLess(m.final_pos, 0.15)
        self.assertTrue(
            np.all(np.abs(run.estimates[:, YAW]) <= np.pi + 1e-9)
        )

    def test_02_constant_velocity(self):
        _, run, m = _run_scenario(make_constant_velocity())
        self.assertLess(m.rmse_pos, 0.25)
        self.assertLess(m.rmse_vel, 0.10)
        self.assertLess(m.rmse_yaw, 0.10)

    def test_03_turn(self):
        _, run, m = _run_scenario(make_turn())
        self.assertLess(m.rmse_pos, 0.40, "转弯场景位置误差过大")
        self.assertLess(m.rmse_vel, 0.15)
        self.assertLess(m.rmse_yaw, 0.08)

    def test_04_bias(self):
        _, run, m = _run_scenario(make_bias())
        print(
            "  真值 bias: bax=0.15, bay=-0.10 m/s^2, bg=0.03 rad/s | "
            f"末端估计: bax={run.estimates[-1, BAX]:+.4f}, "
            f"bay={run.estimates[-1, BAY]:+.4f}, "
            f"bg={run.estimates[-1, BG]:+.4f}"
        )
        self.assertLess(m.rmse_bax, 0.08, "加速度零偏未有效估计")
        self.assertLess(m.rmse_bay, 0.08)
        self.assertLess(m.rmse_bg, 0.02, "陀螺零偏未有效估计")
        self.assertLess(m.rmse_yaw, 0.08)
        self.assertLess(m.rmse_vel, 0.20)

    def test_05_yaw_wrap(self):
        _, run, m = _run_scenario(make_yaw_wrap())
        self.assertLess(m.rmse_yaw, 0.06, "跨界场景航向误差过大")
        self.assertLess(m.rmse_pos, 0.30)
        # 任何时刻航向都必须规范化在 (-pi, pi]
        self.assertTrue(
            np.all(run.estimates[:, YAW] > -np.pi - 1e-12)
        )
        self.assertTrue(
            np.all(run.estimates[:, YAW] <= np.pi + 1e-12)
        )
        # 轨迹累计转角约 ±9.6 rad，覆盖多次 ±pi 跨界
        total_span = run.times[-1]
        self.assertGreater(total_span, 20.0)

    def test_06_outlier_gating(self):
        _, run, m = _run_scenario(make_outlier())
        self.assertEqual(run.outliers_total, 4)
        self.assertEqual(
            run.outliers_rejected, 4,
            "所有注入离群值都应被 Mahalanobis 门限拒绝",
        )
        self.assertLessEqual(run.inliers_rejected, 20, "正常观测被误拒过多")
        self.assertLess(m.rmse_pos, 0.30)
        self.assertLess(m.rmse_vel, 0.15)

    def test_07_all_scenarios_finite_yaw(self):
        """全部场景：估计状态/协方差必须始终有限。"""
        for factory in ALL_SCENARIOS:
            scenario = factory()
            _, run, _ = _run_scenario(scenario)
            self.assertTrue(
                np.all(np.isfinite(run.estimates)),
                f"{scenario.name} 出现非有限状态",
            )
            self.assertTrue(
                np.all(np.isfinite(run.covariances)),
                f"{scenario.name} 出现非有限协方差",
            )


# ---------------------------------------------------------------------------
# 滤波器单元测试
# ---------------------------------------------------------------------------
class TestAngleWrap(unittest.TestCase):
    def test_wrap_scalar(self):
        self.assertAlmostEqual(float(wrap_pi(3 * np.pi / 2)), -np.pi / 2)
        self.assertAlmostEqual(float(wrap_pi(-3 * np.pi / 2)), np.pi / 2)
        self.assertAlmostEqual(float(wrap_pi(np.pi + 0.01)), -np.pi + 0.01)
        self.assertAlmostEqual(float(wrap_pi(0.0)), 0.0)
        self.assertAlmostEqual(float(wrap_pi(10 * np.pi)), 0.0)

    def test_wrap_array(self):
        a = np.array([3 * np.pi / 2, -3 * np.pi / 2, 7.0])
        w = wrap_pi(a)
        self.assertTrue(np.all(np.abs(w) <= np.pi))


class TestCovarianceValidation(unittest.TestCase):
    def test_accepts_valid(self):
        P = np.diag([1.0, 2.0])
        self.assertTrue(np.allclose(validate_covariance(P, "P"), P))
        # 半正定（含零特征值）在允许时通过
        Psd = np.diag([1.0, 0.0])
        validate_covariance(Psd, "Psd", positive_semidefinite=True)

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            validate_covariance(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            validate_covariance(np.eye(3), dim=2)

    def test_rejects_nan_inf(self):
        with self.assertRaises(ValueError):
            validate_covariance(np.array([[1.0, np.nan], [np.nan, 1.0]]))
        with self.assertRaises(ValueError):
            validate_covariance(np.array([[1.0, np.inf], [np.inf, 1.0]]))

    def test_rejects_non_symmetric(self):
        with self.assertRaises(ValueError):
            validate_covariance(np.array([[1.0, 0.5], [0.2, 1.0]]))

    def test_rejects_indefinite(self):
        with self.assertRaises(ValueError):
            validate_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))
        with self.assertRaises(ValueError):
            validate_covariance(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            validate_covariance(
                np.diag([1.0, 0.0]), positive_semidefinite=False
            )


class TestConstruction(unittest.TestCase):
    def test_bad_state_dim(self):
        with self.assertRaises(ValueError):
            EKF2D(initial_state=np.zeros(7))

    def test_bad_initial_covariance(self):
        with self.assertRaises(ValueError):
            EKF2D(initial_covariance=np.eye(7))
        with self.assertRaises(ValueError):
            EKF2D(initial_covariance=-np.eye(STATE_DIM))

    def test_bad_noise_config(self):
        with self.assertRaises(ValueError):
            NoiseConfig(acc_psd=-1.0)
        with self.assertRaises(ValueError):
            NoiseConfig(gyro_bias_psd=float("nan"))


class TestPredict(unittest.TestCase):
    def test_bad_dt(self):
        ekf = EKF2D()
        with self.assertRaises(ValueError):
            ekf.predict(0.0, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            ekf.predict(0.0, 0.0, 0.0, -0.01)

    def test_stationary_dynamics(self):
        """静止时白噪声零均值：长时间预测后状态漂移应很小。"""
        rng = np.random.default_rng(0)
        ekf = EKF2D(noise_config=NoiseConfig(acc_psd=1e-6, gyro_psd=1e-8,
                                             acc_bias_psd=0.0,
                                             gyro_bias_psd=0.0))
        dt = 0.01
        for _ in range(1000):
            ekf.predict(
                rng.normal(0, 1e-3), rng.normal(0, 1e-3),
                rng.normal(0, 1e-4), dt,
            )
        self.assertLess(abs(ekf.position[0]), 0.05)
        self.assertLess(abs(ekf.velocity[0]), 0.05)
        self.assertLess(abs(ekf.yaw), 0.02)

    def test_constant_acceleration(self):
        """世界系 x 向 1 m/s^2，10 步 0.1s 应得到 v=1, x=0.5。"""
        ekf = EKF2D(noise_config=NoiseConfig(acc_psd=0.0, gyro_psd=0.0,
                                             acc_bias_psd=0.0,
                                             gyro_bias_psd=0.0))
        # 机体系与世界系初始对齐，x 向加速度
        for _ in range(10):
            ekf.predict(1.0, 0.0, 0.0, 0.1)
        self.assertAlmostEqual(ekf.velocity[0], 1.0, places=6)
        self.assertAlmostEqual(ekf.position[0], 0.5, places=6)

    def test_covariance_grows_without_observations(self):
        ekf = EKF2D()
        p0 = ekf.P[VX, VX]
        for _ in range(100):
            ekf.predict(0.0, 0.0, 0.0, 0.01)
        self.assertGreater(ekf.P[VX, VX], p0)
        validate_covariance(ekf.P, "P", STATE_DIM,
                            positive_semidefinite=True)


class TestUpdate(unittest.TestCase):
    def test_dimension_mismatch(self):
        ekf = EKF2D()
        good_R = np.diag([0.1, 0.1])
        with self.assertRaises(ValueError):
            ekf.update([0.0, 0.0], np.zeros((2, 7)), good_R)
        with self.assertRaises(ValueError):
            ekf.update([0.0], np.zeros((2, STATE_DIM)), good_R)
        with self.assertRaises(ValueError):
            ekf.update(
                [0.0, 0.0],
                np.zeros((2, STATE_DIM)),
                np.array([[0.1, 0.5], [0.5, -0.1]]),
            )
        with self.assertRaises(ValueError):
            ekf.update(
                [np.nan, 0.0], np.zeros((2, STATE_DIM)), good_R
            )

    def test_position_update_reduces_variance(self):
        ekf = EKF2D(initial_state=np.zeros(STATE_DIM),
                    initial_covariance=np.eye(STATE_DIM))
        p_before = ekf.P[PX, PX]
        R = np.diag([0.25, 0.25])
        z = np.array([1.0, -1.0])
        res = ekf.update_position(z, R, gate_probability=None)
        self.assertTrue(res.accepted)
        np.testing.assert_allclose(
            res.innovation, z - res.predicted_measurement
        )
        # 被精确观测后，位置方差应减小并趋近 R
        self.assertLess(ekf.P[PX, PX], p_before)
        self.assertLess(ekf.P[PX, PX], 0.3)
        # 状态向观测靠拢
        self.assertAlmostEqual(ekf.position[0], 0.8, places=6)

    def test_kalman_gain_shape_and_symmetry(self):
        ekf = EKF2D()
        res = ekf.update_position(
            np.array([0.1, -0.1]), np.diag([0.1, 0.1]),
            gate_probability=None,
        )
        self.assertEqual(res.K.shape, (STATE_DIM, 2))
        self.assertEqual(res.S.shape, (2, 2))
        # S = H P H^T + R 对称正定
        np.testing.assert_allclose(res.S, res.S.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(res.S)[0], 0.0)
        validate_covariance(ekf.P, "P", STATE_DIM,
                            positive_semidefinite=True)

    def test_outlier_rejected_and_state_unchanged(self):
        x0 = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ekf = EKF2D(initial_state=x0,
                    initial_covariance=0.01 * np.eye(STATE_DIM))
        x_before, P_before = ekf.x.copy(), ekf.P.copy()
        res = ekf.update_position(
            np.array([50.0, -40.0]), np.diag([0.04, 0.04]),
            gate_probability=0.99,
        )
        self.assertFalse(res.accepted)
        self.assertGreater(res.mahalanobis, res.gate_threshold)
        np.testing.assert_array_equal(ekf.x, x_before)
        np.testing.assert_array_equal(ekf.P, P_before)

    def test_gate_can_be_disabled(self):
        x0 = np.zeros(STATE_DIM)
        ekf = EKF2D(initial_state=x0,
                    initial_covariance=0.01 * np.eye(STATE_DIM))
        res = ekf.update_position(
            np.array([50.0, -40.0]), np.diag([0.04, 0.04]),
            gate_probability=None,
        )
        self.assertTrue(res.accepted)

    def test_gate_sigma_parameter(self):
        ekf = EKF2D(initial_covariance=0.01 * np.eye(STATE_DIM))
        # 1-sigma 门限：3σ 外的观测被拒
        res_bad = ekf.update_position(
            np.array([0.5, 0.0]), np.diag([0.01, 0.01]), gate_sigma=3.0
        )
        self.assertFalse(res_bad.accepted)
        res_ok = ekf.update_position(
            np.array([0.0, 0.0]), np.diag([0.01, 0.01]), gate_sigma=3.0
        )
        self.assertTrue(res_ok.accepted)


class TestYawObservationWrap(unittest.TestCase):
    def test_innovation_does_not_jump_across_boundary(self):
        # 估计航向在 +pi 附近，观测在 -pi 附近，物理上只差 0.1 rad
        x0 = np.zeros(STATE_DIM)
        x0[YAW] = np.pi - 0.05
        P0 = 0.1 * np.eye(STATE_DIM)
        ekf = EKF2D(initial_state=x0, initial_covariance=P0)
        z = wrap_pi(-np.pi + 0.05)
        res = ekf.update_yaw(float(z), np.array([[1e-4]]),
                             gate_probability=None)
        self.assertTrue(res.accepted)
        self.assertAlmostEqual(res.innovation[0], 0.1, places=6)
        # 更新后航向在物理上仍靠近 +pi（规范化后落在 -pi 端，
        # 到 +pi 的圆周距离应仍远小于 pi，而不是跳到反方向 0 附近）
        self.assertLess(abs(wrap_pi(ekf.yaw - np.pi)), 0.1)
        self.assertLess(abs(ekf.yaw), np.pi + 1e-12)
        self.assertGreater(ekf.yaw, -np.pi - 1e-12)

    def test_innovation_other_crossing(self):
        x0 = np.zeros(STATE_DIM)
        x0[YAW] = -np.pi + 0.05
        P0 = 0.1 * np.eye(STATE_DIM)
        ekf = EKF2D(initial_state=x0, initial_covariance=P0)
        res = ekf.update_yaw(np.pi - 0.05, np.array([[1e-4]]),
                             gate_probability=None)
        self.assertAlmostEqual(res.innovation[0], -0.1, places=6)

    def test_yaw_outlier_rejected(self):
        x0 = np.zeros(STATE_DIM)
        x0[YAW] = 0.0
        ekf = EKF2D(initial_state=x0,
                    initial_covariance=0.01 * np.eye(STATE_DIM))
        res = ekf.update_yaw(3.0, np.array([[1e-6]]),
                             gate_probability=0.99)
        self.assertFalse(res.accepted)


class TestSaveRestore(unittest.TestCase):
    def test_save_restore_matches_continuous(self):
        from imu_ekf2d.runner import advance

        scenario = make_save_restore()
        t_cut = 11.0
        imu_rate = scenario.sim.config.imu_rate
        obs_rate = scenario.sim.config.obs_rate

        # 连续运行（基准）
        ekf_cont = EKF2D(
            initial_state=scenario.initial_state,
            initial_covariance=scenario.initial_covariance,
            noise_config=scenario.noise_config,
        )
        run_cont = run_filter(
            scenario.sim, ekf=ekf_cont,
            gate_probability=scenario.gate_probability,
        )

        # 分段运行：前半段处理 t < t_cut 的样本，t == t_cut 的样本
        # 留给恢复后的滤波器处理（二者共同覆盖全部样本且不重复）
        n_imu_cut = int(round(t_cut * imu_rate))      # 前半段 IMU 条数
        n_obs_cut = 0
        for ob in scenario.sim.observations:
            if ob.t < t_cut - 1e-9:
                n_obs_cut += 1

        # 只包含前半段 IMU/观测的截断仿真（数据本身不复制修改）
        from imu_ekf2d.simulator import SimResult

        head_sim = SimResult(
            t_gt=scenario.sim.t_gt,
            x_gt=scenario.sim.x_gt,
            imu=scenario.sim.imu[:n_imu_cut],
            observations=[
                o for o in scenario.sim.observations if o.t < t_cut - 1e-9
            ],
            config=scenario.sim.config,
            segments=scenario.sim.segments,
        )
        ekf_head = EKF2D(
            initial_state=scenario.initial_state,
            initial_covariance=scenario.initial_covariance,
            noise_config=scenario.noise_config,
        )
        advance(
            ekf_head, head_sim,
            gate_probability=scenario.gate_probability,
            imu_start=0,
        )

        snapshot = ekf_head.to_dict()
        x_at_cut, P_at_cut = ekf_head.x.copy(), ekf_head.P.copy()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ekf_state.json")
            ekf_head.save(path)
            ekf_b = EKF2D.load(path)

        # 恢复点状态/协方差必须一致
        np.testing.assert_allclose(ekf_b.x, x_at_cut, rtol=0, atol=0)
        np.testing.assert_allclose(ekf_b.P, P_at_cut, rtol=0, atol=0)
        self.assertEqual(ekf_b.noise, ekf_head.noise)

        # 也验证内存字典往返
        ekf_c = EKF2D.from_dict(snapshot)
        np.testing.assert_allclose(ekf_c.x, x_at_cut, rtol=0, atol=0)

        # 用恢复后的滤波器继续处理与连续运行相同的剩余输入，
        # 逐时刻比较（纯确定性运算，应完全一致）
        run_tail = advance(
            ekf_b, scenario.sim,
            gate_probability=scenario.gate_probability,
            imu_start=n_imu_cut,
            obs_start=n_obs_cut,
        )
        for j, imu_index in enumerate(range(n_imu_cut, len(scenario.sim.imu))):
            t = scenario.sim.imu[imu_index][0]
            np.testing.assert_allclose(
                run_tail.estimates[j],
                run_cont.estimates[imu_index + 1],
                rtol=0, atol=1e-12,
                err_msg=f"恢复后继续处理与连续运行在 t={t:.2f} 不一致",
            )
            np.testing.assert_allclose(
                run_tail.covariances[j],
                run_cont.covariances[imu_index + 1],
                rtol=0, atol=1e-12,
            )

        # 误差指标仍然合格
        metrics = compute_metrics(
            scenario.sim, run_cont, steady_after=scenario.steady_after
        )
        print("\n" + format_metrics(scenario.name, metrics, run_cont))
        self.assertLess(metrics.rmse_pos, 0.40)
        self.assertLess(metrics.rmse_vel, 0.15)
        self.assertLess(metrics.rmse_bax, 0.08)
        self.assertLess(metrics.rmse_bg, 0.02)

    def test_snapshot_is_plain_json(self):
        """快照必须是可 JSON 序列化的纯本地数据（无外部依赖）。"""
        import json
        ekf = EKF2D()
        ekf.predict(0.1, -0.2, 0.01, 0.1)
        text = json.dumps(ekf.to_dict())
        restored = EKF2D.from_dict(json.loads(text))
        np.testing.assert_allclose(restored.x, ekf.x)
        np.testing.assert_allclose(restored.P, ekf.P)


class TestDeterminism(unittest.TestCase):
    def test_repeated_runs_identical(self):
        scenario = make_static()
        _, run1, _ = _run_scenario(scenario)
        _, run2, _ = _run_scenario(scenario)
        np.testing.assert_array_equal(run1.estimates, run2.estimates)
        np.testing.assert_array_equal(run1.covariances, run2.covariances)


# ---------------------------------------------------------------------------
# 入口：先打印各场景误差，再跑全部单元测试
# ---------------------------------------------------------------------------
def _print_scenario_overview() -> None:
    print("=" * 72)
    print("场景概览（代码生成轨迹 + 带噪 IMU/观测）")
    print("=" * 72)
    for factory in ALL_SCENARIOS:
        scenario = factory()
        ekf = EKF2D(
            initial_state=scenario.initial_state,
            initial_covariance=scenario.initial_covariance,
            noise_config=scenario.noise_config,
        )
        run = run_filter(
            scenario.sim, ekf=ekf,
            gate_probability=scenario.gate_probability,
        )
        metrics = compute_metrics(
            scenario.sim, run, steady_after=scenario.steady_after
        )
        print(format_metrics(scenario.name, metrics, run))
    print("=" * 72)
    print("运行单元测试...")
    print("=" * 72)


def main() -> int:
    _print_scenario_overview()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("\n全部测试通过 ✔")
        return 0
    print("\n存在失败测试 ✘", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

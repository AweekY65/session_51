"""端到端场景测试：代码生成真值轨迹与带噪 IMU/观测，运行 EKF 并输出估计误差。

覆盖：静止、匀速、转弯、IMU bias、匀加速、异常观测 gating、
航向角跨 ±pi、保存/恢复一致性。所有测试仅使用本地 numpy 数据。

直接运行（终端打印误差）::

    python -m tests.test_ekf_scenarios -v
或
    pytest tests/ -s
"""

import os
import tempfile

import numpy as np
import pytest

from imu_ekf2d import EKFConfig, EKFError, EKFGateReject, IMU2DEKF
from imu_ekf2d.math_utils import wrap_angle
from imu_ekf2d.runner import run_scenario
from imu_ekf2d.simulator import TrajectorySimulator

from .conftest import make_ekf_from_truth, run as run_data


def _print_errors(name: str, m) -> dict:
    s = m.summary()
    print(
        f"\n[{name}] 位置RMSE={s['pos_rmse_m']:.4f} m | "
        f"速度RMSE={s['vel_rmse_mps']:.4f} m/s | "
        f"航向RMSE={np.degrees(s['heading_rmse_rad']):.3f} deg "
        f"({s['heading_rmse_rad']:.4f} rad) | "
        f"加速度bias RMSE={s['accel_bias_rmse']:.4f} | "
        f"陀螺bias RMSE={s['gyro_bias_rmse']:.5f} | "
        f"拒绝观测={s['rejected_obs']}/{s['rejected_obs'] + sum(
            1 for r in m.update_results if r.accepted)}"
    )
    return s


# ---------------------------------------------------------------------------
# 1. 静止
# ---------------------------------------------------------------------------
class TestStationary:
    def test_stationary_errors(self, simulator):
        data = simulator.stationary(duration=20.0)
        ekf, m = run_data(data)
        s = _print_errors("静止", m)
        assert s["pos_rmse_m"] < 0.6
        assert s["vel_rmse_mps"] < 0.3
        assert s["heading_rmse_rad"] < 0.15

    def test_stationary_no_drift(self, simulator):
        """无观测纯预测在静止时不应产生显著位置漂移（bias 初值为 0）。"""
        data = simulator.stationary(duration=5.0)
        ekf = make_ekf_from_truth(data)
        m = run_scenario(data, ekf, use_obs=False)
        assert abs(ekf.position[0]) < 0.5
        assert abs(ekf.position[1]) < 0.5


# ---------------------------------------------------------------------------
# 2. 匀速直线
# ---------------------------------------------------------------------------
class TestConstantVelocity:
    def test_constant_velocity(self, simulator):
        data = simulator.constant_velocity(duration=20.0, speed=2.0)
        ekf, m = run_data(data)
        s = _print_errors("匀速直线", m)
        assert s["pos_rmse_m"] < 0.6
        assert s["vel_rmse_mps"] < 0.5
        # 末态速度误差远小于 1 m/s
        assert abs(m.est[-1, 2] - 2.0) < 0.3


# ---------------------------------------------------------------------------
# 3. 转弯
# ---------------------------------------------------------------------------
class TestTurning:
    def test_turning_tracks_arc(self, simulator):
        yaw_rate = 0.35
        data = simulator.turning(
            duration=24.0, speed=1.5, yaw_rate=yaw_rate)
        ekf, m = run_data(data)
        s = _print_errors("匀速转弯", m)
        assert s["pos_rmse_m"] < 0.8
        assert s["vel_rmse_mps"] < 0.5
        assert s["heading_rmse_rad"] < 0.2
        # 真值总转角 8.4 rad；末态航向误差应较小
        true_last_theta = data.truth[-1].theta
        assert abs(wrap_angle(ekf.heading - true_last_theta)) < 0.25


# ---------------------------------------------------------------------------
# 4. IMU bias 估计
# ---------------------------------------------------------------------------
class TestBias:
    def test_bias_converges(self, simulator):
        data = simulator.bias_drift(
            duration=30.0, ba=(0.15, -0.10), bg=0.03)
        ekf, m = run_data(data)
        s = _print_errors("IMU bias", m)
        ba_true = np.array([0.15, -0.10])
        ba_est = m.est[-1, 5:7]
        bg_err = abs(m.est[-1, 7] - 0.03)
        print(f"  末态加速度bias估计={np.round(ba_est, 4)} "
              f"(真值 {ba_true})，误差={np.round(ba_est - ba_true, 4)}；"
              f"陀螺bias误差={bg_err:.5f}")
        assert np.linalg.norm(ba_est - ba_true) < 0.08
        assert bg_err < 0.01
        assert s["pos_rmse_m"] < 0.8

    def test_prediction_only_drifts_with_bias(self, simulator):
        """有 bias 但无观测时，纯航位推算应明显漂移——说明 bias 通道有效。"""
        data = simulator.bias_drift(
            duration=10.0, ba=(0.3, 0.0), bg=0.0)
        ekf = make_ekf_from_truth(data)
        run_scenario(data, ekf, use_obs=False)
        # 0.3 m/s^2 未补偿 bias，10 s 位置漂移应在 0.5*0.3*100=15 m 量级
        assert abs(ekf.position[0]) > 3.0


# ---------------------------------------------------------------------------
# 5. 匀加速
# ---------------------------------------------------------------------------
class TestAcceleration:
    def test_accel_straight(self, simulator):
        data = simulator.accel_straight(duration=20.0, accel=0.4)
        ekf, m = run_data(data)
        s = _print_errors("匀加速直线", m)
        assert s["pos_rmse_m"] < 0.8
        assert s["vel_rmse_mps"] < 0.5
        # 末速真值 0.4*20 = 8 m/s
        assert abs(m.est[-1, 2] - 8.0) < 0.6


# ---------------------------------------------------------------------------
# 6. 异常观测 gating（Mahalanobis）
# ---------------------------------------------------------------------------
class TestGating:
    def test_outliers_are_rejected(self, rng):
        sim = TrajectorySimulator(rng=np.random.default_rng(11))
        data = sim.constant_velocity(duration=20.0, speed=1.5)
        n_inj = TrajectorySimulator.inject_position_outliers(
            data, rng=np.random.default_rng(12),
            fraction=0.3, magnitude=20.0)
        assert n_inj >= 5
        ekf = make_ekf_from_truth(data)
        m = run_scenario(data, ekf)
        s = _print_errors("异常观测 gating", m)

        # 所有注入的离群位置观测都应被拒绝
        rejected_pos = {
            (r.kind, round(float(r.z[0]), 6), round(float(r.z[1]), 6))
            for r in m.update_results if not r.accepted
        }
        outlier_zs = {
            ("position", round(float(o.z[0]), 6), round(float(o.z[1]), 6))
            for o in data.obs if o.outlier
        }
        assert outlier_zs.issubset(rejected_pos), (
            f"有离群观测未被拒绝：{outlier_zs - rejected_pos}")
        # 拒绝数量 >= 注入数量；正常观测应全部接受
        assert s["rejected_obs"] >= n_inj
        # 滤波精度不应被离群值破坏
        assert s["pos_rmse_m"] < 1.0

    def test_gating_disabled_accepts_outliers(self, rng):
        sim = TrajectorySimulator(rng=np.random.default_rng(11))
        data = sim.constant_velocity(duration=12.0, speed=1.0)
        TrajectorySimulator.inject_position_outliers(
            data, rng=np.random.default_rng(13),
            fraction=0.4, magnitude=25.0)
        cfg = EKFConfig(enable_gating=False)
        ekf = make_ekf_from_truth(data, cfg)
        m = run_scenario(data, ekf)
        assert m.rejected == 0
        # 没有门限保护时位置误差显著增大
        assert m.pos_rmse > 3.0

    def test_strict_mode_raises(self, simulator):
        data = simulator.stationary(duration=2.0)
        cfg = EKFConfig(gate_threshold=0.5)
        ekf = make_ekf_from_truth(data, cfg)
        # 0.5m 噪声相对门限偏大，至少有一次观测应触发 strict 异常
        with pytest.raises(EKFGateReject):
            for sample in data.imu:
                ekf.predict(sample.dt, sample.accel_body, sample.gyro)
                for o in data.obs:
                    if abs(o.t - sample.t) < 1e-9:
                        ekf.update(o.kind, o.z, strict=True)

    def test_mahalanobis_math(self, simulator):
        """手动核对一次更新的 innovation/K/S/P 更新与 Mahalanobis 距离。"""
        data = simulator.stationary(duration=1.0)
        ekf = make_ekf_from_truth(data)
        sample = data.imu[0]
        ekf.predict(sample.dt, sample.accel_body, sample.gyro)
        P_before = ekf.P.copy()
        z = np.array([0.1, -0.2])
        r = ekf.update("position", z)
        H = np.zeros((2, 8)); H[0, 0] = 1.0; H[1, 1] = 1.0
        S_expected = H @ P_before @ H.T + ekf.config.R_pos
        assert np.allclose(r.S, S_expected)
        K_expected = P_before @ H.T @ np.linalg.inv(S_expected)
        assert np.allclose(r.K, K_expected)
        maha_expected = np.sqrt(
            r.innovation @ np.linalg.inv(S_expected) @ r.innovation)
        assert abs(r.maha - maha_expected) < 1e-10
        # 协方差更新后不确定度下降
        assert ekf.P[0, 0] < P_before[0, 0]


# ---------------------------------------------------------------------------
# 7. 航向角跨 ±pi
# ---------------------------------------------------------------------------
class TestAngleWrap:
    def test_wrap_angle_helper(self):
        # 约定 [-pi, pi)：+pi 与 -pi 等价，统一取 -pi
        assert abs(wrap_angle(np.pi) - (-np.pi)) < 1e-12
        assert abs(wrap_angle(-np.pi) - (-np.pi)) < 1e-12
        assert abs(wrap_angle(3 * np.pi) - (-np.pi)) < 1e-12
        assert abs(wrap_angle(-3 * np.pi) - (-np.pi)) < 1e-12
        assert abs(wrap_angle(2 * np.pi + 0.1) - 0.1) < 1e-12
        assert abs(wrap_angle(np.pi + 0.01) - (-np.pi + 0.01)) < 1e-12
        assert abs(wrap_angle(np.pi - 0.01) - (np.pi - 0.01)) < 1e-12
        # 向量化
        out = wrap_angle(np.array([0.0, 2 * np.pi, -2 * np.pi]))
        assert np.allclose(out, [0, 0, 0], atol=1e-12)

    def test_innovation_no_2pi_jump(self):
        """预测航向 +3.05、观测 -3.05（物理上仅差 ~0.18 rad），
        规范化后的 innovation 必须约为 0.18，而不是 -6.1。"""
        ekf = IMU2DEKF(initial_state=np.array(
            [0, 0, 0, 0, 3.05, 0, 0, 0]))
        r = ekf.update("heading", -3.05)
        assert abs(r.innovation[0] - wrap_angle(-3.05 - 3.05)) < 1e-12
        assert abs(r.innovation[0]) < 0.2
        assert r.accepted

    def test_full_trajectory_crosses_pi(self, simulator):
        """从 pi-0.4 开始持续旋转的完整轨迹，多次穿越 ±pi，
        用真值初值初始化后全程航向误差保持较小。"""
        data = simulator.angle_wrap(
            start_heading=np.pi - 0.4, yaw_rate=0.5, duration=8.0)
        # 真值在 8s 内旋转 4 rad，穿越 +pi 一次
        thetas = np.array([s.theta for s in data.truth])
        assert np.any(thetas > 2.5) and np.any(np.abs(thetas) < 1.0)

        ekf = make_ekf_from_truth(data)
        m = run_scenario(data, ekf)
        s = _print_errors("航向跨±π", m)
        assert s["heading_rmse_rad"] < 0.15
        assert s["pos_rmse_m"] < 0.8

        # 逐步检查每次航向更新的 innovation 都在 (-pi, pi]，无 2pi 跳变
        for r in m.update_results:
            if r.kind == "heading":
                assert -np.pi < r.innovation[0] <= np.pi + 1e-9
                if r.accepted:
                    assert abs(r.innovation[0]) < 1.0  # 实际误差远小于 pi


# ---------------------------------------------------------------------------
# 8. 状态保存 / 恢复
# ---------------------------------------------------------------------------
class TestSaveRestore:
    def test_save_restore_matches_continuous(self, simulator):
        data = simulator.turning(duration=24.0, speed=1.5, yaw_rate=0.3)
        obs_sorted = sorted(data.obs, key=lambda o: o.t)
        split_step = 227  # 11.35 s，位于样本边界

        def apply_due_obs(ekf, t_now, start):
            i = start
            while i < len(obs_sorted) and obs_sorted[i].t <= t_now + 1e-9:
                o = obs_sorted[i]
                ekf.update(o.kind, o.z)
                i += 1
            return i

        # 连续运行
        ekf_full = make_ekf_from_truth(data)
        oi = 0
        for sample in data.imu:
            ekf_full.predict(sample.dt, sample.accel_body, sample.gyro)
            oi = apply_due_obs(ekf_full, sample.t, oi)

        # 分段运行：先跑到 split_step，保存快照
        ekf_a = make_ekf_from_truth(data)
        oi = 0
        for k, sample in enumerate(data.imu[:split_step]):
            ekf_a.predict(sample.dt, sample.accel_body, sample.gyro)
            oi = apply_due_obs(ekf_a, sample.t, oi)
        t_split = ekf_a.t

        fd, snapshot_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        ekf_a.save(snapshot_path)

        # 恢复并从下一个样本继续（严格跳过已处理样本，不重复消费观测）
        ekf_b = IMU2DEKF.load(snapshot_path)
        assert abs(ekf_b.t - t_split) < 1e-12
        for sample in data.imu[split_step:]:
            ekf_b.predict(sample.dt, sample.accel_body, sample.gyro)
            oi = apply_due_obs(ekf_b, sample.t, oi)

        # 恢复后续跑结果必须与连续运行一致
        assert np.allclose(ekf_b.x, ekf_full.x, atol=1e-10)
        assert np.allclose(ekf_b.P, ekf_full.P, atol=1e-10)
        assert abs(ekf_b.t - ekf_full.t) < 1e-10
        assert oi == len(obs_sorted)
        os.remove(snapshot_path)

    def test_dict_roundtrip_preserves_state(self, simulator):
        data = simulator.bias_drift(duration=10.0)
        ekf, _ = run_data(data)
        ekf2 = IMU2DEKF.from_dict(ekf.to_dict())
        assert np.allclose(ekf2.x, ekf.x)
        assert np.allclose(ekf2.P, ekf.P)
        assert ekf2.t == ekf.t
        assert ekf2.config.gate_threshold == ekf.config.gate_threshold
        assert ekf2.config.acc_noise_psd == ekf.config.acc_noise_psd

    def test_resume_prediction_identical(self, simulator):
        """恢复后只做预测，结果也应与连续预测逐位一致。"""
        data = simulator.stationary(duration=6.0)
        ekf1 = make_ekf_from_truth(data)
        samples = data.imu
        for sample in samples[:60]:
            ekf1.predict(sample.dt, sample.accel_body, sample.gyro)
        snapshot = ekf1.to_dict()

        ekf2 = IMU2DEKF.from_dict(snapshot)
        for sample in samples[60:]:
            ekf1.predict(sample.dt, sample.accel_body, sample.gyro)
            ekf2.predict(sample.dt, sample.accel_body, sample.gyro)
        assert np.allclose(ekf1.x, ekf2.x, atol=1e-12)
        assert np.allclose(ekf1.P, ekf2.P, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. 非法 covariance / 维度 / 噪声参数检测
# ---------------------------------------------------------------------------
class TestValidation:
    def test_non_symmetric_P_rejected(self):
        P = np.eye(8)
        P[0, 1] = 0.5  # 只填了上三角，P[1, 0] 仍为 0 => 非对称
        with pytest.raises(EKFError, match="非对称"):
            IMU2DEKF(initial_P=P)

    def test_negative_definite_P_rejected(self):
        P = -np.eye(8)
        with pytest.raises(EKFError, match="非正定"):
            IMU2DEKF(initial_P=P)

    def test_wrong_dim_P_rejected(self):
        with pytest.raises(EKFError, match="维度错误"):
            IMU2DEKF(initial_P=np.eye(7))

    def test_non_finite_P_rejected(self):
        P = np.eye(8); P[0, 0] = np.nan
        with pytest.raises(EKFError):
            IMU2DEKF(initial_P=P)

    def test_wrong_dim_state(self):
        with pytest.raises(EKFError, match="初始状态维度"):
            IMU2DEKF(initial_state=np.zeros(7))

    def test_bad_R_matrix(self):
        ekf = IMU2DEKF()
        with pytest.raises(EKFError):
            ekf.update("position", np.zeros(2), R=-np.eye(2))
        with pytest.raises(EKFError, match="维度"):
            ekf.update("position", np.zeros(2), R=np.eye(3))

    def test_bad_config_noise(self):
        with pytest.raises(EKFError):
            EKFConfig(acc_noise_psd=-1.0)
        with pytest.raises(EKFError):
            EKFConfig(gate_threshold=0.0)
        with pytest.raises(EKFError):
            EKFConfig(R_pos=np.diag([0.1, -0.1]))

    def test_bad_dt_and_inputs(self):
        ekf = IMU2DEKF()
        with pytest.raises(EKFError):
            ekf.predict(0.0, np.zeros(2), 0.0)
        with pytest.raises(EKFError):
            ekf.predict(-0.1, np.zeros(2), 0.0)
        with pytest.raises(EKFError):
            ekf.predict(0.1, np.array([np.nan, 0.0]), 0.0)
        with pytest.raises(EKFError):
            ekf.predict(0.1, np.zeros(2), np.inf)

    def test_unknown_observation_kind(self):
        ekf = IMU2DEKF()
        with pytest.raises(EKFError):
            ekf.update("gps", np.zeros(2))

    def test_wrong_dim_observation(self):
        ekf = IMU2DEKF()
        ekf.predict(0.05, np.zeros(2), 0.0)
        with pytest.raises(EKFError, match="观测维度"):
            ekf.update("position", np.zeros(3))
        with pytest.raises(EKFError, match="观测维度"):
            ekf.update("velocity", [1.0, 2.0, 3.0])
        with pytest.raises(EKFError, match="NaN"):
            ekf.update("position", np.array([np.nan, 0.0]))

    def test_bad_accel_input(self):
        ekf = IMU2DEKF()
        with pytest.raises(EKFError, match="accel_body"):
            ekf.predict(0.05, np.zeros(3), 0.0)

    def test_P_stays_symmetric_positive(self, simulator):
        """整条轨迹跑完，P 始终对称正定（predict/update 内部已做校验，
        这里额外检查特征值为正）。"""
        data = simulator.turning(duration=10.0)
        ekf = make_ekf_from_truth(data)
        for sample in data.imu:
            ekf.predict(sample.dt, sample.accel_body, sample.gyro)
            for o in data.obs:
                if abs(o.t - sample.t) < 1e-9:
                    ekf.update(o.kind, o.z)
            assert np.allclose(ekf.P, ekf.P.T, atol=1e-10)
            assert np.linalg.eigvalsh(ekf.P).min() > -1e-10

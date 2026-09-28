"""代码生成的二维真值轨迹与带噪 IMU/观测仿真器。

完全本地运行：不连接任何硬件或服务，仅用 numpy 生成数据。
真值动力学与滤波器使用相同的平面模型（R(theta) 旋转机体系加速度），
便于在已知 ground truth 下评估 EKF 的估计误差。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .math_utils import wrap_angle


@dataclass
class TruthState:
    """某一时刻的真值状态。"""

    t: float
    px: float
    py: float
    vx: float
    vy: float
    theta: float
    ba_x: float = 0.0
    ba_y: float = 0.0
    b_g: float = 0.0

    @property
    def array(self) -> np.ndarray:
        return np.array(
            [self.px, self.py, self.vx, self.vy,
             self.theta, self.ba_x, self.ba_y, self.b_g]
        )


@dataclass
class IMUSample:
    t: float
    dt: float
    accel_body: np.ndarray  # 含 bias 与白噪声的机体系加速度测量
    gyro: float             # 含 bias 与白噪声的角速度测量
    accel_true: np.ndarray = field(default_factory=lambda: np.zeros(2))
    gyro_true: float = 0.0


@dataclass
class Observation:
    t: float
    kind: str          # "position" | "velocity" | "heading"
    z: np.ndarray
    z_true: np.ndarray
    outlier: bool = False


@dataclass
class ScenarioData:
    """一个完整场景：真值、IMU 流与观测流。"""

    name: str
    truth: list[TruthState]
    imu: list[IMUSample]
    obs: list[Observation]

    def truth_array(self) -> np.ndarray:
        return np.array([s.array for s in self.truth])


@dataclass
class _Segment:
    duration: float
    accel_body_fn: object     # fn(t_local, state) -> np.ndarray(2) 真值机体系加速度
    gyro_fn: object           # fn(t_local, state) -> float 真值角速度
    ba: tuple[float, float] = (0.0, 0.0)
    bg: float = 0.0


class TrajectorySimulator:
    """生成真实轨迹 + 带噪 IMU + 带噪位置/速度/航向观测。

    参数
    ----
    dt : IMU 采样周期（秒）
    accel_noise_std / gyro_noise_std : IMU 白噪声标准差（单步测量）
    pos_noise_std / vel_noise_std / heading_noise_std : 观测噪声标准差
    obs_interval : 每隔多少秒产生一组观测
    rng : 随机数生成器或整数种子，保证测试可复现
    """

    def __init__(
        self,
        dt: float = 0.05,
        accel_noise_std: float = 0.1,
        gyro_noise_std: float = 0.02,
        pos_noise_std: float = 0.5,
        vel_noise_std: float = 0.2,
        heading_noise_std: float = 0.1,
        obs_interval: float = 0.5,
        rng: np.random.Generator | int | None = 0,
    ) -> None:
        self.dt = dt
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.heading_noise_std = heading_noise_std
        self.obs_interval = obs_interval
        self.rng = (
            rng if isinstance(rng, np.random.Generator)
            else np.random.default_rng(rng)
        )

    # ------------------------------------------------------------------
    def simulate(
        self,
        name: str,
        segments: list[_Segment],
        initial: TruthState | None = None,
        include_vel: bool = True,
        include_heading: bool = True,
    ) -> ScenarioData:
        """按分段指令积分真值，并同步采样 IMU 与观测。"""
        if initial is None:
            initial = TruthState(t=0.0, px=0, py=0, vx=0, vy=0, theta=0)

        s = TruthState(**{**initial.__dict__})
        truth: list[TruthState] = [self._copy(s)]
        imu: list[IMUSample] = []
        obs: list[Observation] = []

        t = 0.0
        next_obs_t = self.obs_interval
        for seg in segments:
            n_steps = int(round(seg.duration / self.dt))
            t_seg0 = t
            for _ in range(n_steps):
                tau = t - t_seg0
                a_b_cmd = np.asarray(seg.accel_body_fn(tau, s), dtype=float)
                g_cmd = float(seg.gyro_fn(tau, s))
                s.ba_x, s.ba_y = seg.ba
                s.b_g = seg.bg

                # IMU 测量 = 真值 + bias + 白噪声
                a_meas = a_b_cmd + np.array(seg.ba) + self.rng.normal(
                    0.0, self.accel_noise_std, size=2
                )
                g_meas = g_cmd + seg.bg + self.rng.normal(
                    0.0, self.gyro_noise_std
                )
                imu.append(IMUSample(
                    t=t + self.dt, dt=self.dt,
                    accel_body=a_meas, gyro=g_meas,
                    accel_true=a_b_cmd, gyro_true=g_cmd,
                ))

                # 真值积分（与 EKF 同一模型）
                c, ss = np.cos(s.theta), np.sin(s.theta)
                R = np.array([[c, -ss], [ss, c]])
                a_w = R @ a_b_cmd
                s.px += s.vx * self.dt + 0.5 * a_w[0] * self.dt ** 2
                s.py += s.vy * self.dt + 0.5 * a_w[1] * self.dt ** 2
                s.vx += a_w[0] * self.dt
                s.vy += a_w[1] * self.dt
                s.theta = wrap_angle(s.theta + g_cmd * self.dt)
                t += self.dt
                s.t = t
                truth.append(self._copy(s))

                if t + 1e-9 >= next_obs_t:
                    self._emit_obs(s, obs, include_vel, include_heading)
                    next_obs_t += self.obs_interval

        return ScenarioData(name=name, truth=truth, imu=imu, obs=obs)

    def _emit_obs(
        self, s: TruthState, obs: list, include_vel: bool, include_heading: bool
    ) -> None:
        z_p = np.array([s.px, s.py]) + self.rng.normal(
            0.0, self.pos_noise_std, size=2)
        obs.append(Observation(
            t=s.t, kind="position", z=z_p,
            z_true=np.array([s.px, s.py])))
        if include_vel:
            z_v = np.array([s.vx, s.vy]) + self.rng.normal(
                0.0, self.vel_noise_std, size=2)
            obs.append(Observation(
                t=s.t, kind="velocity", z=z_v,
                z_true=np.array([s.vx, s.vy])))
        if include_heading:
            z_h = wrap_angle(s.theta + self.rng.normal(
                0.0, self.heading_noise_std))
            obs.append(Observation(
                t=s.t, kind="heading", z=np.array([z_h]),
                z_true=np.array([s.theta])))

    @staticmethod
    def inject_position_outliers(
        data: ScenarioData,
        rng: np.random.Generator,
        fraction: float = 0.25,
        magnitude: float = 15.0,
    ) -> int:
        """在位置观测流中随机注入大幅离群值（就地修改），返回注入数量。"""
        pos_obs = [o for o in data.obs if o.kind == "position"]
        n = max(1, int(round(fraction * len(pos_obs))))
        chosen = rng.choice(len(pos_obs), size=n, replace=False)
        for i in chosen:
            o = pos_obs[i]
            direction = rng.uniform(0.0, 2.0 * np.pi)
            o.z = o.z + magnitude * np.array([np.cos(direction),
                                              np.sin(direction)])
            o.outlier = True
        return n

    @staticmethod
    def _copy(s: TruthState) -> TruthState:
        return TruthState(**s.__dict__)

    # ------------------------------------------------------------------
    # 标准测试场景
    # ------------------------------------------------------------------
    def stationary(self, duration: float = 20.0,
                   rng: np.random.Generator | None = None) -> ScenarioData:
        rng = rng or self.rng
        seg = _Segment(
            duration=duration,
            accel_body_fn=lambda tau, s: np.zeros(2),
            gyro_fn=lambda tau, s: 0.0,
        )
        return self.simulate("静止", [seg])

    def constant_velocity(self, duration: float = 20.0,
                          speed: float = 2.0) -> ScenarioData:
        """沿 x 方向匀速直线运动（真值加速度/角速度均为 0）。"""
        init = TruthState(t=0.0, px=0, py=0,
                          vx=speed, vy=0.0, theta=0.0)
        seg = _Segment(
            duration=duration,
            accel_body_fn=lambda tau, s: np.zeros(2),
            gyro_fn=lambda tau, s: 0.0,
        )
        return self.simulate("匀速直线", [seg], initial=init)

    def turning(self, duration: float = 24.0, speed: float = 1.5,
                yaw_rate: float = 0.35) -> ScenarioData:
        """协调转弯：切向速度不变，机体系侧向加速度 = v*yaw_rate。

        初始沿 x 方向运动；真值始终保持速率 speed。
        """
        init = TruthState(t=0.0, px=0, py=0,
                          vx=speed, vy=0.0, theta=0.0)
        # 机体系 y 轴（左侧）提供向心加速度；theta 以 yaw_rate 增长
        seg = _Segment(
            duration=duration,
            accel_body_fn=lambda tau, s: np.array([0.0, speed * yaw_rate]),
            gyro_fn=lambda tau, s: yaw_rate,
        )
        return self.simulate("匀速转弯", [seg], initial=init)

    def bias_drift(self, duration: float = 25.0,
                   ba=(0.15, -0.10), bg: float = 0.03) -> ScenarioData:
        """含恒定 IMU bias 的匀速直线运动：无观测时积分会漂移，
        EKF 应借助观测将 bias 估计出来。"""
        init = TruthState(t=0.0, px=0, py=0, vx=1.0, vy=0.0, theta=0.0)
        seg = _Segment(
            duration=duration,
            accel_body_fn=lambda tau, s: np.zeros(2),
            gyro_fn=lambda tau, s: 0.0,
            ba=tuple(ba), bg=bg,
        )
        return self.simulate("IMU bias", [seg], initial=init)

    def angle_wrap(self, start_heading: float = np.pi - 0.4,
                   yaw_rate: float = 0.5, duration: float = 8.0
                   ) -> ScenarioData:
        """航向从 +pi 附近开始持续旋转，多次跨越 ±pi 边界。"""
        init = TruthState(t=0.0, px=0, py=0, vx=0.0, vy=0.0,
                          theta=start_heading)
        seg = _Segment(
            duration=duration,
            accel_body_fn=lambda tau, s: np.zeros(2),
            gyro_fn=lambda tau, s: yaw_rate,
        )
        return self.simulate("航向跨±π", [seg], initial=init)

    def accel_straight(self, duration: float = 20.0,
                       accel: float = 0.4) -> ScenarioData:
        """沿机体系 x 轴匀加速直线，用于覆盖非零加速度通道。"""
        seg = _Segment(
            duration=duration,
            accel_body_fn=lambda tau, s: np.array([accel, 0.0]),
            gyro_fn=lambda tau, s: 0.0,
        )
        return self.simulate("匀加速直线", [seg])

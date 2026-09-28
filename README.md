# imu_ekf2d — 本地二维 IMU 姿态与运动状态 EKF

一个**纯本地**的二维扩展卡尔曼滤波器：用 IMU（加速度计 + 单轴陀螺）做状态预测，
用位置 / 速度 / 航向观测做更新，估计位置、速度、航向与 IMU 零偏。

* **无外部依赖**：只依赖 Python ≥ 3.9 与 `numpy`。
* **不接入任何服务**：不依赖 ROS、GPS 服务、硬件设备、云平台或网络；
  所有传感器样本、滤波状态、协方差、测试轨迹都只在**内存或本地文件**中。
* 测试轨迹全部由代码生成（含带噪 IMU 与观测），终端运行并输出估计误差。

## 目录结构

```
.
├── imu_ekf2d/
│   ├── __init__.py     # 公共 API
│   ├── filter.py       # EKF2D：状态、预测、更新、门限、协方差校验、保存/恢复
│   ├── simulator.py    # 代码生成真实轨迹 + 带噪 IMU/观测
│   ├── runner.py       # 按时间驱动滤波器 + RMSE 等误差统计
│   └── scenarios.py    # 7 个标准测试场景
├── tests/
│   └── test_ekf.py     # 自动化测试（33 个用例，终端打印误差）
├── examples/
│   └── basic_example.py
├── requirements.txt
└── README.md
```

## 状态定义

8 维状态向量

```
x = [ px, py, vx, vy, yaw, bax, bay, bg ]ᵀ
```

| 分量        | 含义                                   | 单位      |
|-------------|----------------------------------------|-----------|
| `px, py`    | 世界系二维位置                         | m         |
| `vx, vy`    | 世界系二维速度                         | m/s       |
| `yaw`       | 航向角，**恒规范化到 `(-π, π]`**       | rad       |
| `bax, bay`  | 机体系加速度计零偏（随机游走）         | m/s²      |
| `bg`        | 机体系 z 轴陀螺零偏（随机游走）         | rad/s     |

IMU 输入为机体系样本 `u = [ax, ay, wz]`。

## 运动模型（prediction）

去偏后的机体系加速度经航向旋转矩阵 `R(yaw)` 变换到世界系：

```
a_w = R(yaw) · ([ax, ay]ᵀ − [bax, bay]ᵀ)
ω   = wz − bg
```

连续时间状态传播：

```
ṗ = v
v̇ = a_w
ẏ = ω
ḃ_a = 0  (建模为随机游走)
ḃ_g = 0  (建模为随机游走)
```

离散步长 `dt`（在每个 IMU 样本上）的闭式状态传播：

```
pₖ₊₁ = pₖ + vₖ·dt + ½·a_w·dt²
vₖ₊₁ = vₖ + a_w·dt
yₖ₊₁ = wrap(yₖ + ω·dt)
bₖ₊₁ = bₖ
```

EKF 协方差预测 `Pₖ₊₁ = F P Fᵀ + Q`，其中 `F` 是状态方程对 8 维状态的
解析雅可比（对 `yaw` 与 bias 的耦合都有体现），关键子块：

```
∂a_w/∂yaw = J R(yaw) (a_meas − b_a),   J = [[0,−1],[1,0]]
∂p/∂v = dt·I,  ∂p/∂yaw = ½dt² ∂a_w/∂yaw,  ∂p/∂b_a = −½dt² R(yaw)
∂v/∂yaw = dt ∂a_w/∂yaw,                  ∂v/∂b_a = −dt R(yaw)
∂yaw/∂bg = −dt
```

过程噪声 `Q = L N Lᵀ`，噪声参数全部为**连续时间功率谱密度（PSD）**，
在 `NoiseConfig` 中配置：

| 参数           | 含义                         | 典型单位     |
|----------------|------------------------------|--------------|
| `acc_psd`      | 加速度计白噪声 PSD（每轴）   | m²/s³        |
| `gyro_psd`     | 陀螺白噪声 PSD               | rad²/s       |
| `acc_bias_psd` | 加计零偏随机游走 PSD（每轴） | (m/s²)²/s    |
| `gyro_bias_psd`| 陀螺零偏随机游走 PSD         | rad²/s³      |

离散化时白噪声通道按 `psd/dt`、随机游走通道按 `psd·dt` 处理；`L` 矩阵把
噪声通道同时映射到位置、速度、航向和 bias，保留了它们之间的交叉相关。

## 观测模型（update）

通用线性观测 `z = H x + v,  v ~ N(0, R)`，标准 EKF 更新：

```
innovation : ν = z − H x            （航向行会做角度规范化）
innovation 协方差 : S = H P Hᵀ + R
Kalman gain : K = P Hᵀ S⁻¹
状态更新   : x ← x + K ν
协方差更新 : P ← (I − KH) P (I − KH)ᵀ + K R Kᵀ    （Joseph 形式，保对称正定）
```

提供三个便捷观测接口：

* `update_position(z_xy, R_xy)` —— `z = [px, py]ᵀ`；
* `update_velocity(z_v, R_v)` —— 世界系 `z = [vx, vy]ᵀ`；
* `update_yaw(z_yaw, R_yaw)` —— 航向角标量观测。

也可直接调用 `update(z, H, R, angle_components=...)` 接入自定义线性观测。

### 航向角跨界处理

`yaw` 是角度量，预测与更新后都规范化到 `(-π, π]`；航向观测的 innovation
按圆周距离计算：

```
ν_yaw = wrap(z_yaw − yaw_pred)
```

因此估计在 `+π` 附近、观测在 `−π` 附近时，innovation 仍约等于真实的小角差，
不会错误跳变近 `2π`，也不会被 gating 误判为离群值。

### 异常观测 gating

更新前计算 Mahalanobis 距离

```
d² = νᵀ S⁻¹ ν
```

门限支持三种配置：

* `gate_probability=0.99`（默认）：按 χ²(m) 分布 99% 分位数设门限（内置
  常用自由度分位数表，其余自由度用 Wilson–Hilferty 近似）；
* `gate_sigma=k`：直接以标准差倍数 `k²` 为门限；
* `None`：关闭门限。

`d²` 超过门限的观测被**拒绝**，状态与协方差保持不变；`update` 返回
`UpdateResult(accepted, innovation, S, K, mahalanobis, gate_threshold, ...)`，
便于测试与诊断。

### 协方差合法性校验

以下情况立即抛出 `ValueError`：

* 协方差不是二维方阵、维度不匹配、含 NaN/Inf、不对称；
* 初始协方差 / `R` / 更新后 `P` / `S` 非正定（需要半正定的位置可用
  `validate_covariance(..., positive_semidefinite=True)` 显式放宽）；
* 噪声 PSD 为负或非有限；`dt <= 0`；`H` 形状错误；观测含 NaN。

## 保存与恢复

滤波器完整状态（状态向量、协方差、噪声配置）可序列化为 JSON 存到本地文件，
或在内存中以 `dict` 往返：

```python
ekf.save("ekf_state.json")
ekf2 = EKF2D.load("ekf_state.json")

snapshot = ekf.to_dict()      # 纯 JSON 数据，可自行存内存/文件
ekf3 = EKF2D.from_dict(snapshot)
```

恢复后的滤波器继续处理**相同的后续输入**，结果与一次性连续运行**逐位一致**
（滤波器本身不含随机性；测试中以 `atol=1e-12` 断言）。

## 快速上手

```python
import numpy as np
from imu_ekf2d import EKF2D, NoiseConfig

ekf = EKF2D(
    initial_state=np.zeros(8),
    initial_covariance=np.diag([1, 1, 1, 1, 0.25, 0.25, 0.25, 0.04]),
    noise_config=NoiseConfig(acc_psd=0.05, gyro_psd=1e-4),
)

# 100 Hz IMU 预测
ekf.predict(ax=0.02, ay=-0.01, wz=0.003, dt=0.01)

# 10 Hz 位置观测更新（R 为观测噪声协方差）
result = ekf.update_position([1.01, 0.98], np.diag([0.2**2, 0.2**2]))
if not result.accepted:
    print("观测被拒绝:", result.reason)

ekf.save("ekf_state.json")
```

更多用法见 `examples/basic_example.py`。

## 测试命令

只依赖 `numpy`：

```bash
pip install -r requirements.txt        # 若环境中没有 numpy

# 运行全部自动化测试（先打印 7 个场景的估计误差，再跑 33 个单元测试）
python3 tests/test_ekf.py

# 也可以用标准 unittest 入口
python3 -m unittest -v tests/test_ekf.py

# 运行示例（转弯轨迹 + 中途保存/恢复）
python3 examples/basic_example.py
```

测试脚本会在终端输出每个场景的估计误差（位置/速度/航向 RMSE、末端误差、
bias RMSE、门限接受/拒绝统计），任一断言失败时进程返回非零。

### 测试覆盖场景（轨迹与噪声均为代码生成，固定随机种子）

| # | 场景 | 覆盖要点 |
|---|------|----------|
| 1 | 静止 `static` | 静止稳定性、零偏/位置漂移 |
| 2 | 匀速直线 `constant velocity` | 初始位置/速度/航向有偏差时收敛 |
| 3 | 匀速转弯 `circular turn` | 机体系向心加速度 + 恒定角速度圆弧、动态精度 |
| 4 | IMU 零偏 `bias estimation` | 加计/陀螺常值零偏在直线+转弯激励下被估计 |
| 5 | 航向跨界 `yaw wrap` | 累计转角多次跨越 ±π，innovation 不跳变 |
| 6 | 异常观测 `gating` | 注入 4 个大幅位置/速度离群值，全部被拒绝且正常观测不被大面积误拒 |
| 7 | 保存/恢复 `save/restore` | 中途存盘→恢复→继续，与连续运行逐位一致 |

此外单元测试还覆盖：角度规范化、非法协方差（非对称/非正定/NaN/维度错误）、
非法噪声配置、`dt` 校验、`H/R` 维度错误、`S/K` 形状与正定性、门限开关、
航向跨界 innovation、确定性（重复运行完全一致）等。

## 设计说明与局限

* 平面运动模型：位置/速度为 2D，航向与角速度为单轴；
* 加速度计 bias 定义在**机体系**，随航向旋转耦合进世界系运动；
* bias 建模为随机游走（一阶高斯马尔可夫可在 `NoiseConfig`/状态扩展中加入）；
* 观测模型为线性（位置、速度直接是状态分量；航向按角度规范化处理），
  非线性仅存在于运动模型，因此属于 EKF；
* 直线运动中航向相对位置/速度不可观，需要航向观测或转弯激励，测试场景
  已据此配置。

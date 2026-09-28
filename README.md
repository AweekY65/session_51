# imu_ekf2d — 本地二维 IMU 姿态与运动状态 EKF

纯 Python（numpy + scipy）实现的**二维平面 IMU 扩展卡尔曼滤波器**，含代码生成轨迹的
自动化测试。所有传感器样本、滤波状态、协方差和测试轨迹**只保存在内存或本地文件中**，
不依赖 ROS、GPS 服务、任何硬件设备、云平台或外部网络服务。

## 目录结构

```
.
├── imu_ekf2d/
│   ├── __init__.py      # 包入口
│   ├── config.py        # EKFConfig：过程/测量噪声、Mahalanobis 门限配置
│   ├── math_utils.py    # 角度规范化 wrap_angle、协方差合法性校验
│   ├── ekf.py           # IMU2DEKF：predict / update / save / load 核心
│   ├── simulator.py     # TrajectorySimulator：代码生成真值轨迹与带噪 IMU/观测
│   └── runner.py        # 场景运行与 RMSE 误差统计
├── tests/
│   ├── conftest.py
│   └── test_ekf_scenarios.py   # 27 个 pytest 用例，终端输出估计误差
├── run_demo.py          # 终端演示：各场景误差 + gating 对照 + 状态恢复
└── requirements.txt
```

## 状态定义

8 维状态向量：

```
x = [ px, py, vx, vy, theta, ba_x, ba_y, b_g ]ᵀ
       └─位置─┘ └速度┘ 航向角  加速度bias(机体系) 陀螺bias
```

| 分量 | 含义 | 单位 |
|---|---|---|
| `px, py` | 世界系二维位置 | m |
| `vx, vy` | 世界系二维速度 | m/s |
| `theta` | 航向角（世界系 x 轴到机体 x 轴），规范化到 `[-π, π)` | rad |
| `ba_x, ba_y` | 加速度计 bias（机体坐标系，常数 + 随机游走） | m/s² |
| `b_g` | 陀螺仪 z 轴 bias（常数 + 随机游走） | rad/s |

协方差 `P` 为 8×8 对称（半）正定矩阵。

## 运动模型（prediction）

IMU 输入为机体系加速度 `a_m = [ax, ay]` 与绕 z 轴角速度 `gyro`，
旋转矩阵 `R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]`：

```
a_body = a_m − ba                      # 去 bias 的机体系加速度
a_world = R(θ) · a_body                # 旋转到世界系
ω = gyro − b_g

p ← p + v·Δt + ½·a_world·Δt²
v ← v + a_world·Δt
θ ← wrap(θ + ω·Δt)                     # 角度立即规范化
ba, b_g 保持不变（随机游走由过程噪声驱动）
```

**过程噪声**采用连续时间功率谱密度（PSD）模型，通过
Van Loan 离散化（矩阵指数，见 `ekf.py` 中 `_van_loan`）精确得到
离散状态转移阵 `Φ` 与离散过程噪声 `Qd`：

```
P ← Φ P Φᵀ + Qd
```

噪声通道：加速度白噪声（经姿态旋转进入世界系速度通道）、陀螺白噪声、
加速度 bias 随机游走、陀螺 bias 随机游走，强度均可在 `EKFConfig` 中配置。
预测后对 `P` 做对称化与正定校验。

## 观测模型（update）

支持三类观测，均为线性观测（EKF 中 H 为常数选择矩阵）：

| `kind` | 观测 z | H 选取 | 测量噪声 |
|---|---|---|---|
| `"position"` | `[zx, zy]` 位置 | `H·x = [px, py]` | `R_pos`（2×2） |
| `"velocity"` | `[vx, vy]` 速度 | `H·x = [vx, vy]` | `R_vel`（2×2） |
| `"heading"` | 标量航向 | `H·x = θ` | `R_heading`（标量） |

标准 EKF 更新（每次更新返回 `UpdateResult`，含全部中间量）：

```
y = z − H·x                  # innovation；航向观测先 wrap 到 [-π, π)
S = H P Hᵀ + R               # 新息协方差
K = P Hᵀ S⁻¹                 # Kalman 增益
x ← x + K·y                  # 更新后 θ 再次规范化
P ← (I−KH) P (I−KH)ᵀ + K R Kᵀ   # Joseph 形式，保证对称（半）正定
```

### 航向角跨界处理

航向新息计算后立即 `wrap_angle` 规范化到 `[-π, π)`。例如预测航向 `+3.05`、
观测 `-3.05`（物理上相差仅约 0.18 rad），innovation 为 `+0.183` 而非 `-6.1`，
跨越 ±π 不会产生 2π 量级的错误跳变。

### 异常观测 gating

按 Mahalanobis 距离拒绝离群观测：

```
d_M = sqrt(yᵀ S⁻¹ y)
d_M > gate_threshold  ⇒ 拒绝（默认阈值 3.0）
```

被拒绝时状态与协方差不变，结果记录在 `UpdateResult.accepted=False`；
`strict=True` 时抛出 `EKFGateReject`。阈值可配置（可由卡方分位数
`sqrt(chi2.ppf(p, df))` 换算）。

## 非法输入检测

`math_utils.validate_covariance` 与 `EKFConfig` 在初始化、每次 predict/update
后检查：

- 维度错误（状态非 8 维、`P` 非方阵或维度不符、`R` 与观测维度不符）；
- 含 `NaN` / `Inf`；
- 非对称（相对容差）；
- 非（半）正定（对称化后最小特征值为负）；
- 噪声 PSD 为负、门限非正、`dt` 非正等。

任一非法情况抛出 `EKFError`。

## 状态保存 / 恢复

`ekf.save(path)` / `IMU2DEKF.load(path)` 将完整状态（`x`、`P`、时间戳 `t`
及全部噪声配置）序列化为**本地 JSON 文件**；也可用 `to_dict()` /
`from_dict()` 在内存中传递。恢复后继续 predict/update 的结果与不间断连续运行
**逐位一致**（测试断言最大差 `0`）。

## 测试命令

需要 Python 3.10+、numpy、scipy、pytest（见 `requirements.txt`）：

```bash
pip install -r requirements.txt

# 运行全部 27 个自动化测试（-s 打印各场景估计误差）
python3 -m pytest tests/ -s -v

# 终端演示：误差表 + gating 开/关对照 + 保存恢复一致性
python3 run_demo.py
```

测试用 `TrajectorySimulator` 代码生成真值轨迹（固定随机种子，可复现），
在真值 IMU 上叠加 bias 与高斯白噪声，按周期生成带噪位置/速度/航向观测，
覆盖场景：

1. **静止** — 无漂移，位置/速度 RMSE 在观测噪声量级；
2. **匀速直线** — 速度与位置跟踪；
3. **匀加速直线** — 非零加速度通道；
4. **匀速转弯** — 协调转弯，向心加速度 + 持续航向变化；
5. **IMU bias** — 恒定加表/陀螺 bias 由观测在线标定（含纯航位推算漂移对照）；
6. **异常观测 gating** — 注入 20 m 离群位置观测，验证全部被 Mahalanobis
   门限拒绝、正常观测全部接受，并与关闭门限的对照组比较；
7. **航向跨 ±π** — 从 π 附近开始持续旋转，验证 innovation 无 2π 跳变；
8. **状态保存/恢复** — JSON 快照后续跑与连续运行逐位一致（含纯预测分支）；
9. **非法 covariance / 维度 / 参数** — 非对称、负定、错维、NaN、错 dt、错 R 等。

典型终端输出（`run_demo.py`，固定种子）：

```
场景           位置RMSE(m)  速度RMSE(m/s)  航向RMSE(deg)
静止              0.207        0.129          3.76
匀速直线           0.176        0.118          2.53
匀加速直线          0.268        0.124          2.80
匀速转弯           0.213        0.131          2.93
IMU bias         0.223        0.123          2.31
航向跨±π           0.303        0.144          3.68
异常观测+门限        0.307        0.128          3.04   (12/12 离群值被拒绝)
对照（关闭门限）       3.385 m
状态恢复 vs 连续运行：状态/协方差最大差 = 0.00e+00
```

## 最小使用示例

```python
import numpy as np
from imu_ekf2d import EKFConfig, IMU2DEKF

config = EKFConfig(
    acc_noise_psd=0.05,      # 加速度白噪声 PSD
    gyro_noise_psd=0.01,     # 陀螺白噪声 PSD
    R_pos=np.diag([0.25, 0.25]),
    gate_threshold=3.0,
)
ekf = IMU2DEKF(config=config,
               initial_state=np.zeros(8))

for ax_m, gyro_m in imu_stream:
    ekf.predict(dt=0.05, accel_body=np.array([ax_m, 0.0]), gyro=gyro_m)
    r = ekf.update("position", z_measured)      # 或 "velocity" / "heading"
    if not r.accepted:
        ...                                     # 观测被门限拒绝

ekf.save("filter_state.json")
ekf2 = IMU2DEKF.load("filter_state.json")       # 本地恢复
```

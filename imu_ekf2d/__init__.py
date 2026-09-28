"""imu_ekf2d: 本地二维 IMU 姿态与运动状态扩展卡尔曼滤波器.

纯 numpy/scipy 实现，不依赖 ROS / GPS / 硬件 / 云服务。
所有传感器样本、状态、协方差均保存在内存或本地文件中。
"""

from .config import EKFConfig
from .ekf import IMU2DEKF, EKFError, EKFGateReject
from .math_utils import wrap_angle, normalize_angle, validate_covariance
from .simulator import TrajectorySimulator, TruthState

__all__ = [
    "EKFConfig",
    "IMU2DEKF",
    "EKFError",
    "EKFGateReject",
    "wrap_angle",
    "normalize_angle",
    "validate_covariance",
    "TrajectorySimulator",
    "TruthState",
]

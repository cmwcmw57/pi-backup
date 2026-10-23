# ================================================================
# joint_convention.py
# 설명: 펌웨어(KSRC) 규약의 바퀴별 (조향각 θ, 바퀴 속도)와 시뮬 조인트 규약(steering_sign, rolling_heading_rad)을 같은 물리 속도 벡터로 서로 바꾼다. RL 정책은 펌웨어 규약으로 학습·출력하고, 시뮬 조인트는 URDF 규약으로 움직이기 때문에 필요하다.
# 사용법:
#   q, wheel_rad_s = to_joint(module, theta_fw, speed_mps, wheel_radius_m)     # 목표: 펌웨어 → 조인트
#   theta_fw, wheel_fw_rad_s = from_joint(module, q, wheel_rad_s)               # 측정: 조인트 → 펌웨어
# ================================================================
import math

from experiments.orbslam3_ros2.swerve import ModuleGeometry, fold_steering_angle


def to_joint(module: ModuleGeometry, theta_fw: float, speed_mps: float, wheel_radius_m: float) -> tuple[float, float]:
    """펌웨어 목표(조향각 θ, 부호 있는 속도 [m/s])를 같은 물리 속도 벡터를 만드는 시뮬 (조향 조인트 [rad], 바퀴 조인트 [rad/s])로 바꾼다."""
    heading = theta_fw+(math.pi if speed_mps < 0 else 0.)
    relative, sign = fold_steering_angle(heading-module.rolling_heading_rad)
    return module.steering_sign*relative, sign*abs(speed_mps)/wheel_radius_m


def from_joint(module: ModuleGeometry, steering_rad: float, wheel_rad_s: float) -> tuple[float, float]:
    """시뮬 조인트 측정값을 펌웨어 규약 (θ in ±90°, 바퀴 속도 [rad/s], 접은 방향 기준 부호)로 바꾼다."""
    heading = module.rolling_heading_rad+module.steering_sign*steering_rad
    theta = math.atan2(math.sin(heading), math.cos(heading))
    sign = 1.
    if theta > math.pi/2:
        theta, sign = theta-math.pi, -1.
    elif theta < -math.pi/2:
        theta, sign = theta+math.pi, -1.
    return theta, sign*wheel_rad_s

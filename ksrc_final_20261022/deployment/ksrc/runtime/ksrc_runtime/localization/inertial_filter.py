# ================================================================
# inertial_filter.py
# 설명: IMU 중심의 15차 error-state와 이전 pose clone으로 상대 영상 관측을 융합한다.
# 사용법: from ksrc_runtime.localization.inertial_filter import InertialFilter
# ================================================================
import numpy as np
from scipy.spatial.transform import Rotation


def skew(vector: np.ndarray) -> np.ndarray:
    """3D vector의 cross-product 행렬이다."""
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def perturb(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """world translation·body 우측 rotation 오차를 pose에 적용한다."""
    result = pose.copy()
    result[:3, 3] += delta[:3]
    result[:3, :3] = result[:3, :3] @ Rotation.from_rotvec(delta[3:]).as_matrix()
    return result


class InertialFilter:
    """p,v,body rotation,gyro bias,accel bias와 6차 pose clone의 교차 covariance를 유지한다."""

    def __init__(self, pose: np.ndarray, noise: dict, initial_covariance: np.ndarray | None = None) -> None:
        self.pose = pose.copy()
        self.velocity = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.noise = noise
        # Synthetic test assumptions, not a claim of calibrated hardware uncertainty.
        sigma = np.array([.03]*3+[.1]*3+[np.deg2rad(2)]*3+[.005]*3+[.05]*3)
        self.covariance = np.diag(sigma**2) if initial_covariance is None else initial_covariance.copy()
        if self.covariance.shape != (15, 15) or not np.isfinite(self.covariance).all():
            raise ValueError('initial covariance must be a finite 15x15 matrix')
        self.clone = pose.copy()
        self.gap_s = 0.      # 현재 이어지는 IMU 공백의 누적 길이
        self.degraded = False  # 공백 뒤 영상 보정이 한 번 받아들여질 때까지 True
        self.align_gravity = False  # 공백 뒤 첫 IMU 표본에서 기울기를 중력으로 다시 맞춘다
        self.clone_current()

    @property
    def status(self) -> str:
        """'imu_gap'(공백 중), 'recovering'(공백 뒤 영상 보정 대기), 'ok'를 반환한다."""
        return 'imu_gap' if self.gap_s > 0 else 'recovering' if self.degraded else 'ok'

    def imu_gap(self, duration: float, max_speed: float = .3, max_yaw_rate: float = 1.,
                twist: tuple[float, float, float] | None = None) -> None:
        """IMU가 없는 구간은 적분하지 않고 속도를 0으로 둔 채 위치·속도·자세 불확실성만 키운다.

        누적 공백 길이 T에 대해 위치 σ=max_speed·T, 속도 σ=max_speed, 자세 σ=min(max_yaw_rate·T, π/2)가 되도록
        호출마다 차이만 더한다. 영상 상대 관측이 이 불확실성 안에서 상태를 다시 당겨 온다."""
        if not duration > 0 or not np.isfinite(duration):
            raise ValueError('IMU gap duration must be positive and finite')
        first = self.gap_s == 0
        previous, self.gap_s = self.gap_s, self.gap_s+duration
        self.degraded = self.align_gravity = True
        if twist is not None:
            # 엔코더 body twist(vx, vy, yaw rate, 슬립 보정 후)로 평면 추측 항법을 하고 불확실성 증가율을 낮춘다.
            vx, vy, wz = twist
            planar = self.pose[:3, :3] @ np.array([vx, vy, 0.])
            self.pose[:3, 3] += planar*duration
            self.pose[:3, :3] = self.pose[:3, :3] @ Rotation.from_rotvec([0., 0., wz*duration]).as_matrix()
            self.velocity[:] = self.pose[:3, :3] @ np.array([vx, vy, 0.])
            max_speed, max_yaw_rate = .05+.3*float(np.hypot(vx, vy)), .15+.3*abs(wz)
            if first:
                self.covariance[3:6, :] = 0.
                self.covariance[:, 3:6] = 0.
                self.covariance[3:6, 3:6] = np.eye(3)*max_speed**2
        elif first:
            self.velocity[:] = 0.
            self.covariance[3:6, :] = 0.
            self.covariance[:, 3:6] = 0.
            self.covariance[3:6, 3:6] = np.eye(3)*max_speed**2
        grow_p = (max_speed*self.gap_s)**2-(max_speed*previous)**2
        grow_r = min(max_yaw_rate*self.gap_s, np.pi/2)**2-min(max_yaw_rate*previous, np.pi/2)**2
        self.covariance[:3, :3] += np.eye(3)*grow_p
        self.covariance[6:9, 6:9] += np.eye(3)*grow_r
        self.covariance = (self.covariance+self.covariance.T)/2

    def _align_tilt_to_gravity(self, force: np.ndarray) -> None:
        """공백 중 바뀐 roll·pitch를 비력 방향(저가속 가정)으로 다시 잡고 yaw는 유지하며 해당 불확실성을 낮춘다."""
        self.align_gravity = False
        yaw = Rotation.from_matrix(self.pose[:3, :3]).as_euler('xyz')[2]
        roll, pitch = np.arctan2(force[1], force[2]), np.arctan2(-force[0], np.linalg.norm(force[1:]))
        self.pose[:3, :3] = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        yaw_variance = max(self.covariance[8, 8], np.deg2rad(10.)**2)
        self.covariance[6:9, :] = 0.
        self.covariance[:, 6:9] = 0.
        tilt = np.deg2rad(3.)**2
        self.covariance[6:9, 6:9] = np.diag([tilt, tilt, yaw_variance])

    def clone_current(self) -> None:
        """이전 clone을 주변화하고 현재 pose의 stochastic copy와 cross-covariance를 저장한다."""
        self.clone = self.pose.copy()
        jac = np.zeros((21, 15))
        jac[:15] = np.eye(15)
        jac[15:18, :3] = np.eye(3)
        jac[18:21, 6:9] = np.eye(3)
        self.covariance = jac @ self.covariance[:15, :15] @ jac.T

    def propagate(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """표본 평균 IMU로 nominal state와 bias random walk covariance를 전파한다."""
        if not 0 < dt <= .1:
            raise ValueError('IMU integration step must be in (0,0.1] seconds')
        self.gap_s = 0.  # 정상 IMU가 다시 들어왔다. 영상 보정이 받아들여질 때까지 degraded는 유지한다.
        if self.align_gravity:
            self._align_tilt_to_gravity(accel-self.accel_bias)
        omega, force = gyro-self.gyro_bias, accel-self.accel_bias
        rotation = self.pose[:3, :3]
        mid = rotation @ Rotation.from_rotvec(omega*dt/2).as_matrix()
        acceleration = mid @ force + np.array([0., 0., -9.81])
        self.pose[:3, 3] += self.velocity*dt+.5*acceleration*dt**2
        self.velocity += acceleration*dt
        self.pose[:3, :3] = rotation @ Rotation.from_rotvec(omega*dt).as_matrix()
        dynamics = np.zeros((21, 21))
        dynamics[:3, 3:6] = np.eye(3)
        dynamics[3:6, 6:9] = -mid @ skew(force)
        dynamics[3:6, 12:15] = -mid
        dynamics[6:9, 6:9] = -skew(omega)
        dynamics[6:9, 9:12] = -np.eye(3)
        transition = np.eye(21)+dynamics*dt+.5*dynamics@dynamics*dt**2
        process = np.zeros((21, 21))
        qa = self.noise['IMU.NoiseAcc']**2
        for start, name in ((3, 'IMU.NoiseAcc'), (6, 'IMU.NoiseGyro'),
                            (9, 'IMU.GyroWalk'), (12, 'IMU.AccWalk')):
            process[start:start+3, start:start+3] = np.eye(3)*self.noise[name]**2*dt
        process[:3, :3] = np.eye(3)*qa*dt**3/3
        process[:3, 3:6] = process[3:6, :3] = np.eye(3)*qa*dt**2/2
        self.covariance = transition @ self.covariance @ transition.T+process
        self.covariance = (self.covariance+self.covariance.T)/2

    def update(self, residual: np.ndarray, jac: np.ndarray, noise: np.ndarray, gate: float,
               gain_projection: np.ndarray | None = None) -> dict:
        """Innovation gate와 Joseph covariance로 관측을 적용하고 attitude 오차 좌표계를 재설정한다."""
        innovation = jac @ self.covariance @ jac.T+noise
        nis = float(residual @ np.linalg.solve(innovation, residual))
        info = {'accepted': nis <= gate, 'nis': nis, 'gate': gate}
        if not info['accepted']:
            return info
        gain = np.linalg.solve(innovation, jac @ self.covariance).T
        if gain_projection is not None:
            gain = gain_projection @ gain
        delta = gain @ residual
        self.pose = perturb(self.pose, np.r_[delta[:3], delta[6:9]])
        self.velocity += delta[3:6]
        self.gyro_bias += delta[9:12]
        self.accel_bias += delta[12:15]
        self.clone = perturb(self.clone, delta[15:21])
        identity = np.eye(21)-gain @ jac
        covariance = identity @ self.covariance @ identity.T+gain @ noise @ gain.T
        reset = np.eye(21)
        reset[6:9, 6:9] -= .5*skew(delta[6:9])
        reset[18:21, 18:21] -= .5*skew(delta[18:21])
        self.covariance = reset @ covariance @ reset.T
        self.covariance = (self.covariance+self.covariance.T)/2
        return info

    def velocity_update(self, body_velocity_xy: np.ndarray, sigma: float, gate: float = 9.21) -> dict:
        """body frame 평면 속도(엔코더 등) 관측을 융합한다. gate 기본값은 chi-square(2) 99%다."""
        def residual(pose: np.ndarray, velocity: np.ndarray) -> np.ndarray:
            return np.asarray(body_velocity_xy, float)-(pose[:3, :3].T @ velocity)[:2]
        value = residual(self.pose, self.velocity)
        jac = np.zeros((2, 21))
        for axis in range(3):
            step = np.eye(3)[axis]*1e-6
            jac[:, 3+axis] = -(residual(self.pose, self.velocity+step)-value)/1e-6
            jac[:, 6+axis] = -(residual(perturb(self.pose, np.r_[np.zeros(3), step]), self.velocity)-value)/1e-6
        return self.update(value, jac, np.eye(2)*sigma**2, gate)

    def direction_update(self, direction_xy: np.ndarray, sigma: float, gate: float = 6.635) -> dict:
        """IMU 축 평면 속도의 방향만 관측한다(크기는 믿지 않는다). 관측 방향에 수직인 속도 성분이 0이어야 한다는 스칼라 관측이며
        gate 기본값은 chi-square(1) 99%다. direction_xy는 IMU 축에서 본 이동 방향이다."""
        direction = np.asarray(direction_xy, float).reshape(-1)
        norm = float(np.linalg.norm(direction)) if direction.shape == (2,) and np.isfinite(direction).all() else 0.
        if norm < 1e-9:
            raise ValueError('direction must be a finite non-zero 2D vector')
        normal = np.array([-direction[1], direction[0], 0.])/norm

        def residual(pose: np.ndarray, velocity: np.ndarray) -> np.ndarray:
            return np.array([-normal @ (pose[:3, :3].T @ velocity)])
        value = residual(self.pose, self.velocity)
        jac = np.zeros((1, 21))
        for axis in range(3):
            step = np.eye(3)[axis]*1e-6
            jac[:, 3+axis] = -(residual(self.pose, self.velocity+step)-value)/1e-6
            jac[:, 6+axis] = -(residual(perturb(self.pose, np.r_[np.zeros(3), step]), self.velocity)-value)/1e-6
        return self.update(value, jac, np.array([[sigma**2]]), gate)

    def attitude_update(self, observed_rotation: np.ndarray, sigma_rad: float, gate: float = 11.345) -> dict:
        """world 기준 절대 자세(3차) 관측을 융합한다. gate 기본값은 chi-square(3) 99%다."""
        def residual(pose: np.ndarray) -> np.ndarray:
            return Rotation.from_matrix(pose[:3, :3].T @ observed_rotation).as_rotvec()
        value = residual(self.pose)
        jac = np.zeros((3, 21))
        for component in range(3):
            delta = np.eye(6)[3+component]*1e-6
            jac[:, 6+component] = -(residual(perturb(self.pose, delta))-value)/1e-6
        return self.update(value, jac, np.eye(3)*sigma_rad**2, gate)

    def relative_update(self, observed: np.ndarray, position_scale: float = 1., rotation_scale: float = 1.) -> dict:
        """ORB 상대 IMU pose를 융합한다. 절대 SLAM pose를 독립 절대 관측으로 반복 사용하지 않는다.

        position_scale·rotation_scale은 관측 표준편차 배율이다(경사·회전처럼 영상을 덜 믿을 때 1보다 크게).
        """
        def residual(pose: np.ndarray, clone: np.ndarray) -> np.ndarray:
            predicted = np.linalg.inv(clone) @ pose
            return np.r_[observed[:3, 3]-predicted[:3, 3],
                         Rotation.from_matrix(predicted[:3, :3].T @ observed[:3, :3]).as_rotvec()]
        value = residual(self.pose, self.clone)
        jac = np.zeros((6, 21))
        for component, column in enumerate((0, 1, 2, 6, 7, 8)):
            delta = np.eye(6)[component]*1e-6
            jac[:, column] = -(residual(perturb(self.pose, delta), self.clone)-value)/1e-6
            jac[:, 15+component] = -(residual(self.pose, perturb(self.clone, delta))-value)/1e-6
        # Fixed metric/angle uncertainty assumptions; ORB does not export a calibrated pose covariance.
        noise = np.diag(np.array([.01*position_scale]*3+[np.deg2rad(1.)*rotation_scale]*3)**2)
        info = self.update(value, jac, noise, 16.812)  # chi-square(6), 99 percent.
        info['residual'] = value.tolist()  # [상대 위치 차(m) 3, 상대 회전 차(rad) 3]
        if info['accepted']:
            self.clone_current()
            if self.gap_s == 0:
                self.degraded = False
        return info

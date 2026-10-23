# ================================================================
# rl_residual.py
# 설명: 시뮬 로버의 swerve 목표 아래에 yunbeen(KSRC) Res8 student 정책의 잔차를 얹는다. 실기 Pi의 run_policy.py 루프(관측 조립 → 정책 → 정지 명령 잔차 0 → 잔차 → 펌웨어 C IK)를 같은 코드(student_runtime)로 재현한다. 기본은 꺼짐이다.
# 사용법:
#   rl = RlResidual(ksrc_root, "pi/policy/res8_mlp_v1_rc4000")        # 50 Hz 제어(control_dt 0.02)에서만 쓴다
#   out = rl.step(command=(vx, vy, w), state=dict(gyro_body, force_body, steer_pos, steer_vel, wheel_vel))
#   out["steer_goal"], out["wheel_mps"]   # FL, FR, RL, RR 순서의 조향 목표 [rad], 바퀴 속도 목표 [m/s]
# ================================================================
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np

G = 9.81
STAND_V, STAND_W = 0.01, 0.02          # run_policy.py와 같은 값: 이하면 정지 명령으로 보고 잔차 0


def load_student_runtime(ksrc_root: Path):
    """ksrc 체크아웃의 pi/policy/student_runtime.py를 경로로 불러온다(펌웨어 C IK는 sim/libksrc_common.so가 필요)."""
    path = Path(ksrc_root)/'pi'/'policy'/'student_runtime.py'
    if not path.is_file():
        raise FileNotFoundError(f'student_runtime.py not found under {ksrc_root}')
    if not (Path(ksrc_root)/'sim'/'libksrc_common.so').is_file():
        raise FileNotFoundError('sim/libksrc_common.so missing: run sim/build.sh in the ksrc checkout first')
    spec = importlib.util.spec_from_file_location('ksrc_student_runtime', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['ksrc_student_runtime'] = module
    spec.loader.exec_module(module)
    return module


class NumpyMlpPolicy:
    """정규화기 + Linear/ELU MLP를 numpy로 계산한다(torch 없는 Isaac 컨테이너용). 가중치는 export_policy_npz.py가 만든다."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        self.mean, self.std, self.eps, self.weights, self.biases = mean, std, eps, weights, biases

    @classmethod
    def load(cls, path: Path) -> "NumpyMlpPolicy":
        """export_policy_npz.py가 쓴 .npz를 읽는다."""
        data = np.load(path)
        count = sum(1 for k in data.files if k.startswith('W'))
        return cls(data['mean'].astype(np.float64), data['std'].astype(np.float64), float(data['eps']),
                   [data[f'W{i}'].astype(np.float64) for i in range(count)], [data[f'b{i}'].astype(np.float64) for i in range(count)])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """obs (1, obs_dim) 또는 (obs_dim,) → 행동 (8,)."""
        x = (np.asarray(obs, np.float64).reshape(-1)-self.mean)/(self.std+self.eps)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = w@x+b
            if i < len(self.weights)-1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.)))      # ELU(alpha=1)
        return x.astype(np.float32)


def load_policy(policy_path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """<p>.npz가 있으면 numpy로, 없으면 TorchScript <p>.pt로 정책을 읽는다. 메타는 <p>.json."""
    meta = json.loads(Path(str(policy_path)+'.json').read_text())
    if meta['action_dim'] != 8:
        raise ValueError(f"Res8 정책이 아님 (action_dim {meta['action_dim']})")
    if meta.get('recurrent'):
        raise ValueError('recurrent policy is not supported')
    npz = Path(str(policy_path)+'.npz')
    if npz.is_file():
        return NumpyMlpPolicy.load(npz), meta
    return load_torch_policy(policy_path)


def load_torch_policy(policy_path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """<p>.json 메타와 TorchScript <p>.pt를 읽어 obs (1, obs_dim) → 행동 (8,) 함수를 돌려준다. 순환 정책은 지원하지 않는다."""
    import torch
    meta = json.loads(Path(str(policy_path)+'.json').read_text())
    if meta['action_dim'] != 8:
        raise ValueError(f"Res8 정책이 아님 (action_dim {meta['action_dim']})")
    if meta.get('recurrent'):
        raise ValueError('recurrent policy is not supported')
    module = torch.jit.load(str(policy_path)+'.pt', map_location='cpu').eval()
    with torch.no_grad():
        for _ in range(3):                       # 첫 호출의 TorchScript 최적화를 루프 밖에서 끝낸다
            module(torch.zeros(1, int(meta['obs_dim'])))

    def policy(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return module(torch.as_tensor(obs, dtype=torch.float32))[0].numpy()
    return policy, meta


class RlResidual:
    """관측 이력과 IK 상태를 들고 제어 주기마다 (조향 목표, 바퀴 속도 목표)를 돌려준다."""

    def __init__(self, ksrc_root: Path, policy: str | Path | Callable[[np.ndarray], np.ndarray], *,
                 meta: dict | None = None, dt: float = .02, residual_on: bool = True,
                 imu_realism: object | None = None, bias_s: float = 2.) -> None:
        self.runtime = load_student_runtime(ksrc_root)
        if callable(policy):
            self.policy, self.meta = policy, meta
            if meta is None:
                raise ValueError('meta is required with a callable policy')
        else:
            path = Path(policy)
            path = path if path.is_absolute() else Path(ksrc_root)/path
            self.policy, self.meta = load_policy(path)
        meta_dt = float(self.meta.get('control_dt_s', .02))
        if not math.isclose(dt, meta_dt, rel_tol=1e-6):
            raise ValueError(f'control dt {dt} must equal the policy control_dt_s {meta_dt}')
        self.dt, self.residual_on = dt, residual_on
        self.rate = float(self.meta.get('res8_rate_limit', 0.))
        self.obs = self.runtime.StudentObs(self.meta, dt)
        self.ik = self.runtime.FirmwareIK()
        self.bias = np.zeros(3)                  # 부팅 때 정지해서 잰 자이로 평균(Pi run_policy의 --bias-s와 같은 방식)
        self.imu_realism, self.bias_s = imu_realism, float(bias_s)   # ImuRealism이면 관측용 gyro·비력에도 실기 특성을 더한다
        self._bias_samples, self._bias_done = [], False
        self.previous_action = np.zeros(8, np.float32)
        self.applied = self.runtime.applied_for_obs(*self.ik.step(0., 0., 0.), None)

    def step(self, command: tuple, state: dict, sim_s: float | None = None) -> dict:
        """command=(vx,vy,w) body 명령. state는 body 축 gyro [rad/s], 비력 [m/s²]과 FL,FR,RL,RR 조인트 값."""
        rt = self.runtime
        vx, vy, w = (float(c) for c in command)
        gyro_body, force_body = np.asarray(state['gyro_body'], float), np.asarray(state['force_body'], float)
        if self.imu_realism is not None:
            gyro_body, force_body = self.imu_realism.apply(self.dt, float(state.get('speed', 0.)), np.eye(4), gyro_body, force_body)
        if sim_s is not None and not self._bias_done:
            if sim_s < self.bias_s:
                self._bias_samples.append(gyro_body)        # 부팅 정지 구간: 평균만 모은다
            else:
                self.bias = np.mean(self._bias_samples, axis=0) if self._bias_samples else np.zeros(3)
                self._bias_done = True
        tlm = {'gyro_dps': np.degrees(gyro_body),
               'acc_g': force_body/G,
               'steer_meas_rad': np.asarray(state['steer_pos'], float),
               'steer_speed': np.asarray(state['steer_vel'], float)/rt.STEER_STEP_RAD,   # frame()이 다시 곱한다
               'wheel_mps': np.asarray(state['wheel_vel'], float)*rt.WHEEL_RADIUS_M}
        frame = self.obs.frame((vx, vy, w), tlm, self.bias, self.applied, vo=state.get('vo'))
        obs = self.obs.push(frame)
        obs_frame = np.concatenate([np.asarray(frame[name], float) for name, _ in self.obs.layout])   # 이번 주기 관측(항목 순서, 이력 제외)
        action = np.asarray(self.policy(obs), np.float32).reshape(-1)
        if abs(vx) < STAND_V and abs(vy) < STAND_V and abs(w) < STAND_W:
            action = np.zeros_like(action)
        if self.rate > 0:
            action = self.previous_action+np.clip(action-self.previous_action, -self.rate, self.rate)
        self.previous_action = action.copy()
        d_steer, gain = rt.action_to_residual(action)
        if not self.residual_on:
            d_steer, gain = np.zeros(4, np.float32), np.ones(4, np.float32)
        goal, speed = self.ik.step(vx, vy, w, d_steer, gain)
        self.applied = rt.applied_for_obs(goal, speed, None)
        return {'steer_goal': goal.astype(float), 'wheel_mps': speed.astype(float),
                'd_steer': np.asarray(d_steer, float), 'wheel_gain': np.asarray(gain, float), 'action': action.astype(float), 'obs_frame': obs_frame}

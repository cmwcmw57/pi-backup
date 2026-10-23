# ================================================================
# ksrc_actuation.py
# 설명: KSRC C 펌웨어 PI·정렬 게이트와 sim2sim 서보·모터 응답을 단일 로버에 적용한다.
# 사용법: FirmwareActuation(robot, session, rl.runtime.kc)
# 참고: YunBeeeen/ksrc @6023c2a sim/rl/sim2sim_isaac.py, utils/actuators_torch.py
# ================================================================
from __future__ import annotations

import ctypes as ct
import math

import numpy as np


class ServoProfile:
    """peer SteerServo의 5ms 지연·데드밴드·가속 프로파일을 NumPy로 계산한다."""

    def __init__(self) -> None:
        self.position = np.zeros(4, np.float32)
        self.velocity = np.zeros(4, np.float32)
        self.history = np.zeros((13, 4), np.float32)

    def step(self, goal: np.ndarray) -> np.ndarray:
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = goal
        delayed = self.history[7]
        error = delayed-self.position
        error = np.where(abs(error) <= math.radians(.25), 0., error)
        desired = np.sign(error)*np.minimum(np.sqrt(2*7.7*abs(error)), 5.45)
        self.velocity += np.clip(desired-self.velocity, -7.7*.005, 7.7*.005)
        position = self.position+self.velocity*.005
        crossed = (error != 0.) & ((delayed-position)*error <= 0.)
        self.position[:] = np.where(crossed, delayed, position)
        self.velocity[:] = np.where(crossed, 0., self.velocity)
        return self.position.copy()


class GateState(ct.Structure):
    """배포된 drive_align_gate.h의 C 상태 구조체와 같은 메모리 배치다."""

    _fields_ = [('trigger_rad', ct.c_float), ('release_rad', ct.c_float),
                ('prev_goal', ct.c_float*4), ('has_prev', ct.c_int), ('pending', ct.c_int),
                ('driving', ct.c_int), ('max_hold_ticks', ct.c_uint), ('hold_ticks', ct.c_uint),
                ('timed_out', ct.c_int)]


def motor_effort(duty: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """sim2sim --drive pi의 duty 모터·토크 envelope를 계산한다(추가 관절 마찰0)."""
    # MuJoCo 기본 wheel-friction-duty=0: 학습의 ground_load53을 임의로 더하지 않는다.
    tau = 1.96*np.asarray(duty)/799.-(1.96/7.96)*speed
    lower = np.maximum(-1.96, -1.96*(1.+speed/7.96))
    upper = np.minimum(1.96, 1.96*(1.-speed/7.96))
    return np.clip(tau, lower, upper)


class FirmwareActuation:
    """50Hz C PI·게이트와 200Hz 서보·DC 모터를 기존 joint 규약으로 연결한다."""

    def __init__(self, robot, session, common) -> None:
        self.robot, self.session, self.common = robot, session, common
        self.servo = ServoProfile()
        self.pi = [common.WheelPi() for _ in range(4)]
        self.gate = GateState()
        lib = common._lib
        floats = ct.POINTER(ct.c_float)
        lib.drive_gate_init.argtypes = [ct.POINTER(GateState), ct.c_float, ct.c_float, ct.c_uint]
        lib.drive_gate_init.restype = None
        lib.drive_gate_update_goal.argtypes = [ct.POINTER(GateState), floats, floats]
        lib.drive_gate_update_goal.restype = ct.c_int
        lib.drive_gate_allow.argtypes = [ct.POINTER(GateState), floats, floats, ct.c_int]
        lib.drive_gate_allow.restype = ct.c_int
        lib.drive_gate_init(ct.byref(self.gate), math.radians(60.), math.radians(10.), 75)
        self.goal, self.wheel_goal, self.duty, self.effort = (np.zeros(4) for _ in range(4))
        self.allowed = True
        self.previous_ticks = self._ticks()
        self.encoder_speed = np.zeros(4)

    def _ticks(self) -> np.ndarray:
        q = np.asarray(self.robot.get_joint_positions())[self.session.wheel_indices]
        return np.floor(q*8384/(2*math.pi))

    def read_encoder(self) -> np.ndarray:
        """현재 20ms 창의 속도를 읽되 PI 갱신 전까지 tick 기준은 유지한다."""
        return (self._ticks()-self.previous_ticks)*(2*math.pi/8384)/.02

    def command(self, steering: np.ndarray, wheel: np.ndarray) -> None:
        self.goal, self.wheel_goal = np.array(steering, float), np.array(wheel, float)
        actual = np.asarray(self.robot.get_joint_positions())[self.session.steering_indices]
        values = [np.ascontiguousarray(x, dtype=np.float32) for x in (steering, wheel, actual)]
        pointers = [x.ctypes.data_as(ct.POINTER(ct.c_float)) for x in values]
        self.common._lib.drive_gate_update_goal(ct.byref(self.gate), pointers[0], pointers[1])
        self.allowed = bool(self.common._lib.drive_gate_allow(ct.byref(self.gate), pointers[0], pointers[2], 1))
        ticks = self._ticks()
        self.encoder_speed = (ticks-self.previous_ticks)*(2*math.pi/8384)/.02
        self.previous_ticks = ticks
        reference = wheel if self.allowed else np.zeros(4)
        self.duty = np.array([p.step(float(ref), float(meas)) for p, ref, meas in zip(self.pi, reference, self.encoder_speed)])

    def apply(self) -> None:
        s = self.session
        profile = self.servo.step(self.goal)
        speed = np.asarray(self.robot.get_joint_velocities())[s.wheel_indices]
        self.effort = motor_effort(self.duty, speed)
        s.controller.apply_action(s._action_type(joint_positions=profile, joint_indices=s.steering_indices))
        s.controller.apply_action(s._action_type(joint_efforts=self.effort, joint_indices=s.wheel_indices))

    def stop(self) -> None:
        self.duty[:] = 0.;self.effort[:] = 0.
        for p in self.pi:p.reset()
        s = self.session
        s.controller.apply_action(s._action_type(joint_efforts=np.zeros(4), joint_indices=s.wheel_indices))

    def telemetry(self) -> dict:
        return dict(model='ksrc-pi', gate_allow=self.allowed, duty=self.duty.tolist(),
                    encoder_rad_s=self.encoder_speed.tolist(), servo_profile_rad=self.servo.position.tolist(),
                    wheel_effort_nm=self.effort.tolist())


def configure_joint_physics(stage, geometry) -> list[dict]:
    """물리 초기화 전에 peer 모터 관성과 조향 속도 상한을 USD에 설정한다."""
    from pxr import PhysxSchema, UsdPhysics
    steering = {m.steering_joint for m in geometry.modules}
    wheels = {m.wheel_joint for m in geometry.modules}
    changed = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if name not in steering|wheels or not prim.IsA(UsdPhysics.Joint):continue
        api = PhysxSchema.PhysxJointAPI.Apply(prim)
        if name in steering:api.CreateMaxJointVelocityAttr(math.degrees(5.45))
        if name in wheels:api.CreateArmatureAttr(.0098)
        changed.append(dict(path=str(prim.GetPath()), joint=name))
    if len(changed) != 8:raise RuntimeError(f'expected 8 KSRC joints, found {len(changed)}')
    return changed


def configure_drives(session) -> None:
    """조향 PD12/.3과 wheel effort 제어를 켜고 기존 wheel velocity drive를 해제한다."""
    controller = session.controller
    kp, kd = (np.asarray(a, float) for a in controller.get_gains())
    kp[session.steering_indices] = 12.;kd[session.steering_indices] = .3
    kp[session.wheel_indices] = 0.;kd[session.wheel_indices] = 0.
    controller.set_gains(kps=kp, kds=kd)
    for index in session.wheel_indices:controller.switch_dof_control_mode(int(index), 'effort')
    controller.set_max_efforts(np.full(4, 2.94), session.steering_indices)
    controller.set_max_efforts(np.full(4, 1.96), session.wheel_indices)

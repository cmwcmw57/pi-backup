"""Three rover action spaces sharing one eight-joint physical interface."""

from __future__ import annotations

import torch

from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from ksrc_rover.utils.actuators_torch import PWM_MAX, DriveGate, SteerServo, WheelPI, wheel_friction
from ksrc_rover.utils.contract import (
    MODULE_NAMES, RES8_STEER_SCALE_RAD, RES8_WHEEL_SCALE, RESIDUAL3_SCALE,
    RESIDUAL_RATE_LIMIT, STEER_LIMIT_RAD, WHEEL_MAX_RAD_S, WHEEL_RADIUS_M,
)
from ksrc_rover.utils.ik_torch import BatchedSwerveIK
from ksrc_rover.utils.sensors_torch import EncoderSpeed

FW_MAX_WHEEL_MPS = 0.378     # main.c SWERVE_MAX_WHEEL_MPS (open-loop duty scale)


class RoverAction(ActionTerm):
    """Map policy output to steering position and wheel velocity targets."""

    action_size = 8

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.steer_ids, steer_names = self._asset.find_joints(
            [f"{name}_steering_joint" for name in MODULE_NAMES], preserve_order=True
        )
        self.wheel_ids, wheel_names = self._asset.find_joints(
            [f"{name}_wheel_joint" for name in MODULE_NAMES], preserve_order=True
        )
        self.rocker_ids, rocker_names = self._asset.find_joints(
            ["left_rocker_joint", "right_rocker_joint"], preserve_order=True
        )
        if len(steer_names) != 4 or len(wheel_names) != 4 or len(rocker_names) != 2:
            raise ValueError(f"Unexpected rover joints: {steer_names}, {wheel_names}, {rocker_names}")
        self._raw = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed = torch.zeros(self.num_envs, 8, device=self.device)
        # STS3215 model (measured 2026-10-13), run every physics step in apply_actions:
        # goal -> dead time -> deadband -> trapezoid profile -> PD position target.
        # utils/actuators_torch.py, shared with sim2sim (MuJoCo); DR writes its per-env tensors.
        self.servo = SteerServo(self.num_envs, float(env.physics_dt), self.device)
        # Commanded joint targets before the servo rate limit -- the common physical
        # command of every action space (smoothness rewards read this).
        self._target = torch.zeros(self.num_envs, 8, device=self.device)
        self.ik = BatchedSwerveIK(self.num_envs, self.device)
        # Stage-2 DR hook (mdp/dr.py): a short history of joint targets so env._act_delay
        # can apply an older one.
        self._target_hist = torch.zeros(4, self.num_envs, 8, device=self.device)
        # Flat-rollout calibration against the MuJoCo hard differential:
        # mean |theta_L+theta_R| fell from 1.76 to 0.18 deg at K=20,D=0.3.
        self.rocker_k = 20.0
        self.rocker_d = 0.3
        # Wheel drive.  "isaac": the DCMotor actuator tracks the speed target itself (D*(w_t - w),
        # clipped to the torque-speed line).  "pi" (2026-10-13, WPI ON on the rover): the firmware
        # wheel PI (WheelPI = wheel_pi.c) runs once per control period on the encoder speed of the
        # last period and its duty is held for the 4 physics steps; the DC motor sees it as the
        # equivalent speed target duty/799 * w_noload (per motor, voltage DR included), and the
        # gearbox friction (stiction + Stribeck, bench 10-13) is added as an effort.
        # "openloop": the firmware's WPI OFF path (main.c: duty = v / 0.378 m/s * 799, truncated) through
        # the same duty motor + friction, so "pi" vs "openloop" differ only in the controller.
        self.drive = cfg.drive
        if self.drive not in ("isaac", "pi", "openloop"):
            raise ValueError(f"drive must be isaac, pi or openloop, got {self.drive}")
        self._ctrl_dt = float(env.step_dt)
        self.pi = WheelPI(self.num_envs, self.device, dt=self._ctrl_dt)
        self.enc = EncoderSpeed(self.num_envs, self._ctrl_dt, self.device)
        self.duty = torch.zeros(self.num_envs, 4, device=self.device)
        self.friction_scale = torch.ones(self.num_envs, 4, device=self.device)   # DR (mdp/dr.py)
        # Firmware drive-align gate (main.c Swerve_Control, 2026-10-13): after a big steering change the
        # wheels get 0 until every servo is within 10 deg of its goal.  Off = the pre-10-13 behaviour.
        self.gate_on = bool(cfg.drive_gate)
        self.gate = DriveGate(self.num_envs, self.device)
        self.gate_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self):
        return self.action_size

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def reset(self, env_ids=None):
        self._raw[env_ids] = 0.0
        self._processed[env_ids] = 0.0
        self.servo.reset(env_ids)
        self._target[env_ids] = 0.0
        self._target_hist[:, env_ids] = 0.0
        self.ik.reset(env_ids)
        self.pi.clear_fault(env_ids)      # a new episode = a freshly booted rover (WPI ON clears the latch)
        self.enc.reset(env_ids)
        self.gate.reset(env_ids)
        self.gate_hold[env_ids] = False
        self.duty[env_ids] = 0.0

    @property
    def commanded_targets(self):
        return self._target

    def _limit_steering(self):
        self._target[:] = self._processed
        # Command latency (Pi -> SPI -> next Nucleo tick): apply the target issued
        # env._act_delay policy steps ago. Smoothness rewards keep using _target.
        delay = getattr(self._env, "_act_delay", None)
        if delay is not None:
            self._target_hist = torch.roll(self._target_hist, 1, dims=0)
            self._target_hist[0] = self._processed
            d = delay.clamp(max=self._target_hist.shape[0] - 1)
            self._processed[:] = self._target_hist[d, torch.arange(self.num_envs, device=self.device)]
        # _processed[:, :4] is now the steering goal as the Nucleo writes it to the servos;
        # the servo's own dead time and profile run per physics step in apply_actions.
        if self.gate_on:             # Nucleo control tick: IK (+ residual) -> gate -> drive
            actual = self._asset.data.joint_pos[:, self.steer_ids] - self.servo.offset   # servo's own reading
            allow = self.gate.step(self._processed[:, :4], self._processed[:, 4:], actual)
            self.gate_hold[:] = ~allow
            self._processed[:, 4:] = torch.where(allow.unsqueeze(1), self._processed[:, 4:],
                                                 torch.zeros_like(self._processed[:, 4:]))
        if self.drive == "pi":       # Nucleo control tick: encoder speed of the last 20 ms -> PI -> duty
            w_enc = self.enc.step(self._asset.data.joint_pos[:, self.wheel_ids])
            self.duty[:] = self.pi.step(self._processed[:, 4:], w_enc)
        elif self.drive == "openloop":
            v = self._processed[:, 4:] * WHEEL_RADIUS_M
            self.duty[:] = torch.trunc((v / FW_MAX_WHEEL_MPS).clamp(-1.0, 1.0) * PWM_MAX)

    def apply_actions(self):
        self._asset.set_joint_position_target(self.servo.step(self._processed[:, :4]), joint_ids=self.steer_ids)
        if self.drive in ("pi", "openloop"):
            self._apply_duty()
        else:
            self._asset.set_joint_velocity_target(self._processed[:, 4:], joint_ids=self.wheel_ids)
        # Equal torque on both rockers constrains their sum to zero (1:1 differential).
        q = self._asset.data.joint_pos[:, self.rocker_ids]
        qd = self._asset.data.joint_vel[:, self.rocker_ids]
        tau = (-self.rocker_k * q.sum(dim=1, keepdim=True)
               - self.rocker_d * qd.sum(dim=1, keepdim=True)).clamp(-1.0, 1.0)
        self._asset.set_joint_effort_target(tau.expand(-1, 2), joint_ids=self.rocker_ids)


    def _apply_duty(self):
        act = self._asset.actuators["drive"]
        w_nl = torch.as_tensor(act.velocity_limit, device=self.device)        # (N,4) after drive DR
        v_eq = self.duty / PWM_MAX * w_nl
        # motor torque as the DCMotor actuator will compute it (needed for the stiction balance)
        sat = torch.as_tensor(act._saturation_effort, device=self.device)
        w = self._asset.data.joint_vel[:, self.wheel_ids]
        tau = torch.as_tensor(act.damping, device=self.device) * (v_eq - w)
        tau = torch.minimum(torch.maximum(tau, -sat * (1.0 + w / w_nl)), sat * (1.0 - w / w_nl))
        arm = self._asset.data.joint_armature[:, self.wheel_ids]
        f = wheel_friction(tau, w, arm, float(self._env.physics_dt), self.friction_scale)
        self._asset.set_joint_velocity_target(v_eq, joint_ids=self.wheel_ids)
        self._asset.set_joint_effort_target(f, joint_ids=self.wheel_ids)


class Residual3Action(RoverAction):
    """Body velocity residual followed by the shared firmware-style swerve IK."""

    action_size = 3

    def process_actions(self, actions):
        previous = self._raw.clone()
        self._raw[:] = torch.clamp(actions, -1.0, 1.0)
        limited = torch.clamp(self._raw - previous, -RESIDUAL_RATE_LIMIT, RESIDUAL_RATE_LIMIT) + previous
        self._raw[:] = limited
        nominal = self._env.command_manager.get_command("base_velocity")
        cmd = nominal + limited * torch.tensor(RESIDUAL3_SCALE, device=self.device)
        steer, speed_mps = self.ik.compute(cmd)
        self._processed[:] = torch.cat((steer, speed_mps / WHEEL_RADIUS_M), dim=1)
        self._limit_steering()


class Res8Action(RoverAction):
    """Per-wheel residual on the same nominal IK command."""

    def process_actions(self, actions):
        self._raw[:] = torch.clamp(actions, -1.0, 1.0)
        nominal = self._env.command_manager.get_command("base_velocity")
        steer, speed_mps = self.ik.compute(nominal)
        target_steer = (steer + self._raw[:, :4] * RES8_STEER_SCALE_RAD).clamp(
            -STEER_LIMIT_RAD, STEER_LIMIT_RAD
        )
        target_wheel = (speed_mps / WHEEL_RADIUS_M) * (1.0 + self._raw[:, 4:] * RES8_WHEEL_SCALE)
        self._processed[:] = torch.cat((target_steer, target_wheel.clamp(-WHEEL_MAX_RAD_S, WHEEL_MAX_RAD_S)), dim=1)
        self._limit_steering()


class Direct8Action(RoverAction):
    """Direct steering angle and wheel-speed targets without IK."""

    def process_actions(self, actions):
        self._raw[:] = torch.clamp(actions, -1.0, 1.0)
        self._processed[:, :4] = self._raw[:, :4] * STEER_LIMIT_RAD
        self._processed[:, 4:] = self._raw[:, 4:] * WHEEL_MAX_RAD_S
        self._limit_steering()


@configclass
class Residual3ActionCfg(ActionTermCfg):
    class_type: type = Residual3Action
    asset_name: str = "robot"
    drive: str = "isaac"          # "isaac" | "pi" (firmware wheel PI) | "openloop" (firmware duty), both + friction
    drive_gate: bool = False      # firmware drive-align gate (hold wheels during big steering changes)


@configclass
class Res8ActionCfg(ActionTermCfg):
    class_type: type = Res8Action
    asset_name: str = "robot"
    drive: str = "isaac"          # "isaac" | "pi" (firmware wheel PI) | "openloop" (firmware duty), both + friction
    drive_gate: bool = False      # firmware drive-align gate (hold wheels during big steering changes)


@configclass
class Direct8ActionCfg(ActionTermCfg):
    class_type: type = Direct8Action
    asset_name: str = "robot"
    drive: str = "isaac"          # "isaac" | "pi" (firmware wheel PI) | "openloop" (firmware duty), both + friction
    drive_gate: bool = False      # firmware drive-align gate (hold wheels during big steering changes)

"""가짜 Nucleo: 노트북에서 SPI 측정 도구(bringup.py 등)를 실물 없이 시험하기 위한 펌웨어 흉내.

바이트 수준에서 실물과 같은 경로를 탄다: Pi 프레임 -> C spi_link_parse_cmd -> (50 Hz 틱마다)
워치독 -> C IK·접기(±90°) -> 잔차 (타입 0x02, 100 ms 타임아웃) -> C 구동 정렬 게이트
-> 바퀴(띄운 상태, 40 ms 1차) -> C spi_link_pack_telemetry.  C 는 sim/libksrc_common.so
(firmware/common 그대로).  main.c Swerve_Control·pi_spi_poll·제어 루프 순서를 따른다.

    from fake_nucleo import FakeSpi
    link = SpiLink(spi=FakeSpi())          # 실물 대신
    link = SpiLink(spi=FakeSpi(enc_sign=(1, 1, -1, 1)))   # 모듈 2 엔코더 부호 실수

흉내 내지 않는 것: 서보 지연·데드밴드(조향은 3.5 rad/s 로 감), PI(개루프 duty), 지면.
"""

from __future__ import annotations

import ctypes
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
import ksrc_common as kc  # noqa: E402
import nucleo_link as nl  # noqa: E402

_LIB = kc._lib
F4 = ctypes.c_float * 4


class CCmd(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint8), ("seq", ctypes.c_uint8),
                ("vx", ctypes.c_float), ("vy", ctypes.c_float), ("omega", ctypes.c_float),
                ("d_steer", ctypes.c_float * 4), ("wheel_gain", ctypes.c_float * 4),
                ("wpi_mode", ctypes.c_uint8), ("wpi_corr", ctypes.c_float)]


class CTlm(ctypes.Structure):
    _fields_ = [("seq_echo", ctypes.c_uint8), ("tick", ctypes.c_uint8),
                ("flags", ctypes.c_uint16), ("t_ms", ctypes.c_uint32),
                ("cmd_vx", ctypes.c_float), ("cmd_vy", ctypes.c_float), ("cmd_omega", ctypes.c_float),
                ("enc_ticks", ctypes.c_int32 * 4), ("wheel_mps", ctypes.c_float * 4),
                ("steer_rad", ctypes.c_float * 4), ("duty", ctypes.c_int16 * 4),
                ("imu", ctypes.c_int16 * 6), ("rx_err", ctypes.c_uint16),
                ("steer_meas_rad", ctypes.c_float * 4), ("steer_speed", ctypes.c_int16 * 4),
                ("steer_load", ctypes.c_int16 * 4), ("steer_volt", ctypes.c_uint8 * 4),
                ("steer_temp", ctypes.c_uint8 * 4), ("steer_status", ctypes.c_uint8 * 4),
                ("servo_valid", ctypes.c_uint8), ("imu_n", ctypes.c_uint8), ("wpi_corr", ctypes.c_uint16)]


class CGate(ctypes.Structure):
    _fields_ = [("trigger_rad", ctypes.c_float), ("release_rad", ctypes.c_float),
                ("prev_goal", ctypes.c_float * 4), ("has_prev", ctypes.c_int), ("pending", ctypes.c_int),
                ("driving", ctypes.c_int), ("max_hold_ticks", ctypes.c_uint), ("hold_ticks", ctypes.c_uint),
                ("timed_out", ctypes.c_int)]


Frame = ctypes.c_uint8 * nl.SPI_FRAME_LEN
_LIB.spi_link_parse_cmd.restype = ctypes.c_int
_LIB.drive_gate_init.argtypes = [ctypes.POINTER(CGate), ctypes.c_float, ctypes.c_float, ctypes.c_uint]

MODULES_XY = ((0.102431, 0.1325), (0.102431, -0.1325), (-0.102431, 0.1325), (-0.102431, -0.1325))
MAX_MPS, PWM_MAX, R = 0.378, 799, 0.047503
TICKS_PER_REV = 8384
STEER_RATE = 3.5            # rad/s (STS3215 90° 0.87 s 근처)


class FakeSpi:
    def __init__(self, enc_sign=(1, 1, 1, 1), period_s=0.020, clock=time.monotonic):
        self.clock, self.period = clock, period_s
        self.enc_sign = enc_sign
        self.t0 = clock()
        self.n_tick = 0
        self.modules = (kc.SwerveModulePos * 4)(*[kc.SwerveModulePos(x, y) for x, y in MODULES_XY])
        self.ik_state = (kc.SwerveModuleState * 4)()
        self.gate = CGate()
        _LIB.drive_gate_init(ctypes.byref(self.gate), 60.0 * math.pi / 180.0, 10.0 * math.pi / 180.0, 75)
        self.cmd = [0.0, 0.0, 0.0]
        self.last_cmd_ms = -10_000
        self.res_d, self.res_g, self.res_ms, self.res_have = [0.0] * 4, [1.0] * 4, -10_000, False
        self.teleop = False
        self.seq_echo = 0
        self.rx_err = 0
        self.steer = [0.0] * 4
        self.goal = [0.0] * 4
        self.w = [0.0] * 4
        self.pos = [0.0] * 4
        self.duty = [0] * 4
        self.flags_extra = 0
        self.wpi_on, self.corr = True, 250.0     # 10-14 펌웨어 기본 (WPI ON, corr 250)
        self.tlm = bytes(nl.SPI_FRAME_LEN)
        self._advance()

    def _now_ms(self):
        return int((self.clock() - self.t0) * 1000)

    def _advance(self):
        """밀린 제어 틱을 모두 돈다 (실물은 50 Hz 로 혼자 돈다)."""
        due = int((self.clock() - self.t0) / self.period)
        while self.n_tick <= due:
            self._tick(self.n_tick * int(self.period * 1000))
            self.n_tick += 1

    def _tick(self, now):
        timed_out = now - self.last_cmd_ms > 500
        flags = nl.TLM_FLAG_IMU_VALID | self.flags_extra | (nl.TLM_FLAG_WHEEL_PI if self.wpi_on else 0)
        res_active = False
        if self.teleop:
            flags |= nl.TLM_FLAG_TELEOP_ACTIVE | nl.TLM_FLAG_SRC_SPI
            vx, vy, w = (0.0, 0.0, 0.0) if timed_out else self.cmd
            if timed_out:
                flags |= nl.TLM_FLAG_WATCHDOG
            res_active = (not timed_out) and self.res_have and now - self.res_ms <= 100
            out = kc.swerve_ik_compute(vx, vy, w, self.modules, MAX_MPS, self.ik_state)
            kc.swerve_fold_to_range([-math.pi / 2] * 4, [math.pi / 2] * 4, self.ik_state, out)
            goal, speed = [], []
            for m in range(4):
                g, s = out[m].angle_rad, out[m].speed_mps
                if res_active:
                    g = min(max(g + self.res_d[m], -math.pi / 2), math.pi / 2)
                    s = min(max(s * self.res_g[m], -MAX_MPS), MAX_MPS)
                goal.append(g)
                speed.append(s)
            self.goal = goal
            _LIB.drive_gate_update_goal(ctypes.byref(self.gate), F4(*goal), F4(*speed))
            allow = _LIB.drive_gate_allow(ctypes.byref(self.gate), F4(*goal), F4(*self.steer), 1)
            if self.gate.pending:
                flags |= nl.TLM_FLAG_GATE_PENDING
            if self.gate.timed_out:
                flags |= nl.TLM_FLAG_GATE_TIMEOUT
            for m in range(4):
                s = speed[m] if allow else 0.0
                self.duty[m] = int(max(-1.0, min(1.0, s / MAX_MPS)) * PWM_MAX)
                w_tgt = self.duty[m] / PWM_MAX * MAX_MPS / R
                self.w[m] += (w_tgt - self.w[m]) * self.period / (0.040 + self.period)
                step = STEER_RATE * self.period
                self.steer[m] += max(-step, min(step, goal[m] - self.steer[m]))
        if res_active:
            flags |= nl.TLM_FLAG_RES_ACTIVE
        t = CTlm()
        t.seq_echo, t.tick, t.flags, t.t_ms = self.seq_echo, (now // 20) & 0xFF, flags, now
        t.cmd_vx, t.cmd_vy, t.cmd_omega = self.cmd if (self.teleop and not timed_out) else (0.0, 0.0, 0.0)
        for m in range(4):
            self.pos[m] += self.w[m] * self.period
            ticks = int(math.floor(self.pos[m] / (2 * math.pi) * TICKS_PER_REV)) * self.enc_sign[m]
            prev = getattr(self, "_prev_ticks", [0] * 4)[m]
            t.enc_ticks[m] = ticks
            t.wheel_mps[m] = (ticks - prev) / TICKS_PER_REV * 2 * math.pi / self.period * R
            t.steer_rad[m] = self.goal[m]
            t.steer_meas_rad[m] = round(self.steer[m] / (2 * math.pi / 4096)) * (2 * math.pi / 4096)
            t.duty[m] = self.duty[m]
            t.steer_volt[m] = 120
            t.steer_temp[m] = 30
        self._prev_ticks = [t.enc_ticks[m] for m in range(4)]
        t.imu[5] = int(1.0 / nl.IMU_ACC_G_PER_LSB)       # +1 g (칩 z = 로버 z)
        t.rx_err = self.rx_err & 0xFFFF
        t.servo_valid, t.imu_n = 0x0F, 4
        t.wpi_corr = int(self.corr + 0.5)
        out = Frame()
        _LIB.spi_link_pack_telemetry(ctypes.byref(t), out)
        self.tlm = bytes(out)

    def xfer2(self, data):
        self._advance()
        reply = list(self.tlm)              # 직전 틱에 만든 텔레메트리
        c = CCmd()
        if _LIB.spi_link_parse_cmd(Frame.from_buffer_copy(bytes(data)), ctypes.byref(c)) != 0:
            self.rx_err += 1
            return reply
        self.seq_echo = c.seq
        now = self._now_ms()
        if c.type == nl.SPI_TYPE_CONFIG:
            if c.wpi_mode in (0, 1):
                self.wpi_on = bool(c.wpi_mode)
            if math.isfinite(c.wpi_corr) and 0 <= c.wpi_corr <= 799:
                self.corr = c.wpi_corr
            return reply
        if c.type in (nl.SPI_TYPE_VEL_CMD, nl.SPI_TYPE_VEL_RES):
            if all(math.isfinite(v) for v in (c.vx, c.vy, c.omega)):
                self.teleop = True
                self.cmd = [c.vx, c.vy, c.omega]
                self.last_cmd_ms = now
                self.res_have = False
        if c.type == nl.SPI_TYPE_VEL_RES and all(math.isfinite(v) for v in list(c.d_steer) + list(c.wheel_gain)):
            self.res_d = [max(-0.5236, min(0.5236, v)) for v in c.d_steer]
            self.res_g = [max(0.4, min(1.6, v)) for v in c.wheel_gain]
            self.res_ms, self.res_have = now, True
        return reply

    def close(self):
        pass

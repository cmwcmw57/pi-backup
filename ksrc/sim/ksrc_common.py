"""
libksrc_common.so 에 대한 ctypes 바인딩.

**실제 펌웨어 C 코드**(firmware/common/swerve_kinematics.c, teleop_protocol.c)를
공유 라이브러리로 빌드해서(build.sh 참고) 불러온다. 즉 시뮬레이터가 Nucleo 에
올라가는 바로 그 로직을 그대로 돌린다. 파이썬 재구현이 아니다.

아래 구조체 배치는 C 헤더와 필드 단위로 일치해야 한다 (순서, 타입 모두.
.h 에 packing pragma 가 없어서 양쪽 다 기본 정렬을 쓴다).
맨 아래 validate_bindings() 가 C 테스트 스위트의 알려진 벡터 몇 개를 이
바인딩으로 다시 돌려보고 안 맞으면 예외를 던진다. 시뮬을 믿기 전에 시작할 때
한 번 호출할 것.
"""
import ctypes
import math
from pathlib import Path

_SO_PATH = Path(__file__).resolve().parent / "libksrc_common.so"
_lib = ctypes.CDLL(str(_SO_PATH))

SWERVE_NUM_MODULES = 4
TELEOP_MAX_PAYLOAD = 16


# --- swerve_kinematics.h -------------------------------------------------

class SwerveModulePos(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class SwerveModuleState(ctypes.Structure):
    # C 구조체와 필드 순서/타입이 **정확히** 같아야 한다 (swerve_kinematics.h).
    _fields_ = [("last_angle_rad", ctypes.c_float),
                ("last_cmd_rad", ctypes.c_float),
                ("initialized", ctypes.c_int)]


class SwerveModuleCmd(ctypes.Structure):
    _fields_ = [("angle_rad", ctypes.c_float), ("speed_mps", ctypes.c_float)]


_lib.swerve_ik_compute.argtypes = [
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    SwerveModulePos * SWERVE_NUM_MODULES,
    ctypes.c_float,
    SwerveModuleState * SWERVE_NUM_MODULES,
    SwerveModuleCmd * SWERVE_NUM_MODULES,
]
_lib.swerve_ik_compute.restype = None

_lib.swerve_fold_to_limit.argtypes = [
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    SwerveModuleState * SWERVE_NUM_MODULES,
    SwerveModuleCmd * SWERVE_NUM_MODULES,
]
_lib.swerve_fold_to_limit.restype = None


def swerve_fold_to_limit(limit_rad, state, out, unwind_rad=0.0, slow_mps=0.0):
    """조향 가동범위 안으로 접는다 (속도 부호 반전 동반).  out 을 제자리 수정.

    시뮬과 펌웨어가 **같은 C 함수**를 쓴다.  예전에는 파이썬과 main.c 에 각각
    따로 구현해서 둘이 갈라질 수 있었다.
    """
    _lib.swerve_fold_to_limit(ctypes.c_float(limit_rad), ctypes.c_float(unwind_rad),
                              ctypes.c_float(slow_mps), state, out)
    return out


_lib.swerve_fold_to_range.argtypes = [
    ctypes.c_float * SWERVE_NUM_MODULES, ctypes.c_float * SWERVE_NUM_MODULES,
    ctypes.c_float, ctypes.c_float,
    SwerveModuleState * SWERVE_NUM_MODULES,
    SwerveModuleCmd * SWERVE_NUM_MODULES,
]
_lib.swerve_fold_to_range.restype = None


def swerve_fold_to_range(lo_rad, hi_rad, state, out, unwind_rad=0.0, slow_mps=0.0):
    """바퀴별 가동범위 [lo[i], hi[i]] 안으로 접는다 (C 모듈 순서). out 을 제자리 수정.

    펌웨어 main.c Swerve_Control 이 joint.c 범위로 부르는 것과 **같은 C 함수**.
    """
    lo = (ctypes.c_float * SWERVE_NUM_MODULES)(*lo_rad)
    hi = (ctypes.c_float * SWERVE_NUM_MODULES)(*hi_rad)
    _lib.swerve_fold_to_range(lo, hi, ctypes.c_float(unwind_rad), ctypes.c_float(slow_mps),
                              state, out)
    return out


def swerve_ik_compute(vx, vy, omega, modules, max_wheel_mps, state):
    """modules, state 는 SwerveModulePos*4 / SwerveModuleState*4 ctypes 배열.
    state 는 제자리에서 수정된다(C API 와 동일). SwerveModuleCmd*4 를 돌려준다."""
    out = (SwerveModuleCmd * SWERVE_NUM_MODULES)()
    _lib.swerve_ik_compute(
        ctypes.c_float(vx), ctypes.c_float(vy), ctypes.c_float(omega),
        modules, ctypes.c_float(max_wheel_mps), state, out,
    )
    return out


# --- teleop_protocol.h ----------------------------------------------------

class TeleopFrame(ctypes.Structure):
    _fields_ = [
        ("msg_type", ctypes.c_uint8),
        ("len", ctypes.c_uint8),
        ("payload", ctypes.c_uint8 * TELEOP_MAX_PAYLOAD),
    ]


class TeleopVelocityCmd(ctypes.Structure):
    _fields_ = [("vx", ctypes.c_float), ("vy", ctypes.c_float), ("omega", ctypes.c_float)]


class TeleopParser(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int),
        ("msg_type", ctypes.c_uint8),
        ("len", ctypes.c_uint8),
        ("payload", ctypes.c_uint8 * TELEOP_MAX_PAYLOAD),
        ("payload_idx", ctypes.c_uint8),
        ("crc_error_count", ctypes.c_uint32),
        ("framing_error_count", ctypes.c_uint32),
    ]


_lib.teleop_parser_init.argtypes = [ctypes.POINTER(TeleopParser)]
_lib.teleop_parser_init.restype = None

_lib.teleop_parser_feed_byte.argtypes = [ctypes.POINTER(TeleopParser), ctypes.c_uint8, ctypes.POINTER(TeleopFrame)]
_lib.teleop_parser_feed_byte.restype = ctypes.c_int

_lib.teleop_decode_velocity_cmd.argtypes = [ctypes.POINTER(TeleopFrame), ctypes.POINTER(TeleopVelocityCmd)]
_lib.teleop_decode_velocity_cmd.restype = ctypes.c_int

_lib.teleop_link_is_stale.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
_lib.teleop_link_is_stale.restype = ctypes.c_int


class TeleopLink:
    """바이트 단위 파서 + 워치독을 감싼 상태 보유 헬퍼.
    실시간 바이트 스트림을 받는 시뮬용이다. 실제 펌웨어의 UART RX 인터럽트 +
    메인 루프가 같은 C 함수를 쓰는 방식을 그대로 흉내낸다."""

    def __init__(self, stale_timeout_ms=300):
        self.parser = TeleopParser()
        _lib.teleop_parser_init(ctypes.byref(self.parser))
        self.frame = TeleopFrame()
        self.cmd = TeleopVelocityCmd()
        self.stale_timeout_ms = stale_timeout_ms
        self.last_valid_frame_ms = None

    def feed_byte(self, byte, now_ms):
        if _lib.teleop_parser_feed_byte(ctypes.byref(self.parser), byte, ctypes.byref(self.frame)):
            if _lib.teleop_decode_velocity_cmd(ctypes.byref(self.frame), ctypes.byref(self.cmd)):
                self.last_valid_frame_ms = now_ms
                return (self.cmd.vx, self.cmd.vy, self.cmd.omega)
        return None

    def is_stale(self, now_ms):
        if self.last_valid_frame_ms is None:
            return True
        return bool(_lib.teleop_link_is_stale(
            ctypes.c_uint32(now_ms), ctypes.c_uint32(self.last_valid_frame_ms),
            ctypes.c_uint32(self.stale_timeout_ms)))


# --- wheel_pi.h (바퀴 속도 FF + 기준 모델 PI, 펌웨어 main.c 와 같은 C) -------

class WheelPiCfg(ctypes.Structure):
    # wheel_pi.h 의 wheel_pi_cfg_t 와 필드 순서/타입이 정확히 같아야 한다.
    _fields_ = [(n, ctypes.c_float) for n in
                ("dt", "k_duty", "f_kin", "kick", "w_still", "w_zero", "kp", "ki", "i_max", "tau_m", "duty_max",
                 "corr_max", "stall_w", "stall_t", "enc_duty", "enc_w", "enc_t",
                 "enc_retry_t", "enc_window_t", "enc_trips_max")]


class WheelPiState(ctypes.Structure):
    _fields_ = [("r_m", ctypes.c_float), ("i", ctypes.c_float), ("stall_n", ctypes.c_uint16),
                ("enc_n", ctypes.c_uint16), ("enc_hold", ctypes.c_uint16), ("enc_win", ctypes.c_uint16),
                ("enc_trips", ctypes.c_uint8), ("stall", ctypes.c_uint8), ("enc_dir", ctypes.c_uint8)]


_lib.wheel_pi_default_cfg.argtypes = [ctypes.POINTER(WheelPiCfg)]
_lib.wheel_pi_default_cfg.restype = None
_lib.wheel_pi_reset.argtypes = [ctypes.POINTER(WheelPiState)]
_lib.wheel_pi_reset.restype = None
_lib.wheel_pi_clear_fault.argtypes = [ctypes.POINTER(WheelPiState)]
_lib.wheel_pi_clear_fault.restype = None
_lib.wheel_pi_step.argtypes = [ctypes.POINTER(WheelPiCfg), ctypes.POINTER(WheelPiState), ctypes.c_float, ctypes.c_float]
_lib.wheel_pi_step.restype = ctypes.c_int16


class WheelPi:
    """바퀴 하나의 속도 PI.  step(w_ref, w_meas) -> duty (+ = 전진, -799..799).

    펌웨어가 20 ms 마다 부르는 것과 **같은 C 함수**.  cfg 값은 kw 로 덮어쓸 수 있다
    (예: WheelPi(kp=60)).  Isaac(GPU 배치)은 torch 로 옮긴 판을 이 클래스와 같은 입력열로 비교한다."""

    def __init__(self, **kw):
        self.cfg = WheelPiCfg()
        _lib.wheel_pi_default_cfg(ctypes.byref(self.cfg))
        for k, v in kw.items():
            if not hasattr(self.cfg, k):
                raise KeyError(k)
            setattr(self.cfg, k, v)
        self.state = WheelPiState()

    def reset(self):
        _lib.wheel_pi_reset(ctypes.byref(self.state))

    def clear_fault(self):
        _lib.wheel_pi_clear_fault(ctypes.byref(self.state))

    def step(self, w_ref, w_meas):
        return int(_lib.wheel_pi_step(ctypes.byref(self.cfg), ctypes.byref(self.state),
                                      ctypes.c_float(w_ref), ctypes.c_float(w_meas)))


# --- 자체 검사: 바인딩이 C 테스트 스위트와 일치하는지 증명 ----------------

def _make_modules(half_w=0.12, half_l=0.12):
    m = (SwerveModulePos * 4)()
    m[0] = SwerveModulePos(half_l, half_w)    # FL
    m[1] = SwerveModulePos(half_l, -half_w)   # FR
    m[2] = SwerveModulePos(-half_l, half_w)   # RL
    m[3] = SwerveModulePos(-half_l, -half_w)  # RR
    return m


def validate_bindings():
    modules = _make_modules()
    state = (SwerveModuleState * 4)()

    # 순수 전진: test_swerve_kinematics.c 의 case 1 과 일치
    out = swerve_ik_compute(1.0, 0.0, 0.0, modules, -1.0, state)
    for i in range(4):
        assert abs(out[i].angle_rad - 0.0) < 1e-4, f"fwd angle[{i}]={out[i].angle_rad}"
        assert abs(out[i].speed_mps - 1.0) < 1e-4, f"fwd speed[{i}]={out[i].speed_mps}"

    # 제자리 회전 최적화: test_swerve_kinematics.c 의 case 3 과 일치
    state2 = (SwerveModuleState * 4)()
    out = swerve_ik_compute(0.0, 0.0, 1.0, modules, -1.0, state2)
    assert abs(math.degrees(out[0].angle_rad) - (-45.0)) < 1e-2, out[0].angle_rad
    assert out[0].speed_mps < 0, out[0].speed_mps

    # 텔레옵 인코딩/디코딩 왕복. 교차검증된 알려진 프레임과 비교
    link = TeleopLink()
    frame_hex = "aa 55 01 0c 00 00 c0 3f 00 00 10 c0 00 00 40 40 b1"
    frame = bytes(int(b, 16) for b in frame_hex.split())
    decoded = None
    for i, byte in enumerate(frame):
        decoded = link.feed_byte(byte, now_ms=1000 + i) or decoded
    assert decoded is not None, "known-good frame failed to decode"
    vx, vy, omega = decoded
    assert abs(vx - 1.5) < 1e-5 and abs(vy - (-2.25)) < 1e-5 and abs(omega - 3.0) < 1e-5, decoded

    # 워치독
    assert link.is_stale(now_ms=1000 + len(frame) + 1000) is True
    assert link.is_stale(now_ms=1000 + len(frame) + 10) is False

    # 바퀴 PI: test_wheel_pi.c 의 첫 두 줄과 일치 (첫 주기 333, 움직이는 중 231)
    pi = WheelPi()
    assert pi.step(2.0, 0.0) == 333, "wheel_pi first step"
    pi.reset()
    assert pi.step(2.0, 0.5) == 231, "wheel_pi moving step"
    assert pi.step(0.0, 1.0) == 0 and pi.state.i == 0.0, "wheel_pi zero ref"

    print("ksrc_common bindings self-check: OK")


if __name__ == "__main__":
    validate_bindings()

"""
라즈베리파이 -> Nucleo 시리얼 링크: 프레이밍, CRC, 메시지 인코딩.

firmware/common/teleop_protocol.h/.c 와 **바이트 단위로 동일**하다. 한쪽을
고치면 반드시 다른 쪽도 고치고 재검증해야 한다. 안 그러면 Nucleo 가 모든
프레임을 조용히 버린다 (에러도 안 난다).

프레임워크 의존성이 없는 순수 파이썬이다. ROS2 도, 시리얼 I/O 도 여기 없다.
Nucleo 와 대화해야 하는 쪽(지금은 텔레옵 스크립트, 나중에는 Nav2 cmd_vel
브리지 노드)이 이걸 import 하고 자기 pyserial.Serial 을 직접 연다.
"""
import struct

SYNC0 = 0xAA
SYNC1 = 0x55
MSG_VELOCITY_CMD = 0x01


def crc8(data: bytes) -> int:
    """CRC-8. poly 0x07, init 0x00, reflect/xorout 없음.
    firmware/common/teleop_protocol.c 의 teleop_crc8() 과 비트 단위로 동일."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def encode_velocity_cmd(vx: float, vy: float, omega: float) -> bytes:
    """차체 좌표계 속도 명령 프레임. vx, vy 는 m/s (x=전진, y=좌측),
    omega 는 rad/s (+가 반시계). swerve_kinematics.h 와 같은 규약."""
    payload = struct.pack("<fff", vx, vy, omega)
    body = bytes([MSG_VELOCITY_CMD, len(payload)]) + payload
    return bytes([SYNC0, SYNC1]) + body + bytes([crc8(body)])


# ---------------------------------------------------------------------------
# SPI 링크 (Pi = SPI 마스터, Nucleo SPI2 = 슬레이브)
#
# firmware/common/spi_link.h/.c 와 **바이트 단위로 동일**하다. 레이아웃 설명은
# spi_link.h 가 정본이다. 전이중이라 한 번의 전송(144 바이트)에서 명령을 보내며
# 동시에 Nucleo 가 직전 제어주기(50Hz)에 만든 텔레메트리를 받는다.
# 텔레메트리 하나에 구동 엔코더, 조향 서보 실측(STS3215), IMU(ISM330DHCX) 가
# 같은 제어주기 시각 t_ms 로 묶여 있다.
#
# Pi 설정: SPI 모드 0, 8비트, 1MHz 권장 (Nucleo 가 HSI 16MHz 라 8MHz 가 절대 상한).
# 반드시 매번 SPI_FRAME_LEN 바이트를 한 번의 CS 구간으로 보낸다. 길이가 다르면
# Nucleo 가 그 전송을 버린다.
# ---------------------------------------------------------------------------

SPI_FRAME_LEN = 144
SPI_SYNC0 = 0xA5
SPI_SYNC1 = 0x5A
SPI_VERSION = 2
SPI_TYPE_POLL = 0x00
SPI_TYPE_VEL_CMD = 0x01
SPI_TYPE_VEL_RES = 0x02           # 속도 명령 + 바퀴별 잔차 (RL Res8). firmware/common/spi_link.h
SPI_TYPE_CONFIG = 0x03            # 설정: 바퀴 PI ON/OFF, 보정량 상한 (주행 명령 아님)
SPI_WPI_KEEP = 0xFF
SPI_TYPE_TELEMETRY = 0x81

TLM_FLAG_TELEOP_ACTIVE = 1 << 0
TLM_FLAG_WATCHDOG = 1 << 1
TLM_FLAG_GATE_PENDING = 1 << 2
TLM_FLAG_SRC_SPI = 1 << 3
TLM_FLAG_IMU_VALID = 1 << 4
TLM_FLAG_SERVO_READ_ERR = 1 << 5
TLM_FLAG_STEER_CLAMPED = 1 << 6
TLM_FLAG_GATE_TIMEOUT = 1 << 7    # 정렬 대기 1.5초 초과로 강제 해제 (조향 막힘 의심)
TLM_FLAG_WHEEL_PI = 1 << 8        # 구동이 바퀴 속도 PI (0 = 개루프 duty). firmware/common/wheel_pi.h
TLM_FLAG_WHEEL_STALL = 1 << 9     # 바퀴 PI 스톨 보호 중 (그 바퀴 FF 만)
TLM_FLAG_ENC_DIR = 1 << 10        # 엔코더 방향 불일치로 잠김 (FF 만, WPI ON 으로 해제)
TLM_FLAG_RES_ACTIVE = 1 << 11     # 이번 주기 IK 에 바퀴별 잔차를 더함 (타입 0x02 가 제때 옴)
TLM_FLAG_IMU_READ_TIME_VALID = 1 << 12  # (동료 10-15 펌웨어) [140..141] = IMU 평균 표본 읽기 시각 offset i16 [us]

# [2]..[141] 영역: ver, type, seq_echo, tick, flags, t_ms, cmd x3, ticks x4,
# wheel_mps x4, steer_cmd x4, duty x4, imu x6, rx_err,
# steer_meas x4, steer_speed x4, steer_load x4, volt x4, temp x4, status x4, servo_valid, 예약 3
_TLM_BODY = struct.Struct("<BBBBHI3f4i4f4f4h6hH4f4h4h4B4B4BBBH")
assert 2 + _TLM_BODY.size == SPI_FRAME_LEN - 2

# 원시값 -> 물리량 (펌웨어 imu_ism330.c 설정: ±500 dps, ±4 g)
IMU_GYRO_DPS_PER_LSB = 0.0175
IMU_ACC_G_PER_LSB = 0.000122

# 칩 좌표 -> 로버 좌표 (x 전진, y 좌측, z 위). 로버축 i = 부호 * 칩축[인덱스].
# 2026-10-05 실측: 칩이 z 축 기준 180도 돌아간 채 장착 -> x, y 반대, z 그대로
# (정지 시 칩 acc z = +1g, 손으로 돌려 x/y 부호 확인). 장착을 바꾸면 여기만 고친다.
IMU_CHIP_TO_BODY = ((0, -1), (1, -1), (2, +1))


def _chip_to_body(xyz):
    return tuple(sign * xyz[idx] for idx, sign in IMU_CHIP_TO_BODY)
STEER_STEP_PER_REV = 4096


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE. spi_link.c 의 spi_link_crc16() 과 비트 단위로 동일."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_spi_cmd(vx: float, vy: float, omega: float, seq: int,
                   msg_type: int = SPI_TYPE_VEL_CMD) -> bytes:
    """SPI 명령 프레임 (SPI_FRAME_LEN 바이트, 0 패딩). msg_type=SPI_TYPE_POLL 이면 명령 없이
    텔레메트리만 읽는다 (Nucleo 워치독을 갱신하지 않는다)."""
    body = struct.pack("<BBBBfff", SPI_VERSION, msg_type, seq & 0xFF, 0, vx, vy, omega)
    head = bytes([SPI_SYNC0, SPI_SYNC1]) + body
    frame = head + struct.pack("<H", crc16(body))
    return frame + bytes(SPI_FRAME_LEN - len(frame))


def encode_spi_res_cmd(vx: float, vy: float, omega: float, d_steer, wheel_gain, seq: int) -> bytes:
    """SPI 타입 0x02: 차체 명령 + 바퀴별 잔차 (모듈 순서 FL FR RL RR).
    펌웨어가 지금처럼 IK·접기를 한 뒤 조향각 += d_steer [rad], 바퀴 속도 *= wheel_gain.
    시뮬 Res8Action 과 같은 뜻: d_steer = a[:4] * RES8_STEER_SCALE_RAD, wheel_gain = 1 + a[4:] * RES8_WHEEL_SCALE."""
    d_steer, wheel_gain = list(d_steer), list(wheel_gain)
    if len(d_steer) != 4 or len(wheel_gain) != 4:
        raise ValueError("d_steer, wheel_gain: 4 개씩")
    body = struct.pack("<BBBBfff4f4f", SPI_VERSION, SPI_TYPE_VEL_RES, seq & 0xFF, 0, vx, vy, omega,
                       *d_steer, *wheel_gain)
    frame = bytes([SPI_SYNC0, SPI_SYNC1]) + body + struct.pack("<H", crc16(body))
    return frame + bytes(SPI_FRAME_LEN - len(frame))


def encode_spi_config(seq: int, wpi_on=None, corr=None) -> bytes:
    """SPI 타입 0x03: 바퀴 PI 설정. wpi_on True/False/None(그대로), corr [duty] 또는 None(그대로).
    주행 명령이 아니라 워치독·텔레옵에 영향 없음. 결과는 텔레메트리 flags TLM_FLAG_WHEEL_PI 와 "wpi_corr"."""
    mode = SPI_WPI_KEEP if wpi_on is None else (1 if wpi_on else 0)
    c = float("nan") if corr is None else float(corr)
    body = struct.pack("<BBBBBBf", SPI_VERSION, SPI_TYPE_CONFIG, seq & 0xFF, 0, mode, 0, c)
    frame = bytes([SPI_SYNC0, SPI_SYNC1]) + body + struct.pack("<H", crc16(body))
    return frame + bytes(SPI_FRAME_LEN - len(frame))


def decode_spi_telemetry(frame: bytes):
    """텔레메트리 프레임 -> dict. 동기/버전/CRC/타입이 틀리면 None.

    지연: 받는 값은 Nucleo 가 **직전 CS 해제 시점까지** 만든 프레임이라 0~40ms 묵었고,
    방금 보낸 명령은 아직 반영되지 않았다 (seq_echo 로 확인). Pi 교환과 Nucleo 50Hz
    제어주기가 비동기라 같은 tick 을 두 번 받거나 하나를 건너뛸 수 있다 -> tick 으로
    중복을 거른다. 시각은 수신 시각이 아니라 t_ms (Nucleo 클럭) 를 쓴다.
    배열은 모듈 순서 FL, FR, RL, RR. imu 값은 flags & TLM_FLAG_IMU_VALID,
    서보 값은 servo_valid[m] 일 때만 믿는다."""
    if len(frame) != SPI_FRAME_LEN or frame[0] != SPI_SYNC0 or frame[1] != SPI_SYNC1:
        return None
    body = bytes(frame[2:SPI_FRAME_LEN - 2])
    if struct.unpack_from("<H", frame, SPI_FRAME_LEN - 2)[0] != crc16(body):
        return None
    v = _TLM_BODY.unpack(body)
    if v[0] != SPI_VERSION or v[1] != SPI_TYPE_TELEMETRY:
        return None
    return {
        "seq_echo": v[2],
        "tick": v[3],
        "flags": v[4],
        "t_ms": v[5],
        "cmd": v[6:9],
        "enc_ticks": v[9:13],
        "wheel_mps": v[13:17],
        "steer_cmd_rad": v[17:21],
        "duty": v[21:25],
        "imu_raw": v[25:31],              # gyro xyz, acc xyz (칩 좌표계 원시값)
        # 아래 둘은 로버 좌표계 (IMU_CHIP_TO_BODY 적용)
        "gyro_dps": _chip_to_body([x * IMU_GYRO_DPS_PER_LSB for x in v[25:28]]),
        "acc_g": _chip_to_body([x * IMU_ACC_G_PER_LSB for x in v[28:31]]),
        "rx_err": v[31],
        "steer_meas_rad": v[32:36],
        "steer_speed": v[36:40],          # step/s (4096 step = 1회전)
        "steer_load": v[40:44],           # 0.1 %
        "steer_volt": tuple(x * 0.1 for x in v[44:48]),
        "steer_temp": v[48:52],           # 섭씨
        "steer_status": v[52:56],
        "servo_valid": tuple(bool(v[56] >> m & 1) for m in range(4)),
        # [140..141] 은 펌웨어에 따라 뜻이 다르다: bit12 가 켜진 펌웨어(동료 IMU 시각, 10-15) 는 IMU offset i16 [us],
        # 아니면 바퀴 PI 보정량 상한 [duty] (10-14 이전 펌웨어는 0).  bit12 펌웨어에서 wpi_corr 는 -1 (모름).
        "wpi_corr": -1 if v[4] & TLM_FLAG_IMU_READ_TIME_VALID else v[58],
        "imu_mean_read_offset_us": (v[58] - 65536 if v[58] >= 32768 else v[58]) if v[4] & TLM_FLAG_IMU_READ_TIME_VALID else None,
        "imu_n": v[57],                   # imu 가 이번 20 ms 동안 몇 샘플의 평균인지 (0 = 순간값)
    }


class SpiLink:
    """spidev 로 Nucleo 와 한 번에 한 프레임씩 주고받는다.

    link = SpiLink(bus=0, device=0)
    tlm = link.exchange(0.1, 0.0, 0.0)   # 명령 보내고, 직전 텔레메트리 받기
    """

    def __init__(self, bus: int = 0, device: int = 0, speed_hz: int = 1_000_000, spi=None):
        """spi: xfer2(list) 를 가진 대역 객체 (노트북 시험용 fake_nucleo.FakeSpi). None 이면 spidev."""
        self._seq = 0
        if spi is not None:
            self._spi = spi
            return
        import spidev  # Pi 에서만 필요하므로 여기서 import
        self._spi = spidev.SpiDev()
        self._spi.open(bus, device)
        self._spi.mode = 0
        self._spi.bits_per_word = 8
        self._spi.max_speed_hz = speed_hz

    def exchange(self, vx: float, vy: float, omega: float, poll_only: bool = False):
        self._seq = (self._seq + 1) & 0xFF
        frame = encode_spi_cmd(vx, vy, omega, self._seq,
                               SPI_TYPE_POLL if poll_only else SPI_TYPE_VEL_CMD)
        rx = bytes(self._spi.xfer2(list(frame)))
        return decode_spi_telemetry(rx)

    def exchange_res(self, vx: float, vy: float, omega: float, d_steer, wheel_gain):
        """속도 명령 + 바퀴별 잔차 (RL 정책). 텔레옵은 exchange() 그대로."""
        self._seq = (self._seq + 1) & 0xFF
        frame = encode_spi_res_cmd(vx, vy, omega, d_steer, wheel_gain, self._seq)
        rx = bytes(self._spi.xfer2(list(frame)))
        return decode_spi_telemetry(rx)

    def config(self, wpi_on=None, corr=None):
        """바퀴 PI 켜고 끄기 / 보정량 상한 (모래 A/B 등). 텔레메트리를 돌려준다 (적용은 다음 프레임부터 보임)."""
        self._seq = (self._seq + 1) & 0xFF
        rx = bytes(self._spi.xfer2(list(encode_spi_config(self._seq, wpi_on, corr))))
        return decode_spi_telemetry(rx)

    def close(self):
        self._spi.close()

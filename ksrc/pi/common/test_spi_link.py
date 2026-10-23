"""nucleo_link.py 의 SPI 프레임이 펌웨어 C(firmware/common/spi_link.c)와 바이트 단위로
같은지 교차검증한다. 먼저 `bash sim/build.sh` 로 libksrc_common.so 를 만든다.

    python3 pi/common/test_spi_link.py
"""
import ctypes
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nucleo_link as nl  # noqa: E402

_LIB = ctypes.CDLL(str(Path(__file__).resolve().parents[2] / "sim" / "libksrc_common.so"))


class CCmd(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint8), ("seq", ctypes.c_uint8),
                ("vx", ctypes.c_float), ("vy", ctypes.c_float), ("omega", ctypes.c_float),
                ("d_steer", ctypes.c_float * 4), ("wheel_gain", ctypes.c_float * 4),
                ("wpi_mode", ctypes.c_uint8), ("wpi_corr", ctypes.c_float)]


class CTlm(ctypes.Structure):
    _fields_ = [("seq_echo", ctypes.c_uint8), ("tick", ctypes.c_uint8),
                ("flags", ctypes.c_uint16), ("t_ms", ctypes.c_uint32),
                ("cmd_vx", ctypes.c_float), ("cmd_vy", ctypes.c_float),
                ("cmd_omega", ctypes.c_float),
                ("enc_ticks", ctypes.c_int32 * 4), ("wheel_mps", ctypes.c_float * 4),
                ("steer_rad", ctypes.c_float * 4), ("duty", ctypes.c_int16 * 4),
                ("imu", ctypes.c_int16 * 6), ("rx_err", ctypes.c_uint16),
                ("steer_meas_rad", ctypes.c_float * 4), ("steer_speed", ctypes.c_int16 * 4),
                ("steer_load", ctypes.c_int16 * 4), ("steer_volt", ctypes.c_uint8 * 4),
                ("steer_temp", ctypes.c_uint8 * 4), ("steer_status", ctypes.c_uint8 * 4),
                ("servo_valid", ctypes.c_uint8), ("imu_n", ctypes.c_uint8), ("wpi_corr", ctypes.c_uint16)]


Frame = ctypes.c_uint8 * nl.SPI_FRAME_LEN
_LIB.spi_link_crc16.restype = ctypes.c_uint16
_LIB.spi_link_parse_cmd.restype = ctypes.c_int


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def test_crc16_matches_c():
    rng = random.Random(0)
    for n in (0, 1, 9, 16, 92):
        data = bytes(rng.randrange(256) for _ in range(n))
        buf = (ctypes.c_uint8 * max(n, 1)).from_buffer_copy(data or b"\0")
        assert nl.crc16(data) == _LIB.spi_link_crc16(buf, n)
    assert nl.crc16(b"123456789") == 0x29B1


def test_cmd_frame_bytes_match_c():
    rng = random.Random(1)
    for _ in range(500):
        vx, vy, w = (f32(rng.uniform(-1, 1)) for _ in range(3))
        seq = rng.randrange(256)
        typ = rng.choice((nl.SPI_TYPE_VEL_CMD, nl.SPI_TYPE_POLL))
        py = nl.encode_spi_cmd(vx, vy, w, seq, typ)
        out = Frame()
        _LIB.spi_link_pack_cmd(ctypes.byref(CCmd(typ, seq, vx, vy, w)), out)
        assert py == bytes(out)
        parsed = CCmd()
        assert _LIB.spi_link_parse_cmd(Frame.from_buffer_copy(py), ctypes.byref(parsed)) == 0
        assert (parsed.seq, parsed.vx, parsed.vy, parsed.omega) == (seq, vx, vy, w)


def test_res_cmd_frame_bytes_match_c():
    rng = random.Random(2)
    for _ in range(500):
        vx, vy, w = (f32(rng.uniform(-1, 1)) for _ in range(3))
        ds = [f32(rng.uniform(-0.6, 0.6)) for _ in range(4)]
        g = [f32(rng.uniform(0.3, 1.7)) for _ in range(4)]
        seq = rng.randrange(256)
        py = nl.encode_spi_res_cmd(vx, vy, w, ds, g, seq)
        out = Frame()
        c = CCmd(nl.SPI_TYPE_VEL_RES, seq, vx, vy, w, (ctypes.c_float * 4)(*ds), (ctypes.c_float * 4)(*g))
        _LIB.spi_link_pack_cmd(ctypes.byref(c), out)
        assert py == bytes(out)
        parsed = CCmd()
        assert _LIB.spi_link_parse_cmd(Frame.from_buffer_copy(py), ctypes.byref(parsed)) == 0
        assert list(parsed.d_steer) == ds and list(parsed.wheel_gain) == g and parsed.type == 2


def test_config_frame_bytes_match_c():
    import math
    rng = random.Random(3)
    for _ in range(300):
        seq = rng.randrange(256)
        on = rng.choice((None, True, False))
        corr = rng.choice((None, f32(rng.uniform(0, 799))))
        py = nl.encode_spi_config(seq, on, corr)
        c = CCmd(nl.SPI_TYPE_CONFIG, seq)
        c.wpi_mode = nl.SPI_WPI_KEEP if on is None else int(on)
        c.wpi_corr = float("nan") if corr is None else corr
        out = Frame()
        _LIB.spi_link_pack_cmd(ctypes.byref(c), out)
        assert py == bytes(out)
        p = CCmd()
        assert _LIB.spi_link_parse_cmd(Frame.from_buffer_copy(py), ctypes.byref(p)) == 0
        assert p.type == 3 and p.wpi_mode == c.wpi_mode
        assert (math.isnan(p.wpi_corr) and corr is None) or p.wpi_corr == corr


def test_telemetry_from_c_decodes_in_python():
    rng = random.Random(2)
    for _ in range(500):
        t = CTlm()
        t.seq_echo, t.tick = rng.randrange(256), rng.randrange(256)
        t.flags, t.t_ms = rng.randrange(1 << 16), rng.randrange(1 << 32)
        t.cmd_vx, t.cmd_vy, t.cmd_omega = (f32(rng.uniform(-1, 1)) for _ in range(3))
        for i in range(4):
            t.enc_ticks[i] = rng.randrange(-(1 << 31), 1 << 31)
            t.wheel_mps[i] = f32(rng.uniform(-0.4, 0.4))
            t.steer_rad[i] = f32(rng.uniform(-3.2, 3.2))
            t.duty[i] = rng.randrange(-799, 800)
        for i in range(6):
            t.imu[i] = rng.randrange(-32768, 32768)
        t.rx_err = rng.randrange(1 << 16)
        for i in range(4):
            t.steer_meas_rad[i] = f32(rng.uniform(-1.8, 1.8))
            t.steer_speed[i] = rng.randrange(-4000, 4001)
            t.steer_load[i] = rng.randrange(-1000, 1001)
            t.steer_volt[i] = rng.randrange(256)
            t.steer_temp[i] = rng.randrange(256)
            t.steer_status[i] = rng.randrange(256)
        t.servo_valid = rng.randrange(16)
        t.imu_n = rng.randrange(256)
        t.wpi_corr = rng.randrange(800)
        out = Frame()
        _LIB.spi_link_pack_telemetry(ctypes.byref(t), out)
        d = nl.decode_spi_telemetry(bytes(out))
        assert d is not None
        assert (d["seq_echo"], d["tick"], d["flags"], d["t_ms"], d["rx_err"]) == \
            (t.seq_echo, t.tick, t.flags, t.t_ms, t.rx_err)
        assert d["cmd"] == (t.cmd_vx, t.cmd_vy, t.cmd_omega)
        assert d["enc_ticks"] == tuple(t.enc_ticks)
        assert d["wheel_mps"] == tuple(t.wheel_mps)
        assert d["steer_cmd_rad"] == tuple(t.steer_rad)
        assert d["duty"] == tuple(t.duty)
        assert d["imu_raw"] == tuple(t.imu)
        assert d["steer_meas_rad"] == tuple(t.steer_meas_rad)
        assert d["steer_speed"] == tuple(t.steer_speed)
        assert d["steer_load"] == tuple(t.steer_load)
        assert d["steer_volt"] == tuple(x * 0.1 for x in t.steer_volt)
        assert d["steer_temp"] == tuple(t.steer_temp)
        assert d["steer_status"] == tuple(t.steer_status)
        assert d["servo_valid"] == tuple(bool(t.servo_valid >> m & 1) for m in range(4))
        assert d["imu_n"] == t.imu_n
        if t.flags & nl.TLM_FLAG_IMU_READ_TIME_VALID:      # 동료 IMU 시각 펌웨어: 같은 2 바이트가 IMU offset
            assert d["wpi_corr"] == -1 and d["imu_mean_read_offset_us"] == (t.wpi_corr if t.wpi_corr < 32768 else t.wpi_corr - 65536)
        else:
            assert d["wpi_corr"] == t.wpi_corr and d["imu_mean_read_offset_us"] is None
        bad = bytearray(out)
        bad[rng.randrange(2, nl.SPI_FRAME_LEN)] ^= 1 << rng.randrange(8)
        assert nl.decode_spi_telemetry(bytes(bad)) is None


if __name__ == "__main__":
    test_crc16_matches_c()
    test_cmd_frame_bytes_match_c()
    test_res_cmd_frame_bytes_match_c()
    test_config_frame_bytes_match_c()
    test_telemetry_from_c_decodes_in_python()
    print("ALL SPI LINK CROSS-CHECKS PASSED")

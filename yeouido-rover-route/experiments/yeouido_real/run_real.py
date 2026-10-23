# ================================================================
# run_real.py
# 설명: pose(--pose dr: 텔레메트리 dead-reckoning / orb: ORB base_pose / eskf: ORB+IMU ESKF) -> RouteRunner -> CommandSender 루프. 매 스텝 trace를 npz로 저장한다. --fake는 이상적 로버(텔레메트리 모사)로 노트북에서 경로를 끝까지 돈다. 실제 SPI·모터는 쓰지 않는다 (SPI 쓰기는 run_policy 담당). 키보드 c=체크포인트 재기준(x y yaw_deg 입력), q=정지·종료.
# 사용법:
#   python -m experiments.yeouido_real.run_real --fake
#   python -m experiments.yeouido_real.run_real --fake --pose eskf --log trace.npz   # 합성 IMU·ORB (노트북)
#   python -m experiments.yeouido_real.run_real --pose eskf --tlm-socket /tmp/ksrc-run.sock --cmd-to 127.0.0.1:5005  # 라파, ROS 소싱 후
#   python -m experiments.yeouido_real.run_real --pose orb --cmd-to 127.0.0.1:5005   # 라파, ROS 소싱 후. 시작 위치에서 c 로 정렬
#   python -m experiments.yeouido_real.run_real --pose dr --cmd-to raspberrypi.local:5005 --tlm-port 5007
# ================================================================
import argparse
import json
import math
import select
import socket
import sys
import termios
import time
import tty
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.orbslam3_ros2.swerve import SwerveGeometry, body_twist_to_modules
from experiments.orbslam3_ros2.yeouido_route import STOP, RouteRunner
from experiments.yeouido_real.command_sender import CommandSender, clamp_cmd
from experiments.yeouido_real.orb_pose import OrbAligner, OrbSource
from experiments.yeouido_real.pose_estimator import GEOMETRY_PATH, PoseEstimator

PLAN = Path(__file__).resolve().parents[2] / "docs" / "yeouido_route_plan.json"
TLM_KEYS = ("t_ms", "tick", "flags", "gyro_dps", "acc_g", "steer_meas_rad", "wheel_mps")


class FakeRover:
    """마지막 명령을 즉시 따르는 이상적 로버. 같은 명령 경로(clamp)와 텔레메트리 dict를 흉내내고 진짜 pose도 적분한다."""

    def __init__(self, geometry: SwerveGeometry, x: float, y: float, yaw: float) -> None:
        self.geometry, self.cmd, self.t_ms = geometry, (0.0, 0.0, 0.0), 0
        self.x, self.y, self.yaw = x, y, yaw

    def send(self, vx: float, vy: float, w: float) -> bool:
        """명령을 받는다 (CommandSender.send와 같은 클램프)."""
        self.cmd = clamp_cmd(vx, vy, w)
        return True

    def recv(self) -> dict:
        """20 ms 진행시킨 텔레메트리를 만든다."""
        vx, vy, w = self.cmd
        mods = body_twist_to_modules(self.geometry, vx=vx, vy=vy, yaw_rate=w)
        r = self.geometry.wheel_radius_m
        self.t_ms += 20
        self.x += (math.cos(self.yaw) * vx - math.sin(self.yaw) * vy) * 0.02
        self.y += (math.sin(self.yaw) * vx + math.cos(self.yaw) * vy) * 0.02
        self.yaw += w * 0.02
        return {"t_ms": self.t_ms, "tick": (self.t_ms // 20) & 0xFF, "flags": 1 << 4, "gyro_dps": (0.0, 0.0, math.degrees(w)),
                "acc_g": (0.0, 0.0, 1.0), "steer_meas_rad": tuple(m.steering_angle_rad for m in mods),
                "wheel_mps": tuple(m.wheel_angular_velocity_rad_s * r for m in mods)}


class UdpTelemetry:
    """run_policy가 UDP로 내보내는 텔레메트리 JSON(TLM_KEYS)을 받는다. Pi 쪽 송신은 telemetry_packet()으로 만든다."""

    def __init__(self, port: int, timeout_s: float = 0.5) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(timeout_s)

    def recv(self) -> dict | None:
        """다음 텔레메트리 dict, 시간 내에 없으면 None."""
        try:
            return {**json.loads(self.sock.recv(4096)), "host_s": time.time()}
        except (socket.timeout, ValueError):
            return None


class UnixTelemetry:
    """run_policy가 pi/common/telemetry_tee.py(TelemetryTee)로 내보내는 UNIX datagram JSON을 받는다. payload["telemetry"]를 풀고 host_receipt_ns를 host_s로 쓴다."""

    def __init__(self, path: str, timeout_s: float = 0.5) -> None:
        self.path, self.sock = path, socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(path)
        self.sock.settimeout(timeout_s)

    def recv(self) -> dict | None:
        """다음 텔레메트리 dict, 시간 내에 없으면 None."""
        try:
            m = json.loads(self.sock.recv(65536))
        except (socket.timeout, ValueError):
            return None
        tlm = m.get("telemetry")
        if not isinstance(tlm, dict) or any(k not in tlm for k in TLM_KEYS):
            return None
        return {**tlm, "host_s": m["host_receipt_ns"] * 1e-9 if m.get("host_receipt_ns") else time.time()}

    def close(self) -> None:
        """소켓을 닫고 파일을 지운다."""
        self.sock.close()
        Path(self.path).unlink(missing_ok=True)


class FakeEskfRover(FakeRover):
    """FakeRover에 몸체 좌표 IMU(비력 포함, 50 Hz)와 지연된 15 Hz ORB base pose(지도 좌표, 잡음)를 합성한다. orb_ok=False면 추적 실패."""

    def __init__(self, geometry: SwerveGeometry, x: float, y: float, yaw: float, z: float = 0.0,
                 orb_lag_s: float = 0.25, seed: int = 0) -> None:
        from scipy.spatial.transform import Rotation  # scipy는 eskf 전용
        super().__init__(geometry, x, y, yaw)
        self.rotation = Rotation
        self.z, self.lag, self.rng, self.orb_ok = z, orb_lag_s, np.random.default_rng(seed), True
        self.vw, self.hist, self.n = (0.0, 0.0), deque(maxlen=100), 0
        self.map_r = Rotation.from_euler("z", math.radians(37.0))  # ORB 지도 좌표는 경로 좌표와 임의로 다르다
        self.map_t = np.array([5.0, -3.0, 1.0])

    def recv(self) -> dict:
        """20 ms 진행: IMU 텔레메트리 + (3틱마다) 0.25 s 늦은 ORB 샘플을 sample dict로 낸다."""
        vx, vy, w = self.cmd
        c0, s0 = math.cos(self.yaw), math.sin(self.yaw)
        vw = (c0 * vx - s0 * vy, s0 * vx + c0 * vy)
        tlm = super().recv()
        c, s_ = math.cos(self.yaw), math.sin(self.yaw)
        awx, awy = (vw[0] - self.vw[0]) / 0.02, (vw[1] - self.vw[1]) / 0.02
        self.vw = vw
        tlm["acc_g"] = ((c * awx + s_ * awy) / 9.81, (-s_ * awx + c * awy) / 9.81, 1.0)
        t = self.t_ms / 1000.0
        self.hist.append((t, self.x, self.y, self.yaw))
        self.n += 1
        orb = None
        if self.n % 3 == 0 and t > self.lag:
            tc, x, y, yaw = min(self.hist, key=lambda h: abs(h[0] - (t - self.lag)))
            yaw += self.rng.normal(0.0, math.radians(0.1))
            pos = self.map_r.apply([x + self.rng.normal(0, 0.002), y + self.rng.normal(0, 0.002), self.z]) + self.map_t
            quat = (self.map_r * self.rotation.from_euler("z", yaw)).as_quat()
            orb = {"t_ms": round(tc * 1000), "frame_id": "orb_map_0", "pos": tuple(pos), "quat": tuple(quat),
                   "tracking_ok": self.orb_ok}
        return {"t_ms": self.t_ms, "host_s": 5000.0 + t, "tlm": tlm, "orb": orb}


def telemetry_packet(tlm: dict) -> bytes:
    """decode_spi_telemetry() dict에서 필요한 키만 JSON 바이트로 만든다 (Pi 쪽 송신용)."""
    return json.dumps({k: tlm[k] for k in TLM_KEYS}).encode()


TRACE_COLUMNS = ("t_ms", "t_s", "x_m", "y_m", "yaw_rad", "pitch_rad", "roll_rad", "cmd_vx", "cmd_vy", "cmd_wz",
                 "ready", "step_index", "label", "status")


def save_trace(path: str | Path, trace: list, meta: dict) -> None:
    """trace 행들을 컬럼별 배열로 npz에 저장한다 (meta는 JSON 문자열 'meta'로 함께)."""
    cols = {k: np.array(v) for k, v in zip(TRACE_COLUMNS, zip(*trace))} if trace else {k: np.array([]) for k in TRACE_COLUMNS}
    np.savez(path, meta=json.dumps(meta, ensure_ascii=False), **cols)


def run(runner: RouteRunner, est: PoseEstimator, source, sender, *, key: Callable[[], str | None] = lambda: None,
        ask_pose: Callable[[], tuple[float, float, float] | None] = lambda: None, max_s: float = 3000.0,
        trace: list | None = None) -> str:
    """텔레메트리마다 pose 갱신 -> twist 계산 -> 송신. 'complete'|'aborted'|'quit'|'timeout'을 반환하고 끝에 0 명령을 보낸다. trace에 TRACE_COLUMNS 순서로 한 행씩 쌓는다."""
    t0: float | None = None
    try:
        while True:
            k = key()
            if k == "q":
                return "quit"
            if k == "c" and (p := ask_pose()):
                est.reanchor(*p)
            tlm = source.recv()
            if tlm is None:
                sender.send(0.0, 0.0, 0.0)
                continue
            if not est.update(tlm):
                continue
            t0 = tlm["t_ms"] if t0 is None else t0
            t = (tlm["t_ms"] - t0) / 1000.0
            if t > max_s:
                return "timeout"
            cmd = STOP
            if est.ready:
                x, y, yaw = est.pose
                cmd = runner.command(elapsed_s=t, x=x, y=y, yaw_rad=yaw, pitch_rad=est.pitch, roll_rad=est.roll)
            sender.send(cmd.vx, cmd.vy, cmd.yaw_rate)
            if trace is not None:
                trace.append((tlm["t_ms"], t, *est.pose, est.pitch, est.roll, cmd.vx, cmd.vy, cmd.yaw_rate,
                              est.ready, runner.index, runner.label, getattr(est, "status", "")))
            if not est.ready:
                continue
            if runner.aborted:
                return "aborted"
            if runner.complete:
                return "complete"
    finally:
        sender.send(0.0, 0.0, 0.0)


def main(argv: list[str] | None = None) -> int:
    """인자를 읽어 --fake 또는 UDP 실기 모드로 run()을 돌린다. 완주하면 0."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", default=str(PLAN))
    ap.add_argument("--fake", action="store_true", help="이상적 로버로 노트북에서 실행")
    ap.add_argument("--pose", choices=("dr", "orb", "eskf"), default="dr",
                    help="pose 소스: dr=텔레메트리 dead-reckoning, orb=/orbslam3/base_pose, eskf=ORB+IMU ESKF")
    ap.add_argument("--cmd-to", default="127.0.0.1:5005", help="명령 송신 HOST:PORT (Pi run_policy --cmd-udp)")
    ap.add_argument("--tlm-port", type=int, default=5007, help="텔레메트리 수신 UDP 포트")
    ap.add_argument("--tlm-socket", default=None, help="텔레메트리 UNIX datagram 경로 (run_policy TelemetryTee). 있으면 UDP 대신 사용")
    ap.add_argument("--lag-s", type=float, default=0.25, help="eskf: ORB pose 지연 [s] (IMU를 이만큼 늦춰 영상과 맞춘다). 실측해서 지정")
    ap.add_argument("--imu-offset-s", type=float, default=-0.047, help="eskf: IMU 시각 보정 [s] (legacy SPI 카메라-IMU -47 ms)")
    ap.add_argument("--max-s", type=float, default=3000.0, help="이 경과 시간(s)에 중단")
    ap.add_argument("--log", default=time.strftime("yeouido_trace_%Y%m%d_%H%M%S.npz"), help="trace npz 저장 경로")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.route).read_text(encoding="utf-8"))
    x0, y0, z0 = spec["start"]["base_link_xyz_m"][:3]
    yaw0 = math.radians(spec["start"]["yaw_deg"])
    geometry = SwerveGeometry.from_json(GEOMETRY_PATH)
    runner = RouteRunner(spec)
    if a.pose == "eskf":
        from experiments.yeouido_real.eskf_pose import EskfPose, EskfSource  # scipy·ROVER_MAPPING_ROOT 필요
        est = EskfPose(x0, y0, z0, yaw0, lag_s=a.lag_s, imu_offset_s=a.imu_offset_s)
    else:
        est = OrbAligner() if a.pose == "orb" and not a.fake else PoseEstimator(geometry, x0, y0, yaw0)
    if a.fake:
        source = sender = FakeEskfRover(geometry, x0, y0, yaw0, z0) if a.pose == "eskf" else FakeRover(geometry, x0, y0, yaw0)
    else:
        host, port = a.cmd_to.rsplit(":", 1)
        tlm = UnixTelemetry(a.tlm_socket) if a.tlm_socket else UdpTelemetry(a.tlm_port)
        source = {"orb": OrbSource, "eskf": lambda: EskfSource(tlm, OrbSource())}.get(a.pose, lambda: tlm)()
        sender = CommandSender(host, int(port))
    if a.pose == "orb" and not a.fake:
        print(f"ORB 모드: 로버를 시작 위치에 놓고 추적이 OK가 되면 c 입력 -> '{x0:.3f} {y0:.3f} {math.degrees(yaw0):.0f}'")
    trace: list = []
    if not sys.stdin.isatty():
        result = run(runner, est, source, sender, trace=trace, max_s=a.max_s)
    else:
        fd, saved = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())

        def key() -> str | None:
            return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None

        def ask_pose() -> tuple[float, float, float] | None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            try:
                v = [float(s) for s in input("\n줄자 실측 x[m] y[m] yaw[deg]: ").split()]
                return (v[0], v[1], math.radians(v[2])) if len(v) == 3 else None
            except ValueError:
                return None
            finally:
                tty.setcbreak(fd)

        tty.setcbreak(fd)
        try:
            print("c=재기준(x y yaw_deg), q=정지·종료")
            result = run(runner, est, source, sender, key=key, ask_pose=ask_pose, trace=trace, max_s=a.max_s)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    save_trace(a.log, trace, {"pose_source": a.pose, "fake": a.fake, "route": a.route, "result": result})
    x, y, yaw = est.pose
    print(f"{result}: step={runner.label} pose=({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f}deg) trace={a.log}")
    return 0 if result == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())

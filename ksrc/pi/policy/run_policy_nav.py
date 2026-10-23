"""Run the exported Res8 student on the rover over SPI (structure B: firmware IK + residual).

run_policy_nav.py (10-18) = run_policy.py + 경사 시험용 옵션. 옵션을 안 주면 run_policy.py 와 똑같이 돈다.
  --crab PHI            명령 방향을 PHI 도 옆으로 비스듬히 (크랩, + = 왼쪽).  정책·IK(--residual-off) 둘 다
  --microzz PHI STEP    ±PHI 크랩을 번갈아: 조향이 새 각에 다 돈 뒤(--zz-settle) STEP m 가면 반대쪽 (멀리서 보면 직선)
  --zigzag PHI WIDTH    ±PHI 크랩, 출발선에서 옆으로 WIDTH m 벗어나면 반대쪽 (옆 위치 = VO 속도 + 자이로 방향 적분)
  --tcs K S0            정책 대신 IK + 헛돎 감속 (VO 헛돎)
  --wiper AMP PERIOD    정책 대신 IK + 네 바퀴 같은 방향 사인 조향
  python3 pi/policy/run_policy_nav.py --policy pi/policy/res8_mlp_v3vo_rc4000 --vo-port 5600 --cmd 0.15 0 0 --microzz 25 0.05

  # on the Pi, rover on blocks first (wheels free), then on the ground
  python3 pi/policy/run_policy.py --policy ~/policies/res8_mlp_h5 --cmd 0.10 0 0 --seconds 5
  python3 pi/policy/run_policy.py --policy ... --cmd 0.10 0 0 --residual-off      # pure IK via type 0x02
  python3 pi/policy/run_policy.py --policy ... --fake --seconds 2                  # laptop, no SPI
  # 10-18 경사 A/B: 정책 대신 스크립트 잔차 (IK + 헛돎 감속 / 와이퍼 조향). 정책 출력은 버림
  python3 pi/policy/run_policy.py --policy ... --cmd 0.15 0 0 --vo-port 5600 --tcs 3 0.3
  python3 pi/policy/run_policy.py --policy ... --cmd 0.15 0 0 --wiper 15 2

--policy is the export path without extension (<p>.pt TorchScript + <p>.json layout,
sim/isaac/ksrc_rover/scripts/export_student.py).  Teleop (V / SPI 0x01) is untouched: stop this
script and the old tools work as before; the firmware drops the residual 100 ms after the
last type-0x02 frame and stops 500 ms after the last command.

Loop (50 Hz, Pi clock):  telemetry of the previous exchange -> observation -> policy -> residual
-> SPI type 0x02 (vx, vy, w, d_steer, wheel_gain).  Before driving it polls 2 s standing still to
take the gyro bias (the student was trained with the bias the filter sees, minus nothing).
"""

from __future__ import annotations

import argparse
import os
import json
import math
import time
from pathlib import Path

import numpy as np

STAND_V, STAND_W = 0.01, 0.02      # [m/s], [rad/s] 이하면 정지 명령 -> 잔차 0 (ros/isaac_bridge.py 와 같은 값)
from student_runtime import C, FirmwareIK, StudentObs, action_to_residual, applied_for_obs, nl


class FakeLink:
    """No SPI: a rover that follows the command perfectly (for checking the loop on a laptop)."""

    def __init__(self):
        self.ik = FirmwareIK()
        self.goal, self.speed, self.t = np.zeros(4), np.zeros(4), 0

    def _tlm(self):
        self.t += 20
        return {"flags": nl.TLM_FLAG_IMU_VALID | nl.TLM_FLAG_RES_ACTIVE, "t_ms": self.t, "tick": self.t // 20 & 255,
                "gyro_dps": (0.0, 0.0, 0.0), "acc_g": (0.0, 0.0, 1.0), "steer_meas_rad": tuple(self.goal), "steer_cmd_rad": tuple(self.goal),
                "steer_speed": (0, 0, 0, 0), "wheel_mps": tuple(self.speed), "duty": (0, 0, 0, 0)}

    def exchange(self, vx, vy, w, poll_only=False):
        if not poll_only:
            self.goal, self.speed = self.ik.step(vx, vy, w)
        return self._tlm()

    def exchange_res(self, vx, vy, w, d_steer, wheel_gain):
        self.goal, self.speed = self.ik.step(vx, vy, w, d_steer, wheel_gain)
        return self._tlm()

    def close(self):
        pass


def slip_estimate(vo, gyro_z_rad, steer_rad, wheel_mps):
    """바퀴별 헛돎 추정 (10-18) = 시뮬 wheel_slip_ratio 와 같은 식: (표면 - 지면) / max(|표면|, |지면|, 0.05).
    지면 = VO 몸체 속도 (vx, vy) + 자이로 ω 를 바퀴 위치·조향각으로 옮긴 굴림 방향 속도.  + = 바퀴가 더 빨리 돎 (헛돎),
    - = 바퀴가 끌림.  VO 가 없으면 nan.  VO 는 나이(~230 ms)만큼 늦은 값이라 짧은 출렁임은 어긋날 수 있음."""
    if vo is None:
        return np.full(4, np.nan, np.float32)
    xy = np.asarray(C.MODULE_XY_M, np.float32)
    th = np.asarray(steer_rad, np.float32)
    v_long = (vo[0] - gyro_z_rad * xy[:, 1]) * np.cos(th) + (vo[1] + gyro_z_rad * xy[:, 0]) * np.sin(th)
    surf = np.asarray(wheel_mps, np.float32)
    return ((surf - v_long) / np.maximum(np.maximum(np.abs(surf), np.abs(v_long)), 0.05)).astype(np.float32)


class CrabNav:
    """10-18 경사 등판: 받은 명령 (vx, vy) 를 옆으로 PHI 만큼 돌려 보낸다 (ω 는 그대로, 속력 같음).
    crab: 한쪽으로 계속.  microzz: ±PHI 를 번갈아 -- Nucleo 가 보고하는 조향 실측이 목표(steer_cmd)의 settle 안에
    네 바퀴 다 들어온 **뒤부터** 이동 거리를 세고, step [m] 가면 반대쪽으로 꺾는다.
    (서보가 40° 도는 데 ~0.6 s, 0.15 m/s 로 4 cm 는 0.27 s 라 바로 세면 조향만 하다 끝남 -- MuJoCo 확인.)
    거리 = VO 몸체 속도 적분 (VO 없으면 그 스텝은 바퀴 속도 평균 -- 헛돌면 과대평가라 경고).
    조향 목표가 60° 넘게 바뀌면 펌웨어 정렬 게이트가 바퀴를 세우므로 microzz 는 |PHI| <= 29° 로 막는다."""

    def __init__(self, mode, phi_deg, step=0.05, settle_deg=3.0, settle_max_s=0.8, width=0.25):
        self.mode, self.phi, self.step, self.settle = mode, math.radians(phi_deg), step, math.radians(settle_deg)
        self.side = 1.0
        self.dist = 0.0
        self.settled = mode == "crab"
        self.age = 0                  # 꺾은 뒤 지난 스텝 (steer_cmd 가 새 목표로 바뀌기까지 1–2 스텝)
        self.switches = 0
        self.n_wheel_odom = 0
        self.settle_max_s = settle_max_s
        # zigzag: 출발 방향 기준 옆 위치 [m] (+ = 왼쪽). 몸체 방향 psi = 바이어스 뺀 자이로 적분
        self.width, self.lat, self.psi, self.n_cmd_odom = width, 0.0, 0.0, 0

    def __call__(self, base, tlm, vo, dt, gyro_z=0.0):
        vx, vy, w = base
        self.psi += gyro_z * dt
        if vo is not None and np.isfinite(vo[0]):
            bvx, bvy = vo[0], vo[1]
        else:                         # VO 없으면 지난 명령으로 (헛돌면 옆 위치 과대평가)
            bvx, bvy = getattr(self, "last", (0.0, 0.0))
            self.n_cmd_odom += 1
        self.lat += (bvx * math.sin(self.psi) + bvy * math.cos(self.psi)) * dt
        sp = math.hypot(vx, vy)
        if sp < STAND_V:              # 정지·제자리 회전 명령은 그대로
            return base
        if self.mode == "microzz":
            self.age += 1
            if not self.settled:
                cmd, meas = tlm.get("steer_cmd_rad"), tlm.get("steer_meas_rad")
                ok = self.age >= 3 and cmd is not None and meas is not None and \
                    np.max(np.abs(np.asarray(meas) - np.asarray(cmd))) < self.settle
                # 정책이 조향을 계속 흔들면(stv3 σ 4–10°) 3° 안에 거의 안 듦 (MuJoCo: 50 s 에 2–4 번만 꺾음)
                # -> settle_max_s 지나면 다 돈 것으로 (서보 50° ~0.65 s + 여유)
                if ok or self.age * dt >= self.settle_max_s:
                    self.settled, self.dist = True, 0.0
            else:
                if vo is not None and np.isfinite(vo[0]):
                    self.dist += math.hypot(vo[0], vo[1]) * dt
                else:
                    self.dist += float(np.mean(np.abs(tlm["wheel_mps"]))) * dt
                    self.n_wheel_odom += 1
                if self.dist >= self.step:
                    self.side, self.settled, self.age, self.switches = -self.side, False, 0, self.switches + 1
        if self.mode == "zigzag" and self.side * self.lat > self.width:
            self.side, self.switches = -self.side, self.switches + 1
        ang = math.atan2(vy, vx) + self.side * self.phi
        self.last = (sp * math.cos(ang), sp * math.sin(ang))
        return (self.last[0], self.last[1], w)


class CmdReceiver:
    """텔레옵 명령 UDP 수신 = udp_spi_bridge.py 와 같은 패킷 (<4sIfff: b"KSRC", seq, vx, vy, ω).
    쌓인 것 중 최신만, 순서 뒤집힌 패킷 버림, STALE_S 동안 안 오면 0.  보낸 쪽(--monitor)에 0.2 s 마다 요약 회신."""
    MAGIC, STALE_S = b"KSRC", 0.3

    def __init__(self, port, max_lin, max_ang):
        import socket
        import struct
        self.pkt = struct.Struct("<4sIfff")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.max_lin, self.max_ang = max_lin, max_ang
        self.cmd, self.last_rx, self.last_seq, self.peer = (0.0, 0.0, 0.0), -1e9, None, None
        self.n_rx = self.n_drop = self.n_stale = 0
        self.next_reply = 0.0

    def poll(self, now):
        while True:
            try:
                data, addr = self.sock.recvfrom(64)
            except BlockingIOError:
                break
            if len(data) != self.pkt.size:
                self.n_drop += 1
                continue
            magic, seq, vx, vy, w = self.pkt.unpack(data)
            if magic != self.MAGIC or not all(map(math.isfinite, (vx, vy, w))):
                self.n_drop += 1
                continue
            if self.last_seq is not None and addr == self.peer and 0 < (self.last_seq - seq) % 2**32 < 2**31:
                self.n_drop += 1
                continue
            self.last_seq, self.peer, self.last_rx = seq, addr, now
            c = lambda v, m: max(-m, min(m, v))
            self.cmd = (c(vx, self.max_lin), c(vy, self.max_lin), c(w, self.max_ang))
            self.n_rx += 1
        if now - self.last_rx > self.STALE_S:
            self.n_stale += 1
            return (0.0, 0.0, 0.0)
        return self.cmd

    def reply(self, now, tlm):
        if self.peer is None or now < self.next_reply:
            return
        self.next_reply = now + 0.2
        msg = {"cmd": [round(x, 3) for x in tlm.get("cmd", self.cmd)],
               "deg": [round(math.degrees(a), 1) for a in tlm.get("steer_cmd_rad", (0.0,) * 4)],
               "duty": list(tlm.get("duty", ())), "flags": tlm.get("flags", 0), "crc_bad": 0,
               "stale": now - self.last_rx > self.STALE_S}
        try:
            self.sock.sendto(json.dumps(msg).encode(), self.peer)
        except OSError:
            pass

    def close(self):
        self.sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="내보낸 정책 경로 (확장자 없이: <p>.pt, <p>.json)")
    ap.add_argument("--cmd", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "W"))
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--residual-off", action="store_true", help="잔차 0 으로 (타입 0x02 경로·관측만 확인)")
    ap.add_argument("--res-ema", type=float, default=1.0,
                    help="잔차 EMA α (1 = 끔). 0.7 = 10-16 배포안: 시뮬 이득 손해 없이 조향 떨림 -1/3, 지연 9 ms")
    ap.add_argument("--res-steer-deg", type=float, default=None,
                    help="조향 잔차 폭 [deg] 덮어쓰기. 없으면 학습값 (json res8_steer_scale_rad, 없으면 30). 학습값과 다르면 경고")
    ap.add_argument("--vo-port", type=int, default=None, metavar="PORT",
                    help="VO 몸체 속도 UDP 수신 (vo_relay.py, 기본 5600). student v3 처럼 관측에 vo 가 있는 정책용. "
                         "없으면 v3 도 'VO 없음' 값으로 돈다")
    ap.add_argument("--cmd-udp", type=int, nargs="?", const=5005, default=None, metavar="PORT",
                    help="명령을 UDP 로 받기 (텔레옵, 기본 5005). 노트북: teleop_joystick.py --udp raspberrypi.local "
                         "(udp_spi_bridge 와 같은 패킷). 0.3 s 안 오면 0 명령. 이때 --cmd 는 무시, --seconds 는 최대 시간")
    ap.add_argument("--max-lin", type=float, default=0.2, help="--cmd-udp 받은 vx, vy 상한 [m/s]")
    ap.add_argument("--max-ang", type=float, default=0.8, help="--cmd-udp 받은 ω 상한 [rad/s]")
    ap.add_argument("--crab", type=float, default=None, metavar="PHI_DEG",
                    help="명령 방향을 PHI 도 옆으로 (크랩, + = 왼쪽). 정책·IK 모두")
    ap.add_argument("--microzz", type=float, nargs=2, default=None, metavar=("PHI_DEG", "STEP_M"),
                    help="±PHI 크랩 번갈아: 조향 다 돈 뒤 STEP m 가면 반대쪽 (예: 25 0.05). 거리는 VO (--vo-port 권장)")
    ap.add_argument("--zigzag", type=float, nargs=2, default=None, metavar=("PHI_DEG", "WIDTH_M"),
                    help="±PHI 크랩, 출발선에서 옆으로 WIDTH m 벗어나면 반대쪽 (예: 20 0.25). 옆 위치 = VO + 자이로 (--vo-port 필요)")
    ap.add_argument("--zz-settle", type=float, default=3.0, help="--microzz: 조향 실측이 목표의 이 [deg] 안이면 다 돈 것")
    ap.add_argument("--zz-settle-max", type=float, default=0.8, help="--microzz: 꺾은 뒤 이 [s] 지나면 정렬 안 돼도 다 돈 것으로")
    ap.add_argument("--tcs", type=float, nargs=2, default=None, metavar=("K", "S0"),
                    help="정책 대신 IK + 헛돎 감속: 바퀴 배율 = 1 - clip(K*(헛돎-S0), 0, 0.6). 헛돎은 VO 추정 (--vo-port 필요)")
    ap.add_argument("--wiper", type=float, nargs=2, default=None, metavar=("AMP_DEG", "PERIOD_S"),
                    help="정책 대신 IK + 와이퍼: 네 바퀴 조향에 같은 방향 ±AMP_DEG 사인 (주기 PERIOD_S)")
    ap.add_argument("--fake", action="store_true", help="SPI 없이 (노트북에서 루프 확인)")
    ap.add_argument("--bias-s", type=float, default=2.0, help="시작 전 정지 자이로 바이어스 측정 시간")
    ap.add_argument("--log", default=None, help="npz 로 관측·행동·텔레메트리·루프 주기 저장")
    ap.add_argument("--rt", type=int, default=0, metavar="PRIO",
                    help="실시간 우선순위 (SCHED_FIFO, 예: 50). 권한 필요: sudo 또는 'sudo setcap cap_sys_nice+ep $(readlink -f $(which python3))'")
    ap.add_argument("--cpu", type=int, default=None, help="이 코어에만 (예: 3. ORB-SLAM 등과 코어 나누기)")
    ap.add_argument("--gc-off", action="store_true", help="루프 동안 파이썬 GC 끄기 (가끔 수 ms 멈춤 방지)")
    args = ap.parse_args()
    scripted = args.tcs is not None or args.wiper is not None
    if args.tcs is not None and args.vo_port is None and not args.fake:
        raise SystemExit("--tcs 는 헛돎을 VO 로 추정한다 -> --vo-port 필요 (ORB + vo_relay.py)")
    if args.wiper is not None and not 0.0 < args.wiper[0] <= 30.0:
        raise SystemExit("--wiper 폭은 0 < AMP <= 30 deg (조향 잔차 한계)")
    if sum(x is not None for x in (args.crab, args.microzz, args.zigzag)) > 1:
        raise SystemExit("--crab / --microzz / --zigzag 는 하나만")
    if args.zigzag is not None and args.vo_port is None and not args.fake:
        raise SystemExit("--zigzag 은 옆 위치를 VO 로 센다 -> --vo-port 필요 (ORB + vo_relay.py)")
    if args.zigzag is not None and abs(args.zigzag[0]) > 29.0:
        print("[run] 주의: --zigzag 각 > 29° -> 꺾을 때 펌웨어 정렬 게이트가 바퀴를 잠깐 세움 (MuJoCo 30°: 12% 시간)", flush=True)
    if args.microzz is not None and abs(args.microzz[0]) > 29.0:
        raise SystemExit("--microzz 각이 29° 넘으면 꺾을 때 조향 목표가 60° 바뀌어 펌웨어 정렬 게이트가 바퀴를 세움")
    if args.microzz is not None and args.vo_port is None and not args.fake:
        print("[run] 경고: --microzz 거리를 VO 없이 바퀴 속도로 셈 -- 헛돌면 실제보다 빨리 꺾는다 (--vo-port 권장)", flush=True)
    if scripted and args.residual_off:
        raise SystemExit("--residual-off 와 --tcs/--wiper 는 같이 쓸 수 없음")
    if args.log:                     # 10-18: 폴더가 없어 주행 끝에 저장 실패한 적 있음 -> 시작 전에 만들고 확인
        d = os.path.dirname(os.path.abspath(args.log))
        os.makedirs(d, exist_ok=True)
        if not os.access(d, os.W_OK):
            raise SystemExit(f"로그 폴더에 쓸 수 없음: {d}")

    import gc
    # 코어 고정과 스레드 수는 torch import **전에** (10-16 Pi 실측): import 뒤에 고정하면 이미 만든 OpenMP 스레드가
    # 한 코어에서 서로 기다려 정책 한 번이 0.5 -> 11 ms 로 늘었다.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
        print(f"[run] 코어 {args.cpu} 에 고정", flush=True)
    import torch
    torch.set_num_threads(1)                 # 작은 MLP: 스레드 여럿은 오히려 흔들림만 늘림
    if args.rt:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(args.rt))
            print(f"[run] 실시간 우선순위 SCHED_FIFO {args.rt}", flush=True)
        except PermissionError:
            print("[run] 실시간 우선순위 권한 없음 (--rt 무시). sudo 로 실행하거나 setcap 필요", flush=True)
    meta = json.load(open(args.policy + ".json"))
    if meta["action_dim"] != 8:
        raise SystemExit(f"Res8 정책이 아님 (action_dim {meta['action_dim']})")
    pol = torch.jit.load(args.policy + ".pt", map_location="cpu").eval()
    h = torch.zeros(*meta["hidden_shape"]) if meta.get("recurrent") else None
    with torch.no_grad():                     # 첫 호출은 TorchScript 최적화로 수십 ms -> 루프 전에 한 번
        x0 = torch.zeros(1, int(meta["obs_dim"]))
        for _ in range(3):
            pol(x0) if h is None else pol(x0, h)
    dt = float(meta.get("control_dt_s", 0.02))
    rate = float(meta.get("res8_rate_limit", 0.0))
    a_prev = np.zeros(8, np.float32)
    ema = float(args.res_ema)
    if not 0.0 < ema <= 1.0:
        raise SystemExit("--res-ema 는 0 < α <= 1")
    # 조향 잔차 폭: --res-steer-deg 가 없으면 정책이 학습한 값 (10-18: json res8_steer_scale_rad, 15° 정책은 15°.
    #   옛 json 에는 없음 -> 학습값 30°).  학습값과 다르게 덮어쓰면 경고 (정책 출력 의미가 바뀜)
    k_train = meta.get("res8_steer_scale_rad")
    steer_k = math.radians(args.res_steer_deg) if args.res_steer_deg is not None else k_train
    if args.res_steer_deg is not None and abs(math.radians(args.res_steer_deg) - (k_train or math.radians(30.0))) > 1e-6:
        print(f"[run] 주의: 조향 잔차 폭을 학습값 {math.degrees(k_train or math.radians(30.0)):.0f}° 대신 "
              f"{args.res_steer_deg:g}° 로 덮어씀", flush=True)
    print(f"[run] 잔차 후처리: 변화율 제한 {rate or '없음'}/스텝, 정지 명령 잔차 0, "
          f"EMA α {ema if ema < 1 else '끔'}, 조향 잔차 폭 {math.degrees(steer_k) if steer_k is not None else 30:.0f}°", flush=True)
    obs_b = StudentObs(meta, dt)
    has_vo = any(t["name"] == "vo" for t in meta["terms"])
    vo_rx = None
    if args.vo_port is not None:
        if not has_vo:
            print("[run] 경고: 이 정책은 관측에 vo 가 없음 (v1 등) -- --vo-port 는 기록만 한다", flush=True)
        from vo_link import VoReceiver
        vo_rx = VoReceiver(args.vo_port)
        print(f"[run] VO 수신 UDP :{args.vo_port} (vo_relay.py)", flush=True)
    elif has_vo:
        print("[run] 정책에 vo 관측이 있지만 --vo-port 없음 -> 'VO 없음' 값으로 실행 (학습의 VO 끊김과 같음)", flush=True)
    cmd_rx = None
    if args.cmd_udp is not None:
        cmd_rx = CmdReceiver(args.cmd_udp, args.max_lin, args.max_ang)
        print(f"[run] 텔레옵 명령 UDP :{args.cmd_udp} (상한 {args.max_lin} m/s, {args.max_ang} rad/s, "
              f"{CmdReceiver.STALE_S} s 끊기면 0). 최대 {args.seconds:.0f} s, Ctrl+C 로 끝", flush=True)
    if scripted:
        print("[run] 스크립트 잔차 모드 (정책 출력 버림): "
              + (f"헛돎 감속 K {args.tcs[0]:g}, 문턱 {args.tcs[1]:g} (VO 헛돎 > 문턱이면 그 바퀴 늦춤, 최저 배율 0.4)  "
                 if args.tcs is not None else "")
              + (f"와이퍼 ±{args.wiper[0]:g}°, 주기 {args.wiper[1]:g} s (네 바퀴 같은 방향)" if args.wiper is not None else ""),
              flush=True)
    nav = None
    if args.crab is not None:
        nav = CrabNav("crab", args.crab)
        print(f"[run] 크랩: 명령 방향을 {args.crab:+g}° 옆으로 (+ = 왼쪽)", flush=True)
    elif args.zigzag is not None:
        nav = CrabNav("zigzag", args.zigzag[0], width=args.zigzag[1])
        print(f"[run] 지그재그: ±{args.zigzag[0]:g}°, 출발선에서 옆으로 ±{args.zigzag[1]:g} m 넘으면 반대쪽", flush=True)
    elif args.microzz is not None:
        nav = CrabNav("microzz", args.microzz[0], args.microzz[1], args.zz_settle, args.zz_settle_max)
        print(f"[run] 짧은 지그재그: ±{args.microzz[0]:g}°, 조향이 {args.zz_settle:g}° 안에 든 뒤 "
              f"{100 * args.microzz[1]:g} cm 마다 반대쪽", flush=True)
    ik = FirmwareIK()
    link = FakeLink() if args.fake else nl.SpiLink()

    # 1) standing still: gyro bias (POLL frames do not feed the Nucleo watchdog)
    g = []
    t_end = time.monotonic() + args.bias_s
    while time.monotonic() < t_end:
        tlm = link.exchange(0.0, 0.0, 0.0, poll_only=True)
        if tlm is not None and tlm["flags"] & nl.TLM_FLAG_IMU_VALID:
            g.append(np.asarray(tlm["gyro_dps"], float) * math.pi / 180.0)
        time.sleep(dt)
    if not g:
        raise SystemExit("IMU 값이 안 들어옴 (TLM_FLAG_IMU_VALID) -- 배선·펌웨어 확인")
    bias = np.mean(g, 0)
    print(f"[run] 자이로 바이어스 {np.round(bias, 4)} rad/s ({len(g)} 샘플)", flush=True)

    # 2) Pi IK starts from the angles the Nucleo already holds (else applied_targets is wrong until the first move)
    t_sync = tlm
    for _ in range(10):
        if t_sync is not None and "steer_cmd_rad" in t_sync:
            break
        t_sync = link.exchange(0.0, 0.0, 0.0, poll_only=True)
        time.sleep(dt)
    if t_sync is not None and "steer_cmd_rad" in t_sync:
        ik.sync(t_sync["steer_cmd_rad"])
        print(f"[run] IK 시작 조향각 = Nucleo 유지값 {np.round(np.degrees(t_sync['steer_cmd_rad']), 1)} deg", flush=True)
    else:
        print("[run] 경고: steer_cmd_rad 못 받음 -- IK 상태 0 에서 시작 (첫 움직임 전까지 applied_targets 틀릴 수 있음)", flush=True)

    vx, vy, w = args.cmd
    base = tuple(args.cmd)           # 받은 명령 (--crab/--microzz 는 이걸 돌려 vx, vy 로)
    vo = None
    tlm_ns = time.time_ns()
    applied = applied_for_obs(*ik.step(0.0, 0.0, 0.0), None)
    # 원시 텔레메트리도 저장 (10-15): 속도 추정기·duty–슬립 관계를 17일 로그로 오프라인 재생해 설계하려고.
    #   host_ns = 텔레메트리를 받은 Pi 시각 (time.time_ns, 카메라·ORB 기록과 맞추는 기준), cmd = 보낸 (vx, vy, w)
    log = {k: [] for k in ("t_ms", "host_ns", "cmd", "obs", "action", "flags", "duty", "wheel_mps", "steer_meas", "steer_cmd",
                           "steer_speed", "steer_load", "steer_volt", "gyro_dps", "acc_g", "imu_n", "wpi_corr", "tick", "vo", "slip",
                           # 원시값 (10-18): 엔코더 누적 틱, IMU 칩 좌표 원시 LSB, Nucleo 가 실제 적용한 명령, 서보 온도·상태·유효,
                           # SPI 수신 오류 수, 보낸 명령 순번 되돌림, IMU 평균 시각 offset [us] (bit12 펌웨어)
                           "enc_ticks", "imu_raw", "tlm_cmd", "steer_temp", "steer_status", "servo_valid", "rx_err",
                           "seq_echo", "imu_offset_us", "d_steer", "gain",
                           "cmd_base", "nav_side", "nav_dist", "nav_settled", "nav_lat")}
    n_bad = 0
    t_start, t_comp, n_late, n_resync = [], [], 0, 0
    if args.gc_off:
        gc.collect()
        gc.disable()
    next_t = time.monotonic()
    try:
        for k in range(int(args.seconds / dt)):
            t0 = time.monotonic()
            t_start.append(t0)
            if t0 - next_t > 0.005:
                n_late += 1
            if tlm is None:
                n_bad += 1
                tlm = link.exchange(vx, vy, w, poll_only=True)
                tlm_ns = time.time_ns()
                continue
            if cmd_rx is not None:
                base = cmd_rx.poll(t0)
                cmd_rx.reply(t0, tlm)
            vo = vo_rx.poll() if vo_rx is not None else None
            gz = math.radians(tlm.get("gyro_dps", (0, 0, 0))[2]) - bias[2]
            vx, vy, w = nav(base, tlm, vo, dt, gz) if nav is not None else base
            slip = slip_estimate(vo, math.radians(tlm.get("gyro_dps", (0, 0, 0))[2]) - bias[2], tlm["steer_meas_rad"], tlm["wheel_mps"])
            obs = obs_b.push(obs_b.frame((vx, vy, w), tlm, bias, applied, vo))
            with torch.no_grad():
                x = torch.as_tensor(obs)
                if h is None:
                    a = pol(x)
                else:
                    a, h = pol(x, h)
            a = a[0].numpy()
            # 학습과 같은 잔차 후처리 (10-15, json res8_*): 정지 명령이면 0, 그리고 변화율 제한.
            # 옛 정책(json 에 없음)도 정지 명령 잔차 0 은 적용 (서 있는 동안 조향 떨림 방지).
            standing = abs(vx) < STAND_V and abs(vy) < STAND_V and abs(w) < STAND_W
            if standing:
                a = np.zeros_like(a)
            elif ema < 1.0:          # Isaac Res8 ema 와 같은 식·순서 (정지 0 은 즉시, 그 다음 변화율 제한)
                a = a_prev + ema * (a - a_prev)
            if rate > 0:
                a = a_prev + np.clip(a - a_prev, -rate, rate)
            a_prev = a.copy()
            d_steer, gain = action_to_residual(a, steer_k)
            if args.residual_off:
                d_steer, gain = np.zeros(4, np.float32), np.ones(4, np.float32)
            if scripted:             # 10-18 경사 A/B: 정책 대신 스크립트 (정지 명령이면 0)
                d_steer, gain = np.zeros(4, np.float32), np.ones(4, np.float32)
                if not standing:
                    if args.tcs is not None and np.isfinite(slip).all():   # VO 없으면 감속 안 함 (IK 그대로)
                        cut = np.clip(args.tcs[0] * (np.maximum(slip, 0.0) - args.tcs[1]), 0.0, C.RES8_WHEEL_SCALE)
                        gain = (1.0 - cut).astype(np.float32)
                    if args.wiper is not None:
                        ph = 2.0 * math.pi * (t0 - t_start[0]) / args.wiper[1]
                        d_steer = np.full(4, math.radians(args.wiper[0]) * math.sin(ph), np.float32)
            tlm_next = link.exchange_res(vx, vy, w, d_steer.tolist(), gain.tolist())
            tlm_next_ns = time.time_ns()
            t_comp.append(time.monotonic() - t0)
            # what the Nucleo applies from this command on = the next frame's applied_targets
            applied = applied_for_obs(*ik.step(vx, vy, w, d_steer, gain), tlm_next)
            for key, v in (("t_ms", tlm["t_ms"]), ("host_ns", tlm_ns), ("cmd", (vx, vy, w)), ("obs", obs[0]), ("action", a),
                           ("flags", tlm["flags"]), ("duty", tlm["duty"]), ("wheel_mps", tlm["wheel_mps"]),
                           ("steer_meas", tlm["steer_meas_rad"]), ("steer_cmd", tlm.get("steer_cmd_rad", (0,) * 4)),
                           ("steer_speed", tlm.get("steer_speed", (0,) * 4)), ("steer_load", tlm.get("steer_load", (0,) * 4)),
                           ("steer_volt", tlm.get("steer_volt", (0,) * 4)), ("gyro_dps", tlm.get("gyro_dps", (0,) * 3)),
                           ("acc_g", tlm.get("acc_g", (0,) * 3)), ("imu_n", tlm.get("imu_n", -1)),
                           ("wpi_corr", tlm.get("wpi_corr", -1)), ("tick", tlm.get("tick", -1)),
                           ("vo", vo if vo is not None else (np.nan, np.nan, np.nan)), ("slip", slip),
                           ("enc_ticks", tlm.get("enc_ticks", (0,) * 4)), ("imu_raw", tlm.get("imu_raw", (0,) * 6)),
                           ("tlm_cmd", tlm.get("cmd", (np.nan,) * 3)), ("steer_temp", tlm.get("steer_temp", (0,) * 4)),
                           ("steer_status", tlm.get("steer_status", (0,) * 4)), ("servo_valid", tlm.get("servo_valid", (False,) * 4)),
                           ("rx_err", tlm.get("rx_err", -1)), ("d_steer", d_steer), ("gain", gain),
                           ("cmd_base", base), ("nav_side", 0.0 if nav is None else nav.side),
                           ("nav_dist", 0.0 if nav is None else nav.dist), ("nav_settled", nav is not None and nav.settled),
                           ("nav_lat", 0.0 if nav is None else nav.lat), ("seq_echo", tlm.get("seq_echo", -1)),
                           ("imu_offset_us", np.nan if tlm.get("imu_mean_read_offset_us") is None else tlm["imu_mean_read_offset_us"])):
                log[key].append(np.asarray(v))
            if k % 25 == 0:
                fl = tlm["flags"]
                win = np.diff(t_start[-26:]) * 1000 if len(t_start) > 1 else np.zeros(1)
                print(f"[run] t={tlm['t_ms']} 잔차 {'ON' if fl & nl.TLM_FLAG_RES_ACTIVE else 'off'} "
                      f"게이트 {'보류' if fl & nl.TLM_FLAG_GATE_PENDING else '-'} "
                      f"주기 최대 {win.max():.1f} ms  계산 {t_comp[-1] * 1000:.1f} ms  "
                      f"a={np.round(a, 2)} duty={tlm['duty']}"
                      + (f" VO {'없음' if vo is None else f'({vo[0]:+.2f}, {vo[1]:+.2f}) {vo[2] * 1000:.0f} ms'}"
                         if vo_rx is not None else "")
                      + (f" 헛돎 {np.array2string(np.round(slip * 100).astype(int))}%" if np.isfinite(slip).all() else "")
                      + (f" 크랩 {math.degrees(math.atan2(vy, vx)):+.0f}°"
                         + (f" {'정렬' if nav.settled else '조향중'} {100 * nav.dist:.0f}cm 꺾음 {nav.switches}" if nav.mode == "microzz" else "")
                         + (f" 옆 {100 * nav.lat:+.0f}cm 꺾음 {nav.switches}" if nav.mode == "zigzag" else "")
                         if nav is not None else "")
                      + (f" 배율 {np.round(gain, 2)} 와이퍼 {math.degrees(d_steer[0]):+.0f}° (a= 는 버린 정책 출력)" if scripted else ""), flush=True)
            tlm, tlm_ns = tlm_next, tlm_next_ns
            next_t += dt
            if time.monotonic() - next_t > dt:      # 한 주기 넘게 밀렸으면 몰아서 따라잡지 않고 다시 맞춤
                next_t = time.monotonic()
                n_resync += 1
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:        # 텔레옵은 Ctrl+C 로 끝냄 -> 멈춤·요약·로그 저장은 그대로
        print("\n[run] Ctrl+C -- 멈추고 저장", flush=True)
    finally:
        for _ in range(5):      # stop: plain velocity command 0 (also clears the residual in firmware)
            link.exchange(0.0, 0.0, 0.0)
            time.sleep(dt)
        link.close()
        if vo_rx is not None:
            dur = (t_start[-1] - t_start[0]) if len(t_start) > 1 else float("nan")
            print(f"[run] VO 패킷 {vo_rx.n_rx} 개 (깨짐 {vo_rx.n_bad}), 유효 {vo_rx.n_valid / dur:.1f} Hz", flush=True)
            if log["vo"]:
                age = np.asarray(log["vo"])[:, 2] * 1000
                ok = age[np.isfinite(age)]
                miss = 100.0 * (1 - len(ok) / len(age))
                print(f"[run] VO 나이 [ms]: " + (f"평균 {ok.mean():.0f}  p95 {np.quantile(ok, .95):.0f}  최대 {ok.max():.0f}  "
                      if len(ok) else "") + f"VO 없음 {miss:.1f}% 스텝", flush=True)
            if log["slip"]:
                sl = np.asarray(log["slip"]); mv = np.abs(np.asarray(log["cmd"])[:, 0]) > 0.02
                ok = np.isfinite(sl).all(1) & mv
                if ok.any():
                    print(f"[run] 헛돎 (움직일 때, VO 기준) 평균 {np.round(sl[ok].mean(0) * 100, 1)}%  "
                          f"p95 {np.round(np.quantile(sl[ok], .95, axis=0) * 100, 1)}%  "
                          f"30% 넘은 스텝 {100 * (sl[ok] > 0.3).any(1).mean():.1f}%  (바퀴 순서 FL FR RL RR)", flush=True)
            vo_rx.close()
        if cmd_rx is not None:
            print(f"[run] 텔레옵 패킷 {cmd_rx.n_rx} 개 (버림 {cmd_rx.n_drop}), 끊겨서 0 명령 {cmd_rx.n_stale} 스텝", flush=True)
            cmd_rx.close()
    if nav is not None and nav.mode == "zigzag":
        print(f"[run] 지그재그: 꺾음 {nav.switches}회, 마지막 옆 위치 {100 * nav.lat:+.0f} cm, "
              f"VO 없어 명령으로 센 스텝 {nav.n_cmd_odom}", flush=True)
    if nav is not None and nav.mode == "microzz":
        print(f"[run] 짧은 지그재그: 꺾음 {nav.switches}회, VO 없어 바퀴 속도로 센 스텝 {nav.n_wheel_odom}", flush=True)
    if args.gc_off:
        gc.enable()
    print(f"[run] 끝. 텔레메트리 실패 {n_bad} 회", flush=True)
    per = np.diff(np.asarray(t_start)) * 1000 if len(t_start) > 1 else np.zeros(1)
    comp = np.asarray(t_comp) * 1000 if t_comp else np.zeros(1)
    q = lambda x, p: float(np.quantile(x, p))
    print(f"[run] 루프 주기 [ms] (목표 {dt * 1000:.0f}): 평균 {per.mean():.2f}  중앙 {q(per, .5):.2f}  p95 {q(per, .95):.2f}  "
          f"p99 {q(per, .99):.2f}  최대 {per.max():.2f}  |  25 ms 넘음 {int((per > 25).sum())}회, "
          f"40 ms 넘음 {int((per > 40).sum())}회 / {len(per)}", flush=True)
    print(f"[run] 한 주기 계산 (관측+정책+SPI) [ms]: 중앙 {q(comp, .5):.2f}  p99 {q(comp, .99):.2f}  최대 {comp.max():.2f}  |  "
          f"5 ms 넘게 늦게 시작 {n_late}회, 한 주기 넘게 밀려 재정렬 {n_resync}회", flush=True)
    if args.log:
        log["loop_t"], log["period_ms"], log["compute_ms"] = [t_start], [per], [comp]
        np.savez_compressed(args.log, meta=json.dumps(meta), res_ema=ema, gyro_bias_rad=bias,
                            args=json.dumps(vars(args)),
                            res_steer_deg=np.nan if args.res_steer_deg is None else args.res_steer_deg, **{k: np.asarray(v[0] if k in ("loop_t", "period_ms", "compute_ms") else v)
                                                                 for k, v in log.items()})
        print(f"[run] 저장 {args.log}", flush=True)


if __name__ == "__main__":
    main()

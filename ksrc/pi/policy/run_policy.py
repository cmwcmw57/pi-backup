"""Run the exported Res8 student on the rover over SPI (structure B: firmware IK + residual).

  # on the Pi, rover on blocks first (wheels free), then on the ground
  python3 pi/policy/run_policy.py --policy ~/policies/res8_mlp_h5 --cmd 0.10 0 0 --seconds 5
  python3 pi/policy/run_policy.py --policy ... --cmd 0.10 0 0 --residual-off      # pure IK via type 0x02
  python3 pi/policy/run_policy.py --policy ... --fake --seconds 2                  # laptop, no SPI

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
    ap.add_argument("--fake", action="store_true", help="SPI 없이 (노트북에서 루프 확인)")
    ap.add_argument("--bias-s", type=float, default=2.0, help="시작 전 정지 자이로 바이어스 측정 시간")
    ap.add_argument("--log", default=None, help="npz 로 관측·행동·텔레메트리·루프 주기 저장")
    ap.add_argument("--rt", type=int, default=0, metavar="PRIO",
                    help="실시간 우선순위 (SCHED_FIFO, 예: 50). 권한 필요: sudo 또는 'sudo setcap cap_sys_nice+ep $(readlink -f $(which python3))'")
    ap.add_argument("--cpu", type=int, default=None, help="이 코어에만 (예: 3. ORB-SLAM 등과 코어 나누기)")
    ap.add_argument("--gc-off", action="store_true", help="루프 동안 파이썬 GC 끄기 (가끔 수 ms 멈춤 방지)")
    args = ap.parse_args()
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
    tlm_ns = time.time_ns()
    applied = applied_for_obs(*ik.step(0.0, 0.0, 0.0), None)
    # 원시 텔레메트리도 저장 (10-15): 속도 추정기·duty–슬립 관계를 17일 로그로 오프라인 재생해 설계하려고.
    #   host_ns = 텔레메트리를 받은 Pi 시각 (time.time_ns, 카메라·ORB 기록과 맞추는 기준), cmd = 보낸 (vx, vy, w)
    log = {k: [] for k in ("t_ms", "host_ns", "cmd", "obs", "action", "flags", "duty", "wheel_mps", "steer_meas", "steer_cmd",
                           "steer_speed", "steer_load", "steer_volt", "gyro_dps", "acc_g", "imu_n", "wpi_corr", "tick", "vo", "slip",
                           # 원시값 (10-18): 엔코더 누적 틱, IMU 칩 좌표 원시 LSB, Nucleo 가 실제 적용한 명령, 서보 온도·상태·유효,
                           # SPI 수신 오류 수, 보낸 명령 순번 되돌림, IMU 평균 시각 offset [us] (bit12 펌웨어)
                           "enc_ticks", "imu_raw", "tlm_cmd", "steer_temp", "steer_status", "servo_valid", "rx_err",
                           "seq_echo", "imu_offset_us")}
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
                vx, vy, w = cmd_rx.poll(t0)
                cmd_rx.reply(t0, tlm)
            vo = vo_rx.poll() if vo_rx is not None else None
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
                           ("rx_err", tlm.get("rx_err", -1)), ("seq_echo", tlm.get("seq_echo", -1)),
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
                      + (f" 헛돎 {np.array2string(np.round(slip * 100).astype(int))}%" if np.isfinite(slip).all() else ""), flush=True)
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

# ================================================================
# open_loop.py
# 설명: RouteRunner를 이상적 운동학으로 한 번 돌려 시간별 (vx, vy, w) 명령표(CSV)를 만들고, 그 표를 위치 피드백 없이 50 Hz로 Pi run_policy(UDP, KSRC 패킷)에 재생한다. 실제 경로는 사후에 ORB/ESKF로 추정한다. 모터·SPI는 직접 쓰지 않는다.
# 사용법:
#   python -m experiments.yeouido_real.open_loop make yeouido_open_loop.csv
#   python -m experiments.yeouido_real.open_loop send yeouido_open_loop.csv --to raspberrypi.local:5005   # Ctrl-C로 즉시 정지
# ================================================================
import argparse
import csv
import math
import time
from pathlib import Path

from experiments.orbslam3_ros2.yeouido_route import RouteRunner
from experiments.yeouido_real.command_sender import CommandSender

PLAN = Path(__file__).resolve().parents[2] / "docs" / "yeouido_route_plan.json"
DT = 0.02
LEAD_S = 1.0  # 시작 전 0 명령 구간


def make_schedule(route: Path = PLAN, dt: float = DT) -> list[tuple[float, float, float, float, str]]:
    """이상적 추종 가정으로 경로를 돌려 (t, vx, vy, w, label) 목록을 만든다. 끝에는 0 명령 1 s."""
    runner = RouteRunner.from_json(route)
    spec = runner_spec(route)
    x, y, _ = spec["start"]["base_link_xyz_m"]
    yaw = math.radians(spec["start"]["yaw_deg"])
    rows = [(i * dt, 0.0, 0.0, 0.0, "start") for i in range(int(LEAD_S / dt))]
    t = LEAD_S
    while not runner.complete and t < 3000.0:
        cmd = runner.command(elapsed_s=t - LEAD_S, x=x, y=y, yaw_rad=yaw)
        rows.append((t, cmd.vx, cmd.vy, cmd.yaw_rate, runner.label))
        x += (math.cos(yaw) * cmd.vx - math.sin(yaw) * cmd.vy) * dt
        y += (math.sin(yaw) * cmd.vx + math.cos(yaw) * cmd.vy) * dt
        yaw += cmd.yaw_rate * dt
        t += dt
    rows += [(t + i * dt, 0.0, 0.0, 0.0, "end") for i in range(int(1.0 / dt))]
    return rows


def runner_spec(route: Path) -> dict:
    """경로 JSON을 읽어 반환한다."""
    import json
    return json.loads(Path(route).read_text(encoding="utf-8"))


def write_csv(rows: list[tuple[float, float, float, float, str]], path: Path) -> None:
    """명령표를 t,vx,vy,w,label CSV로 저장한다."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "vx", "vy", "w", "label"])
        w.writerows((f"{t:.3f}", f"{vx:.5f}", f"{vy:.5f}", f"{wz:.5f}", lab) for t, vx, vy, wz, lab in rows)


def read_csv(path: Path) -> list[tuple[float, float, float, float, str]]:
    """write_csv가 만든 CSV를 읽는다."""
    with open(path, newline="", encoding="utf-8") as f:
        return [(float(r["t"]), float(r["vx"]), float(r["vy"]), float(r["w"]), r["label"]) for r in csv.DictReader(f)]


def send(rows: list[tuple[float, float, float, float, str]], sender: CommandSender, clock=time.monotonic, sleep=time.sleep) -> None:
    """명령표를 표의 시각에 맞춰 보낸다. 중단(Ctrl-C)이나 끝에서 0 명령을 5번 보낸다."""
    t0, label = clock(), None
    try:
        for t, vx, vy, wz, lab in rows:
            while clock() - t0 < t:
                sleep(0.001)
            sender.send(vx, vy, wz)
            if lab != label:
                label = lab
                print(f"[{t:7.2f}s] {lab}", flush=True)
    finally:
        for _ in range(5):
            sender.send(0.0, 0.0, 0.0)
            sleep(0.025)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("csv", type=Path)
    m.add_argument("--route", type=Path, default=PLAN)
    s = sub.add_parser("send")
    s.add_argument("csv", type=Path)
    s.add_argument("--to", default="127.0.0.1:5005", help="Pi run_policy --cmd-udp HOST:PORT")
    a = ap.parse_args()
    if a.cmd == "make":
        rows = make_schedule(a.route)
        write_csv(rows, a.csv)
        print(f"{len(rows)} rows, {rows[-1][0]:.1f} s -> {a.csv}")
    else:
        host, port = a.to.rsplit(":", 1)
        sender = CommandSender(host, int(port))
        try:
            send(read_csv(a.csv), sender)
        except KeyboardInterrupt:
            print("정지", flush=True)
        finally:
            sender.close()


if __name__ == "__main__":
    main()

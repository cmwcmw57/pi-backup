# ================================================================
# test_open_loop.py
# 설명: open_loop 명령표 생성이 경로를 재현하는지, 재생이 UDP로 올바른 값과 마지막 0 명령을 보내는지 검증한다.
# 사용법: python3 -m pytest tests/yeouido_real/test_open_loop.py
# ================================================================
import math
import time

from experiments.yeouido_real.command_sender import CommandReceiver, CommandSender
from experiments.yeouido_real.open_loop import DT, make_schedule, read_csv, send, write_csv


def integrate(rows):
    """명령표를 이상적으로 적분해 최종 (x, y, yaw)를 돌려준다."""
    x, y, yaw = 0.29327678, 0.32020833, 0.0
    for _, vx, vy, w, _ in rows:
        x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * DT
        y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * DT
        yaw += w * DT
    return x, y, yaw


def test_schedule_replays_to_route_end_and_stays_within_limits():
    rows = make_schedule()
    x, y, yaw = integrate(rows)
    assert math.hypot(x - 2.2, y - 2.06) < 0.1 and abs(yaw) < math.radians(8)
    assert max(abs(r[1]) for r in rows) <= 0.1501 and max(abs(r[3]) for r in rows) <= 0.4001
    assert rows[0][4] == "start" and rows[-1][4] == "end" and rows[-1][1:4] == (0.0, 0.0, 0.0)
    labels = [r[4] for r in rows]
    assert [l for l in dict.fromkeys(labels)][1:-1][:3] == ["A 램프 오르기", "A 위에서 제자리 회전", "A 내려와 B 입구로 이동"]


def test_csv_roundtrip(tmp_path):
    rows = make_schedule()
    write_csv(rows, tmp_path / "s.csv")
    back = read_csv(tmp_path / "s.csv")
    assert len(back) == len(rows) and back[100][4] == rows[100][4]
    assert abs(back[100][1] - rows[100][1]) < 1e-5


def test_send_streams_values_and_ends_with_zero():
    rx = CommandReceiver(0, host="127.0.0.1")
    tx = CommandSender("127.0.0.1", rx.port, max_hz=1000.0)
    now = [0.0]
    seen = []

    def clock():
        return now[0]

    def sleep(dt):
        now[0] += max(dt, 0.001)
        time.sleep(0.002)  # 송신기 주기 제한은 실제 시계를 쓴다
        seen.append(rx.latest())

    rows = [(i * 0.02, 0.1, 0.0, 0.2, "x") for i in range(10)]
    send(rows, tx, clock=clock, sleep=sleep)
    tx.close()
    assert any(abs(v[0] - 0.1) < 1e-6 and abs(v[2] - 0.2) < 1e-6 for v in seen)
    time.sleep(0.01)
    assert rx.latest() == (0.0, 0.0, 0.0)

"""vo_link 시험 (10-16): 자세 -> 몸체 속도, 패킷, 수신기.   python3 pi/policy/test_vo_link.py"""
import math
import socket
import time

import numpy as np

from vo_link import PoseToVelocity, VoReceiver, pack, unpack


def yaw_q(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def run(est, poses):
    out = None
    for p in poses:
        r = est.add(*p)
        out = r if r is not None else out
    return out


def check(name, cond, info=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {info}" if info else ""))
    return cond


ok = True
# 1. 지도 좌표가 몸체와 30° 돌아가 있어도 직진 0.15 m/s 는 몸체 (0.15, 0)
yaw = math.radians(30)
poses = [(t, (0.15 * t * math.cos(yaw), 0.15 * t * math.sin(yaw), 0.0), yaw_q(yaw), 0) for t in np.arange(0, 1, 1 / 15)]
r = run(PoseToVelocity(), poses)
ok &= check("돌아간 지도에서 직진", r and abs(r[1] - 0.15) < 1e-6 and abs(r[2]) < 1e-6, f"{r}")
# 2. 옆걸음 (몸체 +y 0.1 m/s), 지도 yaw 90°
yaw = math.radians(90)
poses = [(t, (-0.1 * t, 0.0, 0.0), yaw_q(yaw), 0) for t in np.arange(0, 1, 1 / 15)]
r = run(PoseToVelocity(), poses)
ok &= check("옆걸음 vy", r and abs(r[1]) < 1e-6 and abs(r[2] - 0.1) < 1e-6, f"{r}")
# 3. 측정 시각 = 창 가운데 (지연을 age 에 반영)
ok &= check("측정 시각 = 창 가운데", r and 0.75 < r[0] < 0.9, f"t={r[0]:.3f} (마지막 0.933)")
# 4. 잡음 1 cm -> 속도 오차 작음
rng = np.random.default_rng(0)
poses = [(t, (0.15 * t + rng.normal(0, 0.01), rng.normal(0, 0.01), 0.0), yaw_q(0), 0) for t in np.arange(0, 3, 1 / 15)]
est = PoseToVelocity(); vs = [r for p in poses if (r := est.add(*p))]
e = np.array([v[1] - 0.15 for v in vs])
ok &= check("자세 잡음 1 cm -> 속도 std", e.std() < 0.08, f"{e.std():.3f} m/s (학습 VO σ 0.02-0.08)")
# 5. 지도 번호 바뀌면 이력 비움 (튐 없음)
est = PoseToVelocity()
run(est, [(t, (0.15 * t, 0, 0), yaw_q(0), 0) for t in np.arange(0, 0.5, 1 / 15)])
r = est.add(0.55, (50.0, 0, 0), yaw_q(0), 1)
ok &= check("지도 전환 직후 출력 없음", r is None)
# 6. 튐 (한 번에 1 m 점프) -> 버림
est = PoseToVelocity()
run(est, [(t, (0.15 * t, 0, 0), yaw_q(0), 0) for t in np.arange(0, 0.5, 1 / 15)])
r = est.add(0.55, (1.0, 0, 0), yaw_q(0), 0)
ok &= check("1 m 점프 버림", r is None and est.n_reject == 1)
# 7. 패킷
b = pack(7, 123456789, 0.1, -0.2, True)
m = unpack(b)
ok &= check("패킷 왕복", m[0] == 7 and m[1] == 123456789 and abs(m[2] - 0.1) < 1e-6 and m[4] is True)
ok &= check("깨진 패킷 거부", unpack(b"XXXX" + b[4:]) is None and unpack(b[:-1]) is None)
# 8. 수신기: 최신만, age, 무효
rx = VoReceiver(56099)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
now = time.time_ns()
for i, (dt_ms, vx) in enumerate(((-300, 0.05), (-100, 0.12), (-200, 0.08))):   # 순서 뒤바뀐 도착
    s.sendto(pack(i, now + dt_ms * 1_000_000, vx, 0.0), ("127.0.0.1", 56099))
time.sleep(0.05)
v = rx.poll()
ok &= check("가장 최근 측정만 사용", v and abs(v[0] - 0.12) < 1e-6 and 0.1 < v[2] < 0.3, f"{v}")
s.sendto(pack(9, time.time_ns(), 0, 0, False), ("127.0.0.1", 56099)); time.sleep(0.05)
ok &= check("valid=0 -> None (VO 없음)", rx.poll() is None)
rx.close()
print("ALL PASS" if ok else "SOME FAIL")

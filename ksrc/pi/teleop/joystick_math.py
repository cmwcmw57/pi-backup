"""
화면 조이스틱 패드의 순수 수학 부분.

패드 중심으로부터의 픽셀 변위 -> [-1,1] 정규화 스틱 위치 -> 차체 속도.
pygame 도 I/O 도 없어서 **디스플레이 없이 유닛테스트가 된다**
(test_joystick_math.py 참고).

패드 둘:
  왼쪽  -> 병진 (vx, vy)   [스워브는 홀로노믹이라 옆으로 가려고 차체를
                            돌릴 필요가 없다]
  오른쪽 -> 회전 (omega), 가로축만 사용

차체 좌표계는 firmware/common/swerve_kinematics.h 와 동일:
x=전진, y=좌측, omega 는 +가 반시계.
"""
import math


def clamp_to_unit_circle(dx: float, dy: float, radius: float):
    """dx, dy 는 패드 중심으로부터의 픽셀 변위. (nx, ny) 를 돌려준다.
    각각 [-1, 1] 범위이며 **단위 원**(정사각형이 아니라)으로 클램프한다.
    그래야 대각선으로 끈 게 직선으로 끈 것보다 크게 잡히지 않는다. out to the pad's edge."""
    if radius <= 0:
        return 0.0, 0.0
    nx = dx / radius
    ny = dy / radius
    mag = math.hypot(nx, ny)
    if mag > 1.0:
        nx /= mag
        ny /= mag
    return nx, ny


def apply_deadzone(nx: float, ny: float, deadzone: float):
    """원형 데드존. 크기가 deadzone 미만이면 (0, 0), 그 밖은 [0, 1] 로 다시 늘린다.

    이유: 스워브는 아주 작은 속도에서도 바퀴를 **그 방향으로 돌린다**. 패드 중심
    근처에서 마우스가 몇 픽셀 흔들리면 크기는 거의 0 인데 방향이 제각각인 명령이
    나가고, 바퀴 네 개가 그 방향을 따라 홱홱 돈다 (경계를 넘으면 180도 반전까지).
    데드존 밖은 (mag - dz) / (1 - dz) 로 늘려서 경계에서 속도가 갑자기 튀지 않는다.
    """
    mag = math.hypot(nx, ny)
    if mag < deadzone or mag <= 0.0:
        return 0.0, 0.0
    if deadzone <= 0.0:
        return nx, ny
    scale = min(1.0, (mag - deadzone) / (1.0 - deadzone)) / mag
    return nx * scale, ny * scale


def snap_to_axes(nx: float, ny: float, snap_deg: float):
    """방향이 앞·뒤·좌·우 축에서 snap_deg 이내면 그 축으로 맞춘다 (크기는 유지).

    횡걸음 중 손떨림으로 생기는 작은 앞뒤 성분(조향 87도 <-> 93도 잔떨림)을 없앤다.
    """
    mag = math.hypot(nx, ny)
    if mag <= 0.0 or snap_deg <= 0.0:
        return nx, ny
    ang = math.degrees(math.atan2(ny, nx))
    axis = round(ang / 90.0) * 90.0
    if abs(ang - axis) <= snap_deg:
        r = math.radians(axis)
        return mag * math.cos(r), mag * math.sin(r)
    return nx, ny


def _axis_deviation_deg(ax: float, ay: float, bx: float, by: float) -> float:
    """두 방향이 이루는 **직선** 사이 각 [0, 90]. 정반대(180도)는 0 이다.

    바퀴 입장에서 정반대 방향은 조향 없이 구동만 반전하면 되므로 같은 축이다.
    """
    d = abs(math.degrees(math.atan2(ay, ax) - math.atan2(by, bx))) % 180.0
    return min(d, 180.0 - d)


def lock_low_speed_direction(nx: float, ny: float, last_dir, hold_mag: float,
                             max_dev_deg: float):
    """저속에서 바퀴 축을 크게 바꾸는 명령을 0 으로 바꾼다.

    패드를 오른쪽에서 왼쪽으로 옮기다 중앙을 위/아래로 스치면, 그 사이 "전진"/
    "후진" 성분이 실제 명령으로 나가 네 바퀴가 0도 쪽으로 돌았다가 돌아온다
    (가동범위 경계에 걸리면 반대편까지 크게 넘어감). 사람이 의도한 건 구동 반전뿐이다.

    규칙: 크기 < hold_mag 이고, 직전에 내보낸 방향과 **축** 차이가 max_dev_deg 를
    넘으면 (0, 0) 을 낸다. 속도 0 이면 펌웨어가 조향각을 그대로 유지한다.
    정반대 방향(우 <-> 좌, 전 <-> 후)은 축 차이 0 이라 저속에서도 그대로 통과한다.
    새 방향으로 조향하려면 패드를 hold_mag 이상 민다.

    last_dir: 직전에 내보낸 0 아닌 방향 (nx, ny) 또는 None.
    반환: (nx, ny, 새 last_dir)
    """
    mag = math.hypot(nx, ny)
    if mag <= 0.0:
        return 0.0, 0.0, last_dir
    if (last_dir is not None and mag < hold_mag and
            _axis_deviation_deg(nx, ny, last_dir[0], last_dir[1]) > max_dev_deg):
        return 0.0, 0.0, last_dir
    return nx, ny, (nx, ny)


def left_pad_to_linear(nx: float, ny: float, max_lin_mps: float):
    """화면 위 = 전진(+vx), 화면 왼쪽 = +vy (차체 y 는 좌측이 +)."""
    vx = -ny * max_lin_mps
    vy = -nx * max_lin_mps
    return vx, vy


def right_pad_to_angular(nx: float, max_ang_rps: float):
    """오른쪽으로 끌면 시계방향 = omega 음수 (차체 omega 는 반시계가 +)."""
    return -nx * max_ang_rps

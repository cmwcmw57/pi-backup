"""
"가상 Nucleo". 개발 중에 텔레옵 클라이언트가 --port 로 가리킬 수 있는
실제 STM32 펌웨어의 대역이다. 하는 일:

  1. 실제 Nucleo 가 받을 것과 **완전히 동일한 바이트**를 받는다
     (pty 를 시리얼 포트처럼 써서)
  2. **실제** firmware/common/teleop_protocol.c 로 디코딩한다
     (ctypes 경유, ksrc_common.py 참고). 재구현이 아니다
  3. **실제** firmware/common/swerve_kinematics.c 로 기구학을 푼다
  4. 시각화를 위해 차체 자세를 적분하고, 실제 펌웨어와 같은 링크 워치독을
     적용한다 (링크가 끊기면 정지)

순수 로직은 VirtualNucleo 안에만 있어서(pygame 도 I/O 도 없음) 유닛테스트가
된다. test_virtual_nucleo.py 참고. 맨 아래 run_pygame() 이 대화형 사용을 위한
pty + 렌더링을 담당한다.
"""
import math
import threading
import time

from ksrc_common import SwerveModulePos, SwerveModuleState, TeleopLink, swerve_ik_compute

ROVER_HALF_LEN = 0.12   # m, 임시값. firmware/common/test_swerve_kinematics.c 참고
ROVER_HALF_WIDTH = 0.12
MAX_WHEEL_MPS = 1.0     # 모터 특성 측정 전까지 쓰는 임시 상한
WATCHDOG_TIMEOUT_MS = 300


def default_modules():
    m = (SwerveModulePos * 4)()
    m[0] = SwerveModulePos(ROVER_HALF_LEN, ROVER_HALF_WIDTH)    # FL
    m[1] = SwerveModulePos(ROVER_HALF_LEN, -ROVER_HALF_WIDTH)   # FR
    m[2] = SwerveModulePos(-ROVER_HALF_LEN, ROVER_HALF_WIDTH)   # RL
    m[3] = SwerveModulePos(-ROVER_HALF_LEN, -ROVER_HALF_WIDTH)  # RR
    return m


class VirtualNucleo:
    def __init__(self, modules=None, max_wheel_mps=MAX_WHEEL_MPS,
                 watchdog_timeout_ms=WATCHDOG_TIMEOUT_MS):
        self.modules = modules if modules is not None else default_modules()
        self.max_wheel_mps = max_wheel_mps
        self.link = TeleopLink(stale_timeout_ms=watchdog_timeout_ms)
        self.module_state = (SwerveModuleState * 4)()
        self.last_cmd = (0.0, 0.0, 0.0)  # 마지막으로 디코딩된 (vx, vy, omega)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.module_cmds = None  # 마지막 swerve_ik_compute() 출력. 렌더링용

    def feed_serial_bytes(self, data: bytes, now_ms: int):
        for byte in data:
            decoded = self.link.feed_byte(byte, now_ms)
            if decoded is not None:
                self.last_cmd = decoded

    def tick(self, now_ms: int, dt_s: float):
        """시뮬을 dt_s 만큼 진행. 워치독 적용 후 **실제로 인가된** (vx, vy, omega) 를
        돌려준다 (로깅/테스트용)."""
        if self.link.is_stale(now_ms):
            vx, vy, omega = 0.0, 0.0, 0.0
        else:
            vx, vy, omega = self.last_cmd

        self.module_cmds = swerve_ik_compute(vx, vy, omega, self.modules,
                                              self.max_wheel_mps, self.module_state)

        # 차체 자세 적분: 차체 좌표계 속도를 월드 좌표계로 회전
        wx = vx * math.cos(self.theta) - vy * math.sin(self.theta)
        wy = vx * math.sin(self.theta) + vy * math.cos(self.theta)
        self.x += wx * dt_s
        self.y += wy * dt_s
        self.theta += omega * dt_s

        return (vx, vy, omega)


class HeadlessHandle:
    """pygame 없이 실제 pty 로 VirtualNucleo 를 구동/관찰하는 데 필요한 것 전부.
    run_pygame() 이 쓰고, 화면 없이 "가상 Nucleo" 만 필요한 테스트/도구도
    직접 쓴다."""

    def __init__(self, sim, lock, source):
        self.sim = sim
        self.lock = lock
        self.source = source
        self.slave_name = source.slave_name

    def snapshot(self, now_ms=None):
        """(x, y, theta, vx, vy, omega, stale) 을 스레드 안전하게 읽는다."""
        import time as _time
        if now_ms is None:
            now_ms = int(_time.monotonic() * 1000)
        with self.lock:
            x, y, theta = self.sim.x, self.sim.y, self.sim.theta
            vx, vy, omega = self.sim.last_cmd
            stale = self.sim.link.is_stale(now_ms)
        return x, y, theta, vx, vy, omega, stale

    def close(self):
        self.source.close()

    def run_realtime(self, duration_s, dt_s=0.02):
        """블로킹. duration_s 동안 실시간으로 시뮬을 진행하며 dt_s 마다 tick 한다.
        그동안 백그라운드 리더 스레드가 pty 로 들어오는 바이트를 계속
        흘려넣는다. pygame 없이 시뮬을 실제로 "돌리고" 싶은 도구/테스트용.
        마지막 snapshot() 을 돌려준다."""
        end_at = time.monotonic() + duration_s
        while time.monotonic() < end_at:
            now_ms = int(time.monotonic() * 1000)
            with self.lock:
                self.sim.tick(now_ms, dt_s)
            time.sleep(dt_s)
        return self.snapshot()


def start_headless(watchdog_timeout_ms=WATCHDOG_TIMEOUT_MS):
    """pty 를 열고, 받은 바이트를 VirtualNucleo 에 흘려넣는 백그라운드 스레드를
    띄우고(실제 firmware/common 디코더 + 기구학을 거쳐서), HeadlessHandle 을
    돌려준다. pygame/디스플레이 불필요. 시뮬의 그림이 아니라 상태만 필요한
    테스트·도구용."""
    from pty_link import PtyByteSource

    sim = VirtualNucleo(watchdog_timeout_ms=watchdog_timeout_ms)
    lock = threading.Lock()

    def on_bytes(data, now_ms):
        with lock:
            sim.feed_serial_bytes(data, now_ms)

    source = PtyByteSource(on_bytes)
    return HeadlessHandle(sim, lock, source)


def run_pygame(port_out_path_cb=None, watchdog_timeout_ms=WATCHDOG_TIMEOUT_MS):
    """start_headless() 와 같은 가상 Nucleo 에 실시간 탑다운 pygame 렌더를 더한 것."""
    import pygame

    handle = start_headless(watchdog_timeout_ms=watchdog_timeout_ms)
    sim, lock, slave_name = handle.sim, handle.lock, handle.slave_name
    print(f"virtual Nucleo listening on: {slave_name}")
    print(f"  point a teleop client at it, e.g.:")
    print(f"  python3 ../pi/teleop/teleop_joystick.py --port {slave_name}")
    if port_out_path_cb:
        port_out_path_cb(slave_name)

    pygame.init()
    W, H = 700, 700
    PX_PER_M = 250.0
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("KSRC virtual Nucleo -- swerve sim")
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    def world_to_screen(wx, wy):
        return (W / 2 + wx * PX_PER_M, H / 2 - wy * PX_PER_M)

    trail = []
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            dt = clock.tick(60) / 1000.0
            now_ms = int(time.monotonic() * 1000)
            with lock:
                vx, vy, omega = sim.tick(now_ms, dt)
                x, y, theta = sim.x, sim.y, sim.theta
                module_cmds = list(sim.module_cmds) if sim.module_cmds else []
                stale = sim.link.is_stale(now_ms)

            trail.append(world_to_screen(x, y))
            if len(trail) > 2000:
                trail.pop(0)

            screen.fill((22, 22, 26))
            if len(trail) > 1:
                pygame.draw.lines(screen, (50, 70, 60), False, trail, 1)

            # 로버 차체
            corners_body = [(ROVER_HALF_LEN, ROVER_HALF_WIDTH), (ROVER_HALF_LEN, -ROVER_HALF_WIDTH),
                             (-ROVER_HALF_LEN, -ROVER_HALF_WIDTH), (-ROVER_HALF_LEN, ROVER_HALF_WIDTH)]
            pts = []
            for bx, by in corners_body:
                wx = x + bx * math.cos(theta) - by * math.sin(theta)
                wy = y + bx * math.sin(theta) + by * math.cos(theta)
                pts.append(world_to_screen(wx, wy))
            pygame.draw.polygon(screen, (60, 90, 130), pts, 0)
            pygame.draw.polygon(screen, (120, 170, 220), pts, 2)
            # 진행방향 표시
            hx = x + (ROVER_HALF_LEN * 1.3) * math.cos(theta)
            hy = y + (ROVER_HALF_LEN * 1.3) * math.sin(theta)
            pygame.draw.line(screen, (255, 210, 90), world_to_screen(x, y), world_to_screen(hx, hy), 3)

            # 바퀴 모듈
            module_positions = [(sim.modules[i].x, sim.modules[i].y) for i in range(4)]
            for i, (mx, my) in enumerate(module_positions):
                wx = x + mx * math.cos(theta) - my * math.sin(theta)
                wy = y + mx * math.sin(theta) + my * math.cos(theta)
                center = world_to_screen(wx, wy)
                if i < len(module_cmds):
                    steer = theta + module_cmds[i].angle_rad
                    speed = module_cmds[i].speed_mps
                else:
                    steer, speed = theta, 0.0
                wheel_len_px = 22
                dx = wheel_len_px * math.cos(steer)
                dy = -wheel_len_px * math.sin(steer)
                color = (250, 120, 120) if speed < 0 else (120, 250, 160)
                pygame.draw.line(screen, color, (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy), 4)
                pygame.draw.circle(screen, (200, 200, 210), center, 3)

            status = "LINK STALE (stopped)" if stale else "link OK"
            lines = [
                f"pose: x={x:+.2f}m y={y:+.2f}m theta={math.degrees(theta):+.1f}deg",
                f"cmd:  vx={vx:+.2f} vy={vy:+.2f} omega={omega:+.2f}   [{status}]",
                f"port: {slave_name}",
            ]
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, (220, 220, 230)), (10, 10 + 18 * i))

            pygame.display.flip()
    finally:
        handle.close()
        pygame.quit()


if __name__ == "__main__":
    run_pygame()

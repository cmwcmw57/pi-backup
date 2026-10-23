#!/usr/bin/env python3
"""
"가상 Nucleo" 의 MuJoCo/URDF 판.

프로토콜도 같고 **실제 firmware/common C 기구학**(ksrc_common.py 경유)도 같지만,
virtual_nucleo.py 의 2D 추측항법 적분 대신 실제 URDF(ksrc_rover.urdf)를 MuJoCo
강체 물리로 굴린다. 차체 운동이 가정한 기구학 모델이 아니라 **실제 바퀴-지면
접촉력**에서 나온다.

구동에 MuJoCo <actuator> 를 쓰지 **않는다**: .urdf 안에 끼워넣은
<mujoco><actuator>...</mujoco> 블록은 MuJoCo 의 URDF 임포터가 읽지 않는다
(직접 실험함 -- 그 블록에 뭘 넣든 model.nu 가 0 으로 남았다). 대신 매 물리
스텝마다 조향 조인트의 qpos 와 바퀴 조인트의 qvel 을 직접 써넣는다:
  - 조향: swerve_kinematics.c 가 설계상 이미 **연속(unwrap)된** 목표각을
    내주므로 wrap 처리 없이 qpos 에 그대로 넣으면 된다
    (swerve_kinematics.h 의 의도된 설계 속성이지 우연이 아니다).
  - 바퀴: speed_mps / WHEEL_RADIUS_M -> 목표 rad/s -> qvel 에 직접.
조인트 자체는 "기구학적으로" 구동하지만, 그 결과 생기는 지면 접촉력(그리고
자유부유 base_link 의 운동)은 여전히 진짜 MuJoCo 물리다.

사용법:
  python3 mujoco_sim.py
  # 텔레옵 클라이언트가 가리킬 pty 경로를 출력한다. 예:
  python3 ../../pi/teleop/teleop_terminal.py --port /dev/pts/N
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sim/ -> ksrc_common, pty_link
from ksrc_common import SwerveModulePos, SwerveModuleState, TeleopLink, swerve_ik_compute
from pty_link import PtyByteSource

URDF_PATH = Path(__file__).resolve().parent / "ksrc_rover.urdf"

# ksrc_rover.urdf 의 모듈 오프셋 및 바퀴 반경과 **정확히** 일치해야 한다.
# (둘 다 현재는 firmware/common/test_swerve_kinematics.c 의 임시값과
#  제안서의 105mm 바퀴 임시값을 따른다)
MODULE_HALF_LEN = 0.12
MODULE_HALF_WIDTH = 0.12
WHEEL_RADIUS_M = 0.0525
MAX_WHEEL_MPS = 1.0
WATCHDOG_TIMEOUT_MS = 300

JOINT_ORDER = ["fl", "fr", "rl", "rr"]  # matches swerve_ik_compute's module indexing


def build_modules():
    m = (SwerveModulePos * 4)()
    m[0] = SwerveModulePos(MODULE_HALF_LEN, MODULE_HALF_WIDTH)    # FL
    m[1] = SwerveModulePos(MODULE_HALF_LEN, -MODULE_HALF_WIDTH)   # FR
    m[2] = SwerveModulePos(-MODULE_HALF_LEN, MODULE_HALF_WIDTH)   # RL
    m[3] = SwerveModulePos(-MODULE_HALF_LEN, -MODULE_HALF_WIDTH)  # RR
    return m


class MujocoRover:
    """Pure sim-stepping logic, no viewer/pty -- unit-testable (see
    test_mujoco_sim.py) the same way VirtualNucleo is."""

    def __init__(self, model_path=URDF_PATH, watchdog_timeout_ms=WATCHDOG_TIMEOUT_MS):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.modules = build_modules()
        self.module_state = (SwerveModuleState * 4)()
        self.link = TeleopLink(stale_timeout_ms=watchdog_timeout_ms)
        self.last_cmd = (0.0, 0.0, 0.0)

        self.steer_qposadr = []
        self.wheel_dofadr = []
        for name in JOINT_ORDER:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_steer_joint")
            wid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_wheel_joint")
            assert sid >= 0 and wid >= 0, f"joint not found for module {name}"
            self.steer_qposadr.append(self.model.jnt_qposadr[sid])
            self.wheel_dofadr.append(self.model.jnt_dofadr[wid])

    def feed_serial_bytes(self, data: bytes, now_ms: int):
        for byte in data:
            decoded = self.link.feed_byte(byte, now_ms)
            if decoded is not None:
                self.last_cmd = decoded

    def settle(self, n_steps=500):
        """Let the rover drop onto the ground plane before driving starts."""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def step(self, now_ms: int):
        """Apply the current command (post-watchdog) and advance physics
        by one timestep. Returns (vx, vy, omega) actually applied."""
        if self.link.is_stale(now_ms):
            vx, vy, omega = 0.0, 0.0, 0.0
        else:
            vx, vy, omega = self.last_cmd

        cmds = swerve_ik_compute(vx, vy, omega, self.modules, MAX_WHEEL_MPS, self.module_state)

        for i in range(4):
            self.data.qpos[self.steer_qposadr[i]] = cmds[i].angle_rad
            self.data.qvel[self.wheel_dofadr[i]] = cmds[i].speed_mps / WHEEL_RADIUS_M

        mujoco.mj_step(self.model, self.data)
        return (vx, vy, omega)

    @property
    def base_pos(self):
        return tuple(self.data.qpos[0:3])


def main():
    rover = MujocoRover()
    rover.settle()

    source = PtyByteSource(lambda data, now_ms: rover.feed_serial_bytes(data, now_ms))
    print(f"MuJoCo virtual Nucleo listening on: {source.slave_name}")
    print(f"  python3 ../../pi/teleop/teleop_terminal.py --port {source.slave_name}")
    print(f"  python3 ../../pi/teleop/teleop_joystick.py --port {source.slave_name}")

    last_print = 0.0
    try:
        with mujoco.viewer.launch_passive(rover.model, rover.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                now_ms = int(time.monotonic() * 1000)
                vx, vy, omega = rover.step(now_ms)
                viewer.sync()

                if step_start - last_print > 0.5:
                    x, y, z = rover.base_pos
                    stale = rover.link.is_stale(now_ms)
                    status = "STALE" if stale else "OK"
                    print(f"\rpos=({x:+.2f},{y:+.2f},{z:+.2f}) cmd=(vx={vx:+.2f} vy={vy:+.2f} "
                          f"omega={omega:+.2f}) link={status}   ", end="", flush=True)
                    last_print = step_start

                dt_left = rover.model.opt.timestep - (time.time() - step_start)
                if dt_left > 0:
                    time.sleep(dt_left)
    finally:
        source.close()
        print("\nclosed")


if __name__ == "__main__":
    main()

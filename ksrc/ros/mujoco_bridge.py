"""MuJoCo <-> ROS2 브리지.  Nav2 를 우리 로버에 붙여 검증하기 위한 것.

**시간은 wall time 을 쓴다** (`use_sim_time: false`).  브리지가 50Hz 실시간으로
도니까 sim time 을 따로 발행할 이유가 없고, 섞으면 TF 스탬프(wall)와 /clock(sim)이
어긋나 Nav2 가 "Detected jump back in time" 으로 TF 버퍼를 계속 비운다.

**측위를 참값으로 둔다** (map->odom 을 항등으로 발행).  그러면 "Nav2 가 우리
로버에 맞게 도는가" 만 남고, 측위 문제가 섞이지 않는다.  측위(D435 지형 ICP)가
되면 이 항등 변환만 실제 추정으로 갈아끼우면 된다.

Gazebo 를 쓰지 않는 이유: 학습이 MuJoCo 에 있고 터라메크(슬립-침하)와 펌웨어 C
공유가 거기 묶여 있다.  Nav2 검증에만 Gazebo 를 쓰면 물리가 달라져 의미가 줄어든다.

발행
  /tf          map->odom (항등),  odom->base_link (참값)
  /odom        nav_msgs/Odometry
  /cmd_vel_filtered  geometry_msgs/Twist  ->  RL 전의 필터링된 Nav2 명령
  /rl_action         std_msgs/Float32MultiArray -> 정규화된 3D 정책 행동
  /rl_delta          geometry_msgs/Twist  ->  정책이 요청한 차체속도 잔차
  /cmd_vel_applied   geometry_msgs/Twist  ->  포화 처리 후 IK 에 들어간 명령
구독
  /cmd_vel     geometry_msgs/Twist  ->  저역통과 후 env 의 명령으로 주입
  /plan        nav_msgs/Path        ->  정책의 경로 관측 (map 좌표)
서비스 /reset_sim (std_srvs/Trigger) 은 명시적으로 같은 스폰 위치에 다시 놓는다.
서비스 /set_rl (std_srvs/SetBool) 은 정책 적용을 주행 중 켜고 끈다.
목표는 RViz 의 "2D Goal Pose" -> Nav2 -> /cmd_vel 로 들어온다.
"""
import argparse
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger, SetBool
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "sim" / "rl"))
import mujoco                                    # noqa: E402
import env as ksrc_env                           # noqa: E402
import path as pth                                # noqa: E402
from config import RoverCfg                      # noqa: E402
from reward import RewardCfg                     # noqa: E402
from policy_observation import PolicyPath, set_route_fields  # noqa: E402
from teleop_transition import TeleopTransitionGuard  # noqa: E402

CTRL_HZ = ksrc_env.CTRL_HZ


def yaw_to_quat(y):
    return (0.0, 0.0, math.sin(y / 2.0), math.cos(y / 2.0))


def filter_cmd(previous, target, tau, dt):
    """차체 명령을 IK 전에 저역통과. 명시적인 정지는 지연시키지 않는다."""
    if tau <= 0.0 or not np.any(target):
        return target.copy()
    alpha = dt / (tau + dt)
    return previous + alpha * (target - previous)


def as_twist(cmd):
    msg = Twist()
    msg.linear.x = float(cmd[0])
    msg.linear.y = float(cmd[1])
    msg.angular.z = float(cmd[2])
    return msg


class Bridge(Node):
    def __init__(self, args):
        super().__init__("mujoco_bridge")
        self.cfg = RoverCfg()
        # 경로 추종은 Nav2 가 하므로 env 의 내부 pure pursuit 은 쓰지 않는다.
        # paths=None 이면 env 가 자체 경로를 만들지만, cmd 를 외부에서 덮어쓴다.
        # 규사 대회맵의 평탄 구역에서만 시작한다. 학습/평가 스폰 분포는 그대로.
        # --mech-fold: 접기 기준을 '모터가 차체 중심에 가까운 쪽' 으로 바꾼다.
        # 차체 운동은 동일하고 모터가 어디로 튀어나오는지만 달라진다.
        # 측정으로는 손해(서보회전 +14.7%, 반전 +46%)지만 눈으로 비교할 수 있게 둔다.
        env_cls = ksrc_env.RoverEnv
        if args.mech_fold:
            from mech_fold import MechFoldEnv
            env_cls = MechFoldEnv
        cfg = None
        if args.steer_limit_deg is not None:
            lim = float(args.steer_limit_deg)
            cfg = RoverCfg(steer_lo_deg=(-lim,) * 4, steer_hi_deg=(lim,) * 4)
        self.env = env_cls(
            cfg=cfg,
            rew=RewardCfg(), difficulty=args.d,
            arena_eval=True, eval_kind=args.terrain,
            randomize=False, perturb=0.0, seed=args.seed,
            episode_s=1e6,
            spawn_max_slope_deg=3.0 if args.terrain == "sand" else None,
            drive_align_gate_deg=args.drive_align_gate_deg)
        self.teleop_guard = TeleopTransitionGuard() if args.teleop_guard else None
        self.spawn_seed = args.seed
        self.obs, _ = self.env.reset(seed=args.seed)
        self.policy = None
        self.norm = None
        self.policy_enabled = False
        self.rl_scale = args.rl_scale
        if args.model:
            import pickle
            from stable_baselines3 import PPO
            self.policy = PPO.load(args.model, device="cpu")
            if self.policy.action_space.shape != (3,) or \
                    self.policy.observation_space.shape != self.env.observation_space.shape:
                raise ValueError("정책의 관측/행동 차원이 현재 student 환경과 다릅니다")
            with open(args.vecnorm, "rb") as f:
                self.norm = pickle.load(f)
            if self.norm.obs_rms.mean.shape != self.env.observation_space.shape:
                raise ValueError("vecnorm 관측 차원이 정책과 다릅니다")
            self.policy_enabled = True
        self.policy_path = None
        self.new_plan = False
        self.warned_no_plan = False
        self.last_action = np.zeros(3, dtype=np.float32)
        self.cmd = np.zeros(3)
        self.cmd_time = time.monotonic()
        self.cmd_timeout = args.cmd_timeout
        self.cmd_filtered = np.zeros(3)
        self.cmd_filter_tau = args.cmd_filter_tau
        self.lock = threading.Lock()
        self.n_tip = 0
        self.tip_latched = False
        self.warned_oob = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(Twist, "/cmd_vel", self.on_cmd, qos)
        self.sub_plan = self.create_subscription(Path, "/plan", self.on_plan, qos)
        self.pub_cmd_filtered = self.create_publisher(Twist, "/cmd_vel_filtered", qos)
        self.pub_rl_action = self.create_publisher(Float32MultiArray, "/rl_action", qos)
        self.pub_rl_delta = self.create_publisher(Twist, "/rl_delta", qos)
        self.pub_cmd_applied = self.create_publisher(Twist, "/cmd_vel_applied", qos)
        self.pub_odom = self.create_publisher(Odometry, "/odom", qos)
        self.reset_srv = self.create_service(Trigger, "/reset_sim", self.on_reset)
        self.rl_srv = self.create_service(SetBool, "/set_rl", self.on_set_rl)
        self.tf = TransformBroadcaster(self)
        self.t_sim = 0.0
        self.timer = self.create_timer(1.0 / CTRL_HZ, self.tick)
        self.get_logger().info(
            "MuJoCo bridge: %s 지형, 측위=참값(map->odom 항등), %d Hz, "
            "cmd 필터 tau=%.2fs, RL=%s" %
            (args.terrain, CTRL_HZ, self.cmd_filter_tau,
             args.model if self.policy_enabled else "OFF"))

    def on_cmd(self, m):
        with self.lock:
            self.cmd = np.array([m.linear.x, m.linear.y, m.angular.z], float)
            self.cmd_time = time.monotonic()

    def on_plan(self, m):
        if m.header.frame_id.lstrip("/") not in ("map", "odom"):
            self.get_logger().warn("/plan 좌표계가 map/odom 이 아님: %s" % m.header.frame_id)
            return
        try:
            points = [(p.pose.position.x, p.pose.position.y) for p in m.poses]
            plan = PolicyPath(points)
        except ValueError as exc:
            self.get_logger().warn("/plan 무시: %s" % exc)
            return
        # Nav2 는 같은 목표로도 주기적으로 재계획한다. 그때마다 1초 관측
        # 이력을 초기화하면 정책이 시간변화를 볼 수 없으므로 목표가 바뀔 때만 한다.
        changed_goal = (self.policy_path is None or np.linalg.norm(
            plan.ref.P[-1] - self.policy_path.ref.P[-1]) > 0.15)
        self.policy_path = plan
        self.new_plan = self.new_plan or changed_goal
        self.warned_no_plan = False
        if changed_goal:
            self.get_logger().info("RL 관측 경로 수신: %d점, %.2fm" %
                                   (len(plan.ref.P), plan.ref.total))

    def on_set_rl(self, request, response):
        if self.policy is None and request.data:
            response.success = False
            response.message = "정책 미로드: launch 에 rl:=true 를 지정하세요"
            return response
        self.policy_enabled = bool(request.data)
        response.success = True
        response.message = "RL ON" if self.policy_enabled else "RL OFF (순수 IK)"
        self.get_logger().info(response.message)
        return response

    def on_reset(self, request, response):
        # 명시적인 리셋만 위치를 바꾼다. seed 를 다시 주면 처음과 같은 스폰이다.
        self.obs, _ = self.env.reset(seed=self.spawn_seed)
        with self.lock:
            self.cmd[:] = 0.0
            self.cmd_filtered[:] = 0.0
            self.cmd_time = time.monotonic()
        if self.teleop_guard is not None:
            self.teleop_guard.reset()
        self.new_plan = True
        self.tip_latched = False
        self.warned_oob = False
        self.publish()
        response.success = True
        response.message = "시뮬레이션 리셋: 초기 스폰으로 복귀"
        return response

    def tick(self):
        with self.lock:
            stale = time.monotonic() - self.cmd_time > self.cmd_timeout
            cmd = np.zeros(3) if self.tip_latched or stale else self.cmd.copy()
        if self.teleop_guard is not None:
            yaw = self.env._yaw()
            cy, sy = math.cos(yaw), math.sin(yaw)
            vx, vy = self.env.d.qvel[:2]
            velocity_body = np.array([cy * vx + sy * vy,
                                      -sy * vx + cy * vy])
            cmd = self.teleop_guard.apply(cmd, velocity_body)
        self.cmd_filtered = filter_cmd(
            self.cmd_filtered, cmd, self.cmd_filter_tau, 1.0 / CTRL_HZ)
        self.pub_cmd_filtered.publish(as_twist(self.cmd_filtered))
        # 학습 때와 같은 경로 관측 필드를 Nav2 /plan 기준으로 바꾼다.
        # env.step() 은 내부 pure pursuit 관측을 만들어 반환하므로, 매 주기
        # 정책 예측 직전에 최신 프레임을 덮어써야 한다.
        action = np.zeros(3, dtype=np.float32)
        if self.policy_path is not None:
            pos = self.env.d.qpos[:2]
            target_L = pth.pursuit_distance(
                float(np.linalg.norm(self.cmd_filtered[:2])),
                self.cfg.pp_k_v, self.cfg.pp_l_min, self.cfg.pp_l_max)
            route_features = self.policy_path.features(
                pos, self.env._yaw(), target_L)
            set_route_fields(self.env.hist, self.cmd_filtered, self.env.v_max,
                             route_features, all_frames=self.new_plan)
            self.new_plan = False
            self.obs = self.env._obs()
            # 학습 과제는 경로를 따라 이동하는 동안의 보정이다. Nav2 가 목표에서
            # 제자리 yaw 정렬만 할 때는 병진 잔차가 목표 수렴을 방해할 수 있다.
            driving = np.linalg.norm(self.cmd_filtered[:2]) > 0.02
            if self.policy_enabled and not self.tip_latched and driving:
                normalized = np.clip(
                    (self.obs - self.norm.obs_rms.mean) /
                    np.sqrt(self.norm.obs_rms.var + 1e-8), -10, 10).astype(np.float32)
                action = self.policy.predict(normalized, deterministic=True)[0]
                action = np.clip(action * self.rl_scale, -1.0, 1.0)
        elif self.policy_enabled and np.any(self.cmd_filtered) and not self.warned_no_plan:
            self.get_logger().warn("/plan 대기 중: RL 잔차 0, 순수 IK 적용")
            self.warned_no_plan = True

        msg = Float32MultiArray(); msg.data = [float(x) for x in action]
        self.last_action = action.copy()
        self.pub_rl_action.publish(msg)
        lim = np.array([self.cfg.d_vx_max, self.cfg.d_vy_max, self.cfg.d_om_max])
        self.pub_rl_delta.publish(as_twist(action * lim))
        # env 의 pure pursuit 명령을 이 제어주기 동안 Nav2 명령으로 교체한다.
        self.env.cmd_nom = self.cmd_filtered.copy()
        self.obs, _, term, trunc, info = self.env.step(action)
        self.pub_cmd_applied.publish(as_twist(self.env.cmd))
        # 자동 리셋은 RViz 의 위치를 순간이동시킨다. 전복 뒤에는 구동을 멈추고
        # /reset_sim 을 호출할 때까지 위치를 연속적으로 발행한다.
        if info["tip"] and not self.tip_latched:
            self.get_logger().warn("전복 감지: 구동 중지. /reset_sim 으로 수동 리셋")
            self.tip_latched = True
            with self.lock:
                self.cmd[:] = 0.0
            self.n_tip += 1
        if info["oob"] and not self.warned_oob:
            self.get_logger().warn("경계 도달 (리셋하지 않음)")
            self.warned_oob = True
        elif not info["oob"]:
            self.warned_oob = False
        self.t_sim += 1.0 / CTRL_HZ
        self.publish()

    def publish(self):
        d = self.env.d
        now = self.get_clock().now().to_msg()

        x, y = float(d.qpos[0]), float(d.qpos[1])
        w, qx, qy, qz = d.qpos[3:7]
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        qz_, qw_ = math.sin(yaw / 2), math.cos(yaw / 2)

        # map -> odom : 항등 (측위가 완벽하다는 가정).  실제 측위가 들어오면 여기만 바뀐다.
        t1 = TransformStamped()
        t1.header.stamp = now; t1.header.frame_id = "map"; t1.child_frame_id = "odom"
        t1.transform.rotation.w = 1.0
        # odom -> base_link : 참값
        t2 = TransformStamped()
        t2.header.stamp = now; t2.header.frame_id = "odom"; t2.child_frame_id = "base_link"
        t2.transform.translation.x = x; t2.transform.translation.y = y
        t2.transform.translation.z = float(d.qpos[2])
        t2.transform.rotation.z = qz_; t2.transform.rotation.w = qw_
        self.tf.sendTransform([t1, t2])

        o = Odometry()
        o.header.stamp = now; o.header.frame_id = "odom"; o.child_frame_id = "base_link"
        o.pose.pose.position.x = x; o.pose.pose.position.y = y
        o.pose.pose.orientation.z = qz_; o.pose.pose.orientation.w = qw_
        cy, sy = math.cos(yaw), math.sin(yaw)
        vw = d.qvel[:2]
        o.twist.twist.linear.x = float(cy * vw[0] + sy * vw[1])
        o.twist.twist.linear.y = float(-sy * vw[0] + cy * vw[1])
        o.twist.twist.angular.z = float(d.qvel[5])
        self.pub_odom.publish(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="sand", choices=["sand", "rock"])
    ap.add_argument("--d", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmd-filter-tau", type=float, default=0.10,
                    help="IK 전 cmd_vel 저역통과 시정수 [s]; 0이면 끔")
    ap.add_argument("--cmd-timeout", type=float, default=0.5,
                    help="이 시간 동안 /cmd_vel 수신이 없으면 정지 [s]")
    ap.add_argument("--teleop-guard", action="store_true",
                    help="텔레옵 큰 방향 전환 시 차체 감속 (조향각 게이트는 별도 공통 옵션)")
    ap.add_argument("--drive-align-gate-deg", type=float, default=10.0,
                    help="전 모드 공통: 구동 허용 조향 오차 [deg], 0=끔")
    ap.add_argument("--model", default=None, help="student PPO 모델 경로")
    ap.add_argument("--vecnorm", default=None, help="모델과 짝인 VecNormalize pickle")
    ap.add_argument("--rl-scale", type=float, default=1.0,
                    help="정책 잔차 배율 [0,1]")
    ap.add_argument("--mech-fold", action="store_true",
                    help="접기 선택을 '모터가 차체 중심에 가까운 쪽' 으로 바꾼다 "
                         "(횡걸음에서 FL-90 FR+90 RL+90 RR-90). 운동은 동일")
    ap.add_argument("--steer-limit-deg", type=float, default=None,
                    help="조향 가동범위 ±[deg] 덮어쓰기 (기본 config 의 ±100). "
                         "±110 이상은 모터가 타이어보다 바깥으로 나간다")
    ap.add_argument("--view", action="store_true", help="MuJoCo 뷰어도 띄운다")
    args, _ = ap.parse_known_args()
    if not math.isfinite(args.cmd_filter_tau) or args.cmd_filter_tau < 0:
        ap.error("--cmd-filter-tau 는 0 이상의 유한한 숫자여야 합니다")
    if not math.isfinite(args.cmd_timeout) or args.cmd_timeout <= 0:
        ap.error("--cmd-timeout 은 0보다 큰 유한한 숫자여야 합니다")
    if not math.isfinite(args.drive_align_gate_deg) or args.drive_align_gate_deg < 0:
        ap.error("--drive-align-gate-deg 는 0 이상의 유한한 숫자여야 합니다")
    if bool(args.model) != bool(args.vecnorm):
        ap.error("--model 과 --vecnorm 은 함께 지정해야 합니다")
    if not math.isfinite(args.rl_scale) or not 0.0 <= args.rl_scale <= 1.0:
        ap.error("--rl-scale 은 0 이상 1 이하이어야 합니다")

    rclpy.init()
    node = None
    try:
        node = Bridge(args)
        if args.view:
            import mujoco.viewer
            v = mujoco.viewer.launch_passive(node.env.m, node.env.d)
            try:
                while rclpy.ok() and v.is_running():
                    rclpy.spin_once(node, timeout_sec=0.0)
                    v.sync()
                    time.sleep(0.005)
            finally:
                v.close()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

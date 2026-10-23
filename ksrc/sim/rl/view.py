#!/usr/bin/env python3
"""지형·로버를 MuJoCo 뷰어로 띄운다 (눈으로 확인하는 용도).

  python3 view.py                     # 규사 경사지형 STL, 순수 IK 주행
  python3 view.py --terrain proc --d 1.0
  python3 view.py --model runs/reach/final --vecnorm runs/reach/vecnorm.pkl
  python3 view.py --stage1 --watch-run runs/training
  python3 view.py --static            # 물리 정지, 지형만 둘러보기

  # cmdvel 밑단 teacher 를 보려면 (s51: 특권관측 223D, 이력 5프레임, tau 0.10)
  python3 view.py --task cmdvel --terrain proc --history-frames 5 \
      --cmd-filter-tau 0.10 \
      --model runs/s51_cmdvel_steady_tail_bc/model.zip \
      --vecnorm runs/s51_cmdvel_steady_tail_bc/vecnorm.pkl

`o` 로 잔차를 켜고 끄면 **ON/OFF 각각의 누적 추종오차**를 따로 모아 비교해
준다.  이득이 병진 5.7mm/s (최대속도 378mm/s 의 1.5%) 라 눈으로는 안 보인다
-- 숫자를 봐야 한다.

마우스: 좌드래그 회전 / 우드래그 이동 / 휠 줌.  p 로 일시정지.
"""
import argparse, pathlib, re, sys, time
import numpy as np
import mujoco, mujoco.viewer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import RoverEnv, CTRL_HZ, OBS_PER_FRAME, HISTORY

OBS_DIM = OBS_PER_FRAME * HISTORY


# ----------------------------------------------------------------- 명령 벡터
# 논문 그림에서 흔히 보는 "명령 twist 화살표".  cmdvel 은 추종할 경로가 없어서
# 이게 없으면 화면에 아무 단서가 없다 -- 명령의 상당수가 제자리 회전·게걸음이라
# 위치가 거의 안 변하고, 그러면 "안 움직인다" 로 보인다.
ARROW_MPS = 0.6          # 1 m/s 를 몇 m 길이로 그릴지
ARROW_MIN = 0.12         # 작은 명령도 방향은 보이게 하는 최소 길이 [m]
ARROW_OMEGA = 0.18       # 1 rad/s 를 수직으로 몇 m 로 그릴지
C_CMD_V   = (0.20, 0.55, 1.00, 0.95)   # 파랑  명령 선속도
C_ACT_V   = (1.00, 0.55, 0.10, 0.95)   # 주황  실제 선속도
C_CMD_OM  = (0.25, 0.90, 0.35, 0.95)   # 초록  명령 요속도
C_ACT_OM  = (0.95, 0.90, 0.20, 0.95)   # 노랑  실제 요속도


def _arrow(scn, n, p0, p1, rgba, width):
    if n >= scn.maxgeom:
        return n
    g = scn.geoms[n]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, p0, p1)
    return n + 1


def draw_twist(env, scn, n=0):
    """명령/실제 twist 를 차체 위에 화살표로 그린다.  그린 geom 수를 돌려준다.

    가로 화살표 = 선속도 (월드좌표).  길이는 크기 비례지만 ARROW_MIN 바닥이
    있어서 작은 명령도 **방향은** 보인다.  수직 화살표 = 요속도 (위 CCW).
    명령이 제자리 회전이면 가로가 짧아지고 수직만 남는다 -- 그게 로버가
    제자리에 있는 이유를 그대로 보여 준다.
    """
    yaw = env._yaw(); c_, s_ = np.cos(yaw), np.sin(yaw)
    cmd, act = env.cmd_nom, env.body_twist()
    base = np.array([env.d.qpos[0], env.d.qpos[1], float(env.d.qpos[2]) + 0.16])
    for (vx, vy), rgba, w in ((( cmd[0], cmd[1]), C_CMD_V, 0.016),
                              ((act[0], act[1]), C_ACT_V, 0.012)):
        wx, wy = c_ * vx - s_ * vy, s_ * vx + c_ * vy      # 차체 -> 월드
        mag = float(np.hypot(wx, wy))
        if mag < 1e-3:
            continue
        L = max(mag * ARROW_MPS, ARROW_MIN)
        n = _arrow(scn, n, base, base + np.array([wx, wy, 0.0]) / mag * L, rgba, w)
    for om, rgba, w in ((cmd[2], C_CMD_OM, 0.013), (act[2], C_ACT_OM, 0.010)):
        if abs(om) < 1e-2:
            continue
        n = _arrow(scn, n, base,
                   base + np.array([0.0, 0.0, om * ARROW_OMEGA]), rgba, w)
    return n


def tint_rover(m, rgba):
    """로버의 시각 메시만 물들인다.  창 두 개를 띄웠을 때 어느 쪽이
    순수 IK 이고 어느 쪽이 정책인지 창 제목으로는 구분할 수 없다
    (launch_passive 에 title 인자가 없다)."""
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if m.geom_contype[i] == 0 and name != "ground":
            m.geom_rgba[i] = rgba


def latest_checkpoint_pair(run):
    """Find the newest policy and its matching observation statistics."""
    root = pathlib.Path(run)
    final = root / "final.zip"
    final_norm = root / "vecnorm.pkl"
    if final.is_file() and final_norm.is_file():
        return final, final_norm
    pairs = []
    for model in root.glob("ppo_*_steps.zip"):
        match = re.fullmatch(r"ppo_(\d+)_steps\.zip", model.name)
        if match:
            norm = root / f"ppo_vecnormalize_{match.group(1)}_steps.pkl"
            if norm.is_file():
                pairs.append((int(match.group(1)), model, norm))
    if not pairs:
        return None
    _, model, norm = max(pairs, key=lambda pair: pair[0])
    return model, norm


def _ghost_dataid(env):
    """geom id -> **장면용** 메시 dataid 매핑.

    `mjvGeom.dataid` 는 `mjModel.geom_dataid`(모델 메시 id)가 **아니다**.
    MuJoCo 는 모델 메시 하나당 렌더 메시를 둘 할당해서 실제로는 2배가 된다.
    모델 id 를 그대로 넣으면 부품마다 **다른 부품의 메시**가 그려져서
    로버가 부서진 것처럼 보인다 (URDF 가 깨진 걸로 오해하기 딱 좋다).
    배수를 가정하지 않고 MuJoCo 가 만든 장면에서 직접 읽어 캐시한다.
    """
    cached = getattr(env, "_ghost_map", None)
    if cached is not None:
        return cached
    ref = mujoco.MjvScene(env.m, maxgeom=2 * env.m.ngeom + 16)
    mujoco.mjv_updateScene(env.m, env.d, mujoco.MjvOption(), None,
                           mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, ref)
    out = {}
    for k in range(ref.ngeom):
        g = ref.geoms[k]
        if (g.objtype == mujoco.mjtObj.mjOBJ_GEOM
                and g.type == mujoco.mjtGeom.mjGEOM_MESH):
            out[int(g.objid)] = int(g.dataid)
    env._ghost_map = out
    return out


def draw_ghost(env, scn, n=0, rgba=(1.0, 0.45, 0.10, 0.40), offset=(0.0, 0.0)):
    """다른 환경의 로버를 **반투명 고스트**로 같은 장면에 겹쳐 그린다.

    창을 두 개 띄우면 멈춘다 -- mujoco.viewer 의 passive 핸들은 각자
    렌더 스레드에서 GLFW 이벤트를 돌리는데, GLFW 이벤트 처리는 스레드
    안전하지 않아서 둘이 서로 붙잡고 교착한다.  그래서 창은 하나만 띄우고
    비교 대상을 이 함수로 겹친다.  레이싱 게임의 고스트카와 같은 방식이고,
    나란히 놓는 것보다 **궤적이 벌어지는 순간**이 훨씬 잘 보인다.

    자세는 geom_xpos/geom_xmat 를 그대로 쓴다 -- MuJoCo 자신의 장면과
    대조해 위치·회전 오차가 0.000mm 임을 확인했다.  mesh_pos/mesh_quat 로
    **보정하면 오히려 64mm 어긋난다** (그 둘은 컴파일 시 이미 반영돼 있다).
    """
    m, d = env.m, env.d
    dmap = _ghost_dataid(env)
    off = np.array([offset[0], offset[1], 0.0])
    for i in range(m.ngeom):
        if n >= scn.maxgeom:
            break
        # 시각 메시(contype=0)만.  충돌 박스까지 그리면 화면이 지저분하다.
        if m.geom_contype[i] != 0 or i not in dmap:
            continue
        g = scn.geoms[n]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_MESH, m.geom_size[i],
                            d.geom_xpos[i] + off, d.geom_xmat[i],
                            np.array(rgba, np.float32))
        g.dataid = dmap[i]
        n += 1
    return n


def run_compare(env_kw, policy, args, seed):
    """순수 IK 와 정책을 **같은 시드로** 나란히 돌려 비교한다.

    창은 하나다.  실체(불투명)가 순수 IK, 반투명 주황 고스트가 정책이다.
    --headless 면 창 없이 터미널 숫자만 낸다 (훨씬 빠르다).

    두 환경은 같은 seed 로 reset 하므로 지형·명령·초기자세가 같다.  다만
    궤적이 갈라진 뒤에는 잡음 난수 소비가 완전히 같지는 않다.  이건 **눈으로
    보는** 도구이고 정식 판정은 compare_policy.py --task cmdvel 이다.
    """
    ik = RoverEnv(**env_kw, seed=seed)
    rl = RoverEnv(**env_kw, seed=seed)
    zero = np.zeros(ik.action_space.shape[0], np.float32)
    acc = {"ik": [], "rl": []}
    st = {"pause": False, "reset": False, "rt": args.realtime}

    def on_key(k):
        c = chr(k).lower() if 32 <= k < 127 else ""
        if c == "p":
            st["pause"] = not st["pause"]
        elif c == "r":
            st["reset"] = True
        elif c == ".":
            st["rt"] = min(st["rt"] * 1.5, 16.0); print("[view] 배속 %.2f" % st["rt"])
        elif c == ",":
            st["rt"] = max(st["rt"] / 1.5, 0.05); print("[view] 배속 %.2f" % st["rt"])

    def draw(v):
        scn = v.user_scn
        n = 0
        if ik.task != "cmdvel" and getattr(ik, "path", None) is not None:
            P = ik.path.P
            zt = float(ik.d.qpos[2]) + 0.05
            for i in range(len(P) - 1):
                if n >= scn.maxgeom - 80:
                    break
                g = scn.geoms[n]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                    np.zeros(3), np.eye(3).ravel(),
                                    np.array([0.1, 0.4, 1.0, 0.9], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012,
                                     np.array([P[i][0], P[i][1], zt]),
                                     np.array([P[i+1][0], P[i+1][1], zt]))
                n += 1
        n = draw_ghost(rl, scn, n, offset=(args.ghost_offset, 0.0))
        n = draw_twist(ik, scn, n)
        scn.ngeom = n

    def summary():
        out = []
        for tag, key, env in (("순수IK", "ik", ik), ("정책", "rl", rl)):
            a = np.asarray(acc[key]) if acc[key] else None
            if a is None:
                continue
            if env.task == "cmdvel":
                out.append("%s v %5.1fmm/s om %4.2f°/s"
                           % (tag, a[:, 0].mean() * env.v_max * 1000,
                              np.degrees(a[:, 1].mean() * env.om_max)))
            else:
                out.append("%s e_y %5.1fmm e_psi %4.1f°"
                           % (tag, a[:, 0].mean() * 1000, np.degrees(a[:, 1].mean())))
        if acc["ik"] and acc["rl"]:
            a0, a1 = np.asarray(acc["ik"]), np.asarray(acc["rl"])
            k = min(len(a0), len(a1))
            sc = (ik.v_max * 1000) if ik.task == "cmdvel" else 1000.0
            dv = (a1[:k, 0].mean() - a0[:k, 0].mean()) * sc
            out.append("차이 %+.1f%s %s (n=%d)"
                       % (dv, "mm/s" if ik.task == "cmdvel" else "mm",
                          "정책승" if dv < 0 else "IK승", k))
        return "  |  ".join(out)

    def tick():
        """한 제어주기.  에피소드가 끝났으면 True."""
        obs_rl = tick.obs_rl
        _, _, t1, u1, _ = ik.step(zero)
        tick.obs_rl, _, t2, u2, _ = rl.step(np.asarray(policy(obs_rl), np.float32))
        for key, env in (("ik", ik), ("rl", rl)):
            acc[key].append(env.vel_err_hist[-1] if env.task == "cmdvel"
                            else (abs(env.e_y), abs(env.e_psi)))
        if ik.k % CTRL_HZ == 0:
            print("[view] t %4.1fs | %s" % (ik.k / CTRL_HZ, summary()), flush=True)
        # 한쪽이 먼저 끝나면(전복·고착·완주) 둘 다 끊는다.  그래야 비교
        # 구간의 스텝 수가 같다.
        return bool(t1 or u1 or t2 or u2)

    ep = [seed]

    def reset_both():
        ik.reset(seed=ep[0])
        tick.obs_rl, _ = rl.reset(seed=ep[0])
        acc["ik"].clear(); acc["rl"].clear()

    reset_both()
    print("[view] IK vs 정책 비교 | 같은 시드 | 실체 = 순수 IK, "
          "**반투명 주황 고스트 = 정책**")
    if args.ghost_offset:
        print("[view] 고스트를 y 로 %+.2fm 밀어 그린다" % args.ghost_offset)
    if args.headless:
        print("[view] headless: 창 없이 시드 %d..%d 를 돌린다"
              % (seed, seed + args.episodes - 1))
        for i in range(args.episodes):
            ep[0] = seed + i
            reset_both()
            while not tick():
                pass
            print("[view] 시드 %d 끝 | %s\n" % (ep[0], summary()), flush=True)
        ik.close(); rl.close()
        return

    print("[view] 화살표: 파랑 명령 선속도 · 주황 실제 선속도 · "
          "초록 명령 요속도(수직) · 노랑 실제 요속도")
    print("[view] 키: p 정지 · r 다음 시드 · , . 배속")
    dt = 1.0 / CTRL_HZ
    with mujoco.viewer.launch_passive(ik.m, ik.d, key_callback=on_key) as v:
        v.cam.distance = 3.0; v.cam.elevation = -25
        v.cam.lookat[:] = ik.d.qpos[:3]
        draw(v)
        while v.is_running():
            t0 = time.time()
            if st["reset"]:
                st["reset"] = False
                ep[0] += 1
                reset_both()
                # 지형은 self.m 에 제자리로 다시 쓰인다.  GPU 쪽은 자동으로
                # 갱신되지 않으므로 직접 올려야 옛 지형이 남지 않는다.
                v.update_hfield(0)
                v.cam.lookat[:] = ik.d.qpos[:3]
                print("[view] 시드 %d" % ep[0])
            if st["pause"]:
                mujoco.mj_forward(ik.m, ik.d)
            elif tick():
                print("[view] 에피소드 끝 | %s" % summary())
                st["reset"] = True
            draw(v)
            v.sync()
            lag = dt / max(st["rt"], 1e-3) - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)
    ik.close(); rl.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", default="sand", choices=["sand", "rock", "proc"],
                   help="sand/rock=실측 STL, proc=절차생성 학습지형")
    p.add_argument("--d", type=float, default=1.0, help="난이도")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--model", default=None, help="학습된 정책 (없으면 순수 IK)")
    source.add_argument("--watch-run", default=None,
                        help="학습 run의 최신 모델·정규화 쌍을 에피소드마다 자동 로드")
    p.add_argument("--vecnorm", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--path", default=None,
                   help="draw_path.py 로 찍은 경로 파일(.npy).  주면 그 경로만 달린다")
    p.add_argument("--stage1", action="store_true",
                   help="학습과 같은 조건 (경로 세트 고정, 랜덤화 OFF)")
    p.add_argument("--path-seed", type=int, default=1234)
    p.add_argument("--n-paths", type=int, default=10)
    p.add_argument("--static", action="store_true", help="물리 안 돌리고 지형만 본다")
    # 시각 전용 CAD 메시.  뷰어에서는 **반드시 켜야 한다**: 끄면 남는 geom 이
    # 전부 group=3 이고 MuJoCo 뷰어는 group 3 을 기본으로 안 그려서 로버가
    # 화면에서 사라진다 (바닥만 보인다).  물리는 양쪽이 비트 단위로 동일하다
    # (충돌 geom 29개·질량·관성·궤적 전부 같음) -- 학습은 메모리 때문에 끈다.
    p.add_argument("--compare-ik", action="store_true",
                   help="순수 IK 와 정책을 같은 시드로 겹쳐 돌린다. 창은 하나이고 "
                        "정책이 **반투명 주황 고스트**로 겹쳐 그려진다. --model 필요")
    p.add_argument("--ghost-offset", type=float, default=0.0,
                   help="고스트를 y 로 밀어 그린다 [m]. 0=겹침(궤적 벌어짐이 "
                        "잘 보인다), 0.5 쯤 주면 나란히 본다")
    p.add_argument("--headless", action="store_true",
                   help="--compare-ik 에서 창 없이 터미널 숫자만 (훨씬 빠르다)")
    p.add_argument("--episodes", type=int, default=5,
                   help="--compare-ik --headless 로 돌릴 에피소드 수")
    p.add_argument("--no-visual", action="store_true",
                   help="시각 메시 없이 띄운다 (충돌체만, 뷰어 `3` 키로 표시)")
    # cmdvel(밑단 속도추종) 과제.  경로가 없으므로 랜덤 twist 를 추종한다.
    p.add_argument("--task", default="path", choices=("path", "cmdvel"),
                   help="학습 때와 **같은** 값이어야 한다")
    p.add_argument("--history-frames", type=int, default=HISTORY,
                   help="학습 시 관측 이력 프레임 수 (s51 등 cmdvel teacher 는 5)")
    p.add_argument("--cmd-filter-tau", type=float, default=0.28,
                   help="cmdvel 목표속도 필터 [s]; 0.10=ROS 기본값·s45 이후 학습값")
    p.add_argument("--soil-amp", type=float, default=None,
                   help="바퀴별 모래 마찰 비대칭 진폭 (미지정이면 randomize 종속)")
    p.add_argument("--episode-s", type=float, default=None,
                   help="에피소드 길이 [s]; cmdvel 기본 8, path 기본은 env 기본값")
    # nargs="?" 로 두면 `--realtime` 만 써도 1.0 이 된다 (값을 안 적으면 다음
    # 인자를 값으로 먹으려 해서 "expected one argument" 가 났다).
    p.add_argument("--drive-align-gate-deg", type=float, default=10.0,
                   help="구동 정렬 게이트 해제 오차 [deg], 0=게이트 끔. "
                        "뷰어에서 g 키로 실시간 토글 가능")
    p.add_argument("--steer-limit-deg", type=float, default=None,
                   help="조향 가동범위 ±[deg] 덮어쓰기 (기본 config 의 ±90). "
                        "±110 이상은 모터가 타이어보다 바깥으로 나간다")
    p.add_argument("--realtime", type=float, nargs="?", const=1.0, default=1.0,
                   help="재생 배속 (기본 1.0). `--realtime 0.5` 로 느리게")
    args = p.parse_args()

    if args.watch_run and args.vecnorm:
        p.error("--watch-run 은 체크포인트와 짝인 vecnorm을 자동 선택합니다")
    if args.watch_run and not (args.stage1 or args.path):
        p.error("--watch-run 은 모델을 재생할 고정 경로가 필요합니다: --stage1 또는 --path")
    if args.watch_run and args.static:
        p.error("--watch-run 은 주행 재생용이므로 --static 과 함께 사용할 수 없습니다")
    # 정책과 통계는 한 쌍으로 교체한다. 학습 중 파일을 쓰는 순간 읽었다면
    # 다음 에피소드에서 재시도하고, 주행 중에는 정책을 바꾸지 않는다.
    from stable_baselines3 import PPO
    import pickle
    loaded = {"model": None, "norm": None, "pair": None}

    def load_policy(model_path, norm_path=None, expected_obs_dim=None):
        sb = PPO.load(str(model_path), device="cpu")
        if (expected_obs_dim is not None and
                sb.observation_space.shape != (expected_obs_dim,)):
            raise ValueError(f"정책 관측 {sb.observation_space.shape}와 뷰어 환경 "
                             f"({expected_obs_dim},)이 다릅니다")
        stats = None
        if norm_path is not None:
            with open(norm_path, "rb") as f:
                stats = pickle.load(f)
            if stats.obs_rms.mean.shape != sb.observation_space.shape:
                raise ValueError("모델과 관측 정규화 통계의 차원이 다릅니다")
        loaded.update(model=sb, norm=stats, pair=(pathlib.Path(model_path), norm_path))
        print(f"[view] 정책 로드: {model_path} ({sb.num_timesteps:,} steps)")

    if args.model:
        load_policy(args.model, args.vecnorm)
    elif args.watch_run:
        pair = latest_checkpoint_pair(args.watch_run)
        if pair is not None:
            try:
                load_policy(*pair)
            except Exception as exc:
                print(f"[view] 체크포인트 로드 대기: {exc}")
        else:
            print(f"[view] {args.watch_run}: 첫 체크포인트 대기 중 (그동안 순수 IK)")

    # 모델을 먼저 읽어 teacher 정책의 특권 관측 차원을 확인한다.
    # OBS_DIM(= OBS_PER_FRAME * HISTORY) 을 그대로 쓰면 이력 프레임이 다른
    # 모델(s45/s51 은 5프레임)에서 특권 여부를 잘못 판정한다.
    regular_dim = OBS_PER_FRAME * args.history_frames
    priv = (loaded["model"] is not None and
            int(loaded["model"].observation_space.shape[0]) > regular_dim)

    def policy(o):
        sb, norm = loaded["model"], loaded["norm"]
        if norm is not None:
            o = np.clip((o - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + 1e-8),
                        -10, 10).astype(np.float32)
        return sb.predict(o, deterministic=True)[0]

    arena = args.terrain in ("sand", "rock")
    # 경로 지정 / Stage 1 조건 재현
    paths = None; s0f = None
    if args.path:
        import path as pth
        paths = [pth.RefPath(np.load(args.path))]
        s0f = 0.0                     # 지정 경로는 시작점부터
        print("경로 %s: 꼭짓점 %d, 길이 %.2fm" % (args.path, len(paths[0].P), paths[0].total))
    elif args.stage1:
        import terrain as terr, path as pth
        from config import RoverCfg
        kind = args.terrain if args.terrain != "proc" else "sand"
        Z = terr.load_arena(kind); ex, ey = terr.eval_extent(kind)
        cfg = RoverCfg()
        drive, _ = terr.drivable_mask(Z, ex, ey, max_slope_deg=cfg.max_slope_deg)
        route = terr.center_clearance_mask(drive, ex, ey, cfg.route_clearance)
        paths = pth.make_path_set(args.n_paths, route, Z, ex, ey, seed=args.path_seed)
        print("Stage 1 경로 세트 %d개 (seed %d)" % (args.n_paths, args.path_seed))
    cfg = None
    if args.steer_limit_deg is not None:
        if not (90.0 <= args.steer_limit_deg <= 180.0):
            p.error("--steer-limit-deg 는 90~180 사이여야 한다 "
                    "(90 미만이면 낼 수 없는 방향이 생긴다)")
        lim = float(args.steer_limit_deg)
        cfg = RoverCfg(steer_lo_deg=(-lim,) * 4, steer_hi_deg=(lim,) * 4)
    env_kw = dict(cfg=cfg, difficulty=args.d, task=args.task,
                  visual=not args.no_visual,
                  # 뷰어는 MjModel/MjData 객체가 바뀌면 안 된다 (env.py 주석).
                  rebuild_on_reset=False,
                  history_frames=args.history_frames,
                  cmd_filter_tau=args.cmd_filter_tau,
                  soil_amp=args.soil_amp,
                  drive_align_gate_deg=args.drive_align_gate_deg,
                  arena_eval=arena and args.task != "cmdvel",
                  eval_kind=(args.terrain if arena else "sand"),
                  privileged=priv, seed=args.seed,
                  paths=paths, s0_frac=s0f,
                  # --path / --stage1 은 학습 조건 재현이므로 랜덤화·교란을 끈다
                  randomize=not (args.path or args.stage1),
                  perturb=0.0)
    if args.episode_s is not None:
        env_kw["episode_s"] = args.episode_s
    elif args.task == "cmdvel":
        env_kw["episode_s"] = 8.0          # cmdvel 학습 기본값
    if args.compare_ik:
        if loaded["model"] is None:
            p.error("--compare-ik 는 --model 또는 --watch-run 이 있어야 합니다")
        if args.static:
            p.error("--compare-ik 와 --static 은 같이 쓸 수 없습니다")
        if args.episodes < 1:
            p.error("--episodes 는 양수여야 합니다")
        env_kw.pop("seed", None)
        run_compare(env_kw, policy, args, args.seed)
        return
    e = RoverEnv(**env_kw)
    obs, _ = e.reset(seed=args.seed)
    # 어떤 설정으로 보고 있는지 창을 열기 전에 찍는다.  조향범위·게이트는
    # 눈으로 구분이 안 되는데 결과를 크게 바꾼다 (±100 vs ±135, 게이트 ON/OFF).
    _lim = [round(float(np.degrees(x))) for x in e.st_hi_c]
    _gate = ("OFF" if e.drive_align_gate_rad is None
             else "%.0f도" % np.degrees(e.drive_align_gate_rad))
    print("[view] 조향범위 ±%s도 | 구동정렬게이트 %s | 명령 %s | 지형 %s | 배속 %.2f"
          % (_lim[0] if len(set(_lim)) == 1 else _lim, _gate,
             "pure pursuit + 정책잔차" if loaded["model"] else "pure pursuit (순수 IK)",
             args.terrain, args.realtime))
    print("[view] 키: p 정지 | r 리셋 | g 게이트토글 | o 잔차토글 | , 느리게 | . 빠르게")

    def maybe_reload():
        if not args.watch_run:
            return
        pair = latest_checkpoint_pair(args.watch_run)
        if pair is None or pair == loaded["pair"]:
            return
        was_empty = loaded["model"] is None
        try:
            load_policy(*pair, expected_obs_dim=e.observation_space.shape[0])
        except Exception as exc:
            # 학습 프로세스가 체크포인트를 쓰는 도중이면 다음 에피소드에서 재시도.
            print(f"[view] 새 체크포인트 로드 대기: {exc}")
            return
        if was_empty:
            st["resid"] = True
        print("[view] 다음 에피소드부터 새 정책 적용")

    tag = {"sand": "규사 경사지형 STL", "rock": "암석 착륙지 STL",
           "proc": f"절차생성 학습지형 d={args.d}"}[args.terrain]
    print(f"[view] {tag}  |  흙 mu_max {e.terr_p.mu_max:.2f}  모래두께 {e.terr_p.z_max*1000:.0f}mm")
    print(f"[view] 정책: {'순수 IK' if loaded['model'] is None else loaded['pair'][0]}"
          + ("  (특권 관측 teacher)" if priv else ""))
    print("[view] 좌드래그 회전 / 우드래그 이동 / 휠 줌")
    print("[view] 키: p 일시정지 · r 리셋(새 경로) · o 보정ON/OFF · , . 배속")

    # ---- 키보드 (passive viewer 는 마우스 콜백이 없고 key_callback 만 받는다) ----
    st = {"pause": False, "reset": False, "resid": loaded["model"] is not None,
          "rt": args.realtime}

    def on_key(k):
        c = chr(k).lower() if 32 <= k < 127 else ""
        if c == "p":
            st["pause"] = not st["pause"]; print("[view] " + ("일시정지" if st["pause"] else "재생"))
        elif c == "r":
            st["reset"] = True
        elif c == "o" and loaded["model"] is not None:
            st["resid"] = not st["resid"]
            print("[view] 잔차 보정 %s" % ("ON" if st["resid"] else "OFF (순수 IK)"))
        elif c == "g":
            # 같은 에피소드 안에서 켜고 끄며 눈으로 비교할 수 있게 한다.
            if e.drive_align_gate_rad is None:
                e.drive_align_gate_rad = float(np.radians(
                    args.drive_align_gate_deg or 10.0))
                print("[view] 구동 정렬 게이트 ON (%.0f도)"
                      % np.degrees(e.drive_align_gate_rad))
            else:
                e.drive_align_gate_rad = None
                print("[view] 구동 정렬 게이트 OFF")
        elif c == ".":
            st["rt"] = min(st["rt"] * 1.5, 16.0); print("[view] 배속 %.2f" % st["rt"])
        elif c == ",":
            st["rt"] = max(st["rt"] / 1.5, 0.05); print("[view] 배속 %.2f" % st["rt"])

    def draw_cmd(v):
        v.user_scn.ngeom = draw_twist(e, v.user_scn, 0)

    def draw_path(v):
        """기준경로를 뷰어에 그린다.  이게 없으면 무엇을 따라가는지 안 보인다."""
        if e.task == "cmdvel" or getattr(e, "path", None) is None:
            draw_cmd(v)
            return
        v.user_scn.ngeom = 0
        P = e.path.P
        zt = float(e.d.qpos[2]) + 0.05
        n = 0
        for i in range(len(P) - 1):
            if n >= v.user_scn.maxgeom - 2:
                break
            g = v.user_scn.geoms[n]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array([0.1, 0.4, 1.0, 0.9], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012,
                                 np.array([P[i][0], P[i][1], zt]),
                                 np.array([P[i+1][0], P[i+1][1], zt]))
            n += 1
        # 목표점 (마지막 꼭짓점)
        if n < v.user_scn.maxgeom:
            g = v.user_scn.geoms[n]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.05, 0.05, 0.05]),
                                np.array([P[-1][0], P[-1][1], zt]), np.eye(3).ravel(),
                                np.array([1.0, 0.3, 0.1, 0.9], np.float32))
            n += 1
        v.user_scn.ngeom = n

    # 잔차 ON/OFF 각각의 추종오차를 따로 누적한다.  이득이 병진 5.7mm/s
    # (v_max 378mm/s 의 1.5%) 라 **눈으로는 절대 안 보인다**.  같은 창에서
    # 'o' 로 토글하며 두 누적값을 비교하는 것이 유일하게 정직한 확인법이다.
    # 주의: 같은 에피소드 안에서 토글하면 상태가 갈라지므로 엄밀한 짝비교가
    # 아니다.  정식 판정은 compare_cmdvel_models.py / compare_policy.py 다.
    ab = {True: [], False: []}

    def ab_line():
        out = []
        for on in (True, False):
            d = ab[on]
            if not d:
                continue
            arr = np.asarray(d)
            tag = "잔차ON " if on else "순수IK"
            if e.task == "cmdvel":
                out.append("%s v %.1fmm/s om %.2f°/s (n=%d)"
                           % (tag, arr[:, 0].mean() * e.v_max * 1000,
                              np.degrees(arr[:, 1].mean() * e.om_max), len(arr)))
            else:
                out.append("%s e_y %.1fmm (n=%d)"
                           % (tag, arr[:, 0].mean() * 1000, len(arr)))
        return "  |  ".join(out)

    dt = 1.0 / CTRL_HZ
    episode_return = 0.0
    with mujoco.viewer.launch_passive(e.m, e.d, key_callback=on_key) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -25
        draw_path(v)
        while v.is_running():
            t0 = time.time()
            if st["reset"]:
                st["reset"] = False
                obs, _ = e.reset()
                episode_return = 0.0
                maybe_reload()
                v.update_hfield(0)     # 지형은 제자리로 다시 쓰인다 (GPU 갱신 필요)
                v.cam.lookat[:] = e.d.qpos[:3]
                draw_path(v)
                if e.task == "cmdvel":
                    print("[view] 리셋.  랜덤 twist 추종 (2초마다 새 명령)")
                else:
                    print("[view] 리셋.  경로 %.2fm  s0 %.2f" % (e.path.total, e.s0))
            if args.static or st["pause"]:
                mujoco.mj_forward(e.m, e.d)
            else:
                a = (np.zeros(e.action_space.shape[0], np.float32)
                     if (loaded["model"] is None or not st["resid"])
                     else policy(obs))
                resid_on = bool(loaded["model"] is not None and st["resid"])
                obs, r, term, trunc, info = e.step(a)
                episode_return += float(r)
                if e.task == "cmdvel":
                    ab[resid_on].append(e.vel_err_hist[-1])
                else:
                    ab[resid_on].append((abs(e.e_y), abs(e.e_psi)))
                if e.k % CTRL_HZ == 0:
                    if e.task == "cmdvel":
                        cmd, act = e.cmd_nom, e.body_twist()
                        ve, ome = e.vel_err_hist[-1]
                        print("[view] t %4.1fs | 명령 vx%+.3f vy%+.3f om%+.2f | "
                              "실제 vx%+.3f vy%+.3f om%+.2f | 오차 v %4.1fmm/s "
                              "om %4.1f°/s | max|a| %.2f  %s"
                              % (e.k / CTRL_HZ, cmd[0], cmd[1], cmd[2],
                                 act[0], act[1], act[2],
                                 ve * e.v_max * 1000,
                                 np.degrees(ome * e.om_max),
                                 np.max(np.abs(a)) if a.size else 0.0, ab_line()))
                    elif args.watch_run:
                        print("[view] t %.1fs | e_y %.0fmm | heading %.1f° | "
                              "gate %.2f | reward %+.3f | max|a| %.2f" %
                              (e.k / CTRL_HZ, 1000 * info["e_y"], info["e_psi"],
                               info["gate"], r, np.max(np.abs(a))))
                if term or trunc:
                    if e.task == "cmdvel":
                        print("[view] 끝: 전복 %s 고착 %s | %.1fs | 병진오차 "
                              "평균 %.1f p90 %.1f mm/s | 요오차 평균 %.1f "
                              "p90 %.1f °/s | 슬립 %.2f 침하 %.0f%%zmax | "
                              "a %.2f | 총보상 %.1f\n[view] 누적 %s"
                              % (info["tip"], info["stuck"], info["t_elapsed"],
                                 info["v_err_m"] * e.v_max * 1000,
                                 info["v_err_p90"] * e.v_max * 1000,
                                 np.degrees(info["om_err_m"] * e.om_max),
                                 np.degrees(info["om_err_p90"] * e.om_max),
                                 info["slip_m"], 100 * info["sink_m"],
                                 info["a_use"], episode_return, ab_line()))
                    else:
                        print("[view] 끝: 성공 %s 전복 %s 고착 %s | %.1fs %.2fm | "
                              "CTE 평균 %.0fmm p90 %.0fmm | 기수 %.1f도 | "
                              "a %.2f | 총보상 %.1f"
                              % (info["success"], info["tip"], info["stuck"],
                                 info["t_elapsed"], info["dist"],
                                 1000 * info["e_y_m"], 1000 * info["e_y_p90"],
                                 info["e_psi_m"], info["a_use"], episode_return))
                    obs, _ = e.reset()
                    episode_return = 0.0
                    maybe_reload()
                    v.update_hfield(0)
                    v.cam.lookat[:] = e.d.qpos[:3]
                    draw_path(v)
            if e.task == "cmdvel":
                draw_cmd(v)          # 명령은 매 스텝 바뀐다
            v.sync()
            lag = dt / max(st["rt"], 1e-3) - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()

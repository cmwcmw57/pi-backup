#!/usr/bin/env python3
"""teacher 진단 -- 증류를 시작해도 되는지 판정한다.

  python3 probe_teacher.py --model runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl

성공률만 보면 안 된다.  PPO 가 인코더를 무시하고 고유수용감각만으로 풀어버리면
z 가 죽은 변수가 되고, 성공률은 멀쩡한데 **증류할 내용이 없다**.  그래서 세 가지를
따로 잰다:

  [1] z 가 살아있나   -- 에피소드 간 분산.  0 이면 인코더가 상수를 뱉는 것
  [2] z 가 흙을 담나  -- z -> 흙 파라미터 선형 프로브 R^2
  [3] z 가 쓰이나     -- z 를 **다른 에피소드 것으로 바꿔치기** 했을 때 성능 낙폭

[3] 이 안 떨어지면 teacher-student 구조 자체가 무의미하므로 여기서 멈춰야 한다.

검정 설계에서 조심한 것:
  * 씨드를 맞춰 같은 지형/흙을 쓰므로 **쌍대 비교**다.  평균 차이만 보면
    24 에피소드에서 성공률 표준오차가 10%p 라 8%p 임계값이 노이즈에 묻힌다.
    에피소드별 차이의 평균과 표준오차를 같이 낸다.
  * 주 판정은 이진 성공률이 아니라 **연속 지표**(목표 도달 수, 이동거리)로 한다.
    같은 표본 수에서 검정력이 훨씬 높다.
  * z=0 은 정책이 본 적 없는 분포 밖 입력이라, 성능이 떨어져도 "정보를 잃어서"
    인지 "이상한 입력이라"인지 못 가른다.  그래서 **바꿔치기가 주 검정**이고
    z=0 은 보조다.
  * 바꿔치기는 다른 에피소드의 **같은 시각 z** 를 쓴다.  에피소드 평균을 상수로
    넣으면 "틀린 z" 와 "상수 z" 효과가 섞인다 (특권정보에 슬립·침하·지형스캔
    같은 빠른 상태가 있어 z 는 시간에 따라 변한다).
"""
import argparse, pathlib, pickle, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import RoverEnv, PRIV_DIM
from nets import PROPRIO_DIM

SOIL = ["mu_max", "K", "alpha", "beta", "c_r", "k_z", "z_max", "mu_lat"]


def norm(o, rms, clip=10.0):
    return np.clip((o - rms.mean) / np.sqrt(rms.var + 1e-8), -clip, clip).astype(np.float32)


def rollout(tp, enc, rms, n, arena, difficulty, mode, rng, episode_s=20.0,
            seed0=1000, z_bank=None):
    """mode: 'normal' | 'zero' | 'swap'(다른 에피소드의 같은 시각 z)"""
    out = []; zs = []; soils = []; traj = []
    for i in range(n):
        e = RoverEnv(difficulty=difficulty,
                     episode_s=episode_s, arena_eval=arena, eval_kind="sand",
                     privileged=True, seed=seed0 + i)
        o, _ = e.reset(seed=seed0 + i)
        p = e.terr_p
        soils.append([p.mu_max, p.K, p.alpha, p.beta, p.c_r, p.k_z, p.z_max, p.mu_lat])
        # 바꿔치기용: 이 에피소드와 **다른** 에피소드의 z 궤적
        donor = None
        if mode == "swap" and z_bank:
            cand = [k for k in range(len(z_bank)) if k != i] or list(range(len(z_bank)))
            donor = z_bank[int(rng.choice(cand))]
        ep_z = []; k = 0
        term = trunc = False; info = {}
        while not (term or trunc):
            on = norm(o, rms)
            pro = torch.as_tensor(on[:PROPRIO_DIM]).reshape(1, -1)
            with torch.no_grad():
                z = enc(torch.as_tensor(on[PROPRIO_DIM:]).reshape(1, -1))
                ep_z.append(z.numpy().ravel().copy())
                if mode == "zero":
                    z = torch.zeros_like(z)
                elif donor is not None:
                    z = torch.as_tensor(donor[min(k, len(donor) - 1)]).reshape(1, -1)
                feat = torch.cat([pro, z], 1)
                a = tp.action_net(tp.mlp_extractor.forward_actor(feat)).squeeze(0).numpy()
            o, r, term, trunc, info = e.step(a); k += 1
        traj.append(np.array(ep_z, np.float32))
        zs.append(np.mean(ep_z, axis=0))
        out.append((info["success"], info["stuck"], info["tip"], info["dist"],
                    info["slip"], info["sink"], info["goals"]))
    return np.array(out, float), np.array(zs), np.array(soils), traj


def paired(a, b, col):
    """쌍대 차이(정상 - 변형)의 평균과 표준오차."""
    d = a[:, col] - b[:, col]
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/teacher/final")
    p.add_argument("--vecnorm", default="runs/teacher/vecnorm.pkl")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--difficulty", type=float, default=1.0)
    # 예전엔 default=True 라 끌 방법이 없었다.  정책이 바닥을 치는 지형에서 재면
    # z 를 망가뜨려도 잃을 성능이 없어 [3] 이 항상 "노이즈 수준"으로 나온다.
    # z 사용 여부는 **정책이 실제로 작동하는 구간**에서 재야 한다.
    p.add_argument("--arena", action="store_true",
                   help="실측 규사 STL 로 평가 (기본: 절차생성 지형)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    with open(args.vecnorm, "rb") as f:
        rms = pickle.load(f).obs_rms
    tp = model.policy
    enc = tp.features_extractor.encoder
    rng = np.random.default_rng(args.seed)

    # 정상 롤아웃을 먼저 돌려 z 궤적을 모은다 (바꿔치기의 재료).
    res = {}
    r0, zs, soils, bank = rollout(tp, enc, rms, args.episodes, args.arena,
                                  args.difficulty, "normal", rng)
    res["normal"] = r0
    for mode in ("zero", "swap"):
        r, _, _, _ = rollout(tp, enc, rms, args.episodes, args.arena,
                             args.difficulty, mode, rng, z_bank=bank)
        res[mode] = r

    print(f"규사 STL, d={args.difficulty}, {args.episodes} 에피소드\n")
    print(f"{'z 처리':14s} {'성공%':>6s} {'고착%':>6s} {'전복%':>6s} "
          f"{'이동m':>7s} {'슬립':>7s} {'목표':>6s}")
    for mode, label in (("normal", "정상"), ("zero", "z=0 (보조)"),
                        ("swap", "z 바꿔치기 (주)")):
        r = res[mode]
        print(f"{label:14s} {r[:,0].mean()*100:6.0f} {r[:,1].mean()*100:6.0f} "
              f"{r[:,2].mean()*100:6.0f} {r[:,3].mean():7.2f} {r[:,4].mean():7.3f} "
              f"{r[:,6].mean():6.2f}")

    print(f"\n[1] z 가 살아있나 -- 에피소드 간 표준편차 (차원별):")
    print("    " + "  ".join(f"{v:.3f}" for v in zs.std(axis=0)))
    if zs.std(axis=0).max() < 0.02:
        print("    -> 인코더가 사실상 상수를 뱉는다. 특권정보를 안 쓰고 있다.")

    print(f"\n[2] z 가 흙을 담나 -- z(8) -> 흙 파라미터 선형회귀 R^2:")
    A = np.concatenate([zs, np.ones((len(zs), 1))], axis=1)
    for k, name in enumerate(SOIL):
        y = soils[:, k]
        if y.std() < 1e-9:
            print(f"    {name:8s}   (분산 없음)"); continue
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1.0 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        bar = "#" * int(max(r2, 0) * 30)
        print(f"    {name:8s} {r2:6.3f}  {bar}")

    print(f"\n[3] z 가 쓰이나 -- 쌍대 차이(정상 - 변형), ± 는 표준오차 "
          f"({args.episodes} 에피소드, 같은 지형/흙):")
    COLS = [(6, "목표 도달수", 1.0), (3, "이동거리 m", 1.0),
            (0, "성공률 %p", 100.0), (4, "슬립(낮을수록 좋음)", -1.0)]
    for mode, label in (("swap", "z 바꿔치기(주)"), ("zero", "z=0 (보조)")):
        print(f"  [{label}]")
        for col, name, sc in COLS:
            m, se = paired(res["normal"], res[mode], col)
            m, se = m * abs(sc), se * abs(sc)
            if sc < 0:
                m = -m
            sig = "유의" if abs(m) > 2 * se else "노이즈 수준"
            print(f"    {name:22s} {m:+7.2f} ± {se:5.2f}   {sig}")
    m, se = paired(res["normal"], res["swap"], 6)
    print(f"\n  판정: ", end="")
    if m > 2 * se and m > 0:
        print("z 를 실제로 쓴다 -> 증류로 진행")
    else:
        print("z 를 거의 안 쓴다 -> 증류해도 얻을 게 없다. 인코더 병목/관측 노이즈를\n"
              "        먼저 손봐야 한다 (아래 [2] 의 R^2 이 높아도 마찬가지다 --\n"
              "        흙을 담고 있어도 정책이 참조하지 않으면 소용없다).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Teacher -> student 증류 (RMA 2단계).

  python3 distill.py --teacher runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl

student 는 **자기 정책으로 굴러다니면서** 그때 방문한 상태에서 z 를 맞히도록
배운다 (DAgger).  teacher 궤적만 모아서 지도학습하면 배포 때 student 가 만드는
상태 분포가 학습 분포와 달라져 누적오차로 무너진다.

정책 머리는 동결이고 adapt 만 학습한다.  그래서 필요한 손실은 z 회귀 하나뿐이다:
    L = MSE( adapt(고유수용감각), encoder(특권).detach() )
행동 손실도 같이 걸 수 있게 열어뒀다 (--w-act).
"""
import argparse, pathlib, pickle, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import RoverEnv, OBS_PER_FRAME, HISTORY, PRIV_DIM
from nets import AdaptationModule, StudentPolicy, PROPRIO_DIM


def norm_obs(o, rms, clip=10.0):
    return np.clip((o - rms.mean) / np.sqrt(rms.var + 1e-8), -clip, clip).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="runs/teacher/final")
    p.add_argument("--vecnorm", default="runs/teacher/vecnorm.pkl")
    p.add_argument("--out", default="runs/teacher/student.pt")
    p.add_argument("--iters", type=int, default=30, help="DAgger 반복")
    p.add_argument("--steps-per-iter", type=int, default=6000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-act", type=float, default=0.0, help="행동 손실 가중치")
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--buffer", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.teacher, device="cpu")
    with open(args.vecnorm, "rb") as f:
        rms = pickle.load(f).obs_rms

    tp = model.policy
    student = StudentPolicy(tp)
    opt = torch.optim.Adam(student.adapt.parameters(), lr=args.lr)
    enc = tp.features_extractor.encoder

    env = RoverEnv(difficulty=args.difficulty,
                   episode_s=args.episode_s, privileged=True, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    X = np.zeros((args.buffer, PROPRIO_DIM), np.float32)   # 정규화된 고유수용감각
    Y = np.zeros((args.buffer, PRIV_DIM), np.float32)      # 정규화된 특권정보
    n = 0; full = False

    print(f"teacher {args.teacher} | DAgger {args.iters} 회 x {args.steps_per_iter} 스텝")
    for it in range(args.iters):
        # --- 1) student 정책으로 굴러다니며 수집 ---------------------------
        t0 = time.time(); eps = 0; succ = 0; rew = 0.0
        for _ in range(args.steps_per_iter):
            on = norm_obs(obs, rms)
            pro = torch.as_tensor(on[:PROPRIO_DIM]).reshape(1, -1)
            # 초반에는 teacher 가 끌어준다 (student 가 아직 못 굴러 상태가 편향됨)
            beta = max(0.0, 1.0 - it / max(args.iters * 0.4, 1))
            with torch.no_grad():
                if np.random.rand() < beta:
                    z = enc(torch.as_tensor(on[PROPRIO_DIM:]).reshape(1, -1))
                    a = student.act(pro, z)
                else:
                    a = student.act(pro)
            X[n % args.buffer] = on[:PROPRIO_DIM]
            Y[n % args.buffer] = on[PROPRIO_DIM:]
            n += 1; full = full or n >= args.buffer
            obs, r, term, trunc, info = env.step(a.squeeze(0).numpy())
            rew += r
            if term or trunc:
                eps += 1; succ += int(info["success"]); obs, _ = env.reset()

        # --- 2) z 회귀 ------------------------------------------------------
        N = args.buffer if full else n
        Xt = torch.as_tensor(X[:N]); Yt = torch.as_tensor(Y[:N])
        with torch.no_grad():
            Zt = enc(Yt)
        losses = []
        for _ in range(args.epochs):
            perm = torch.randperm(N)
            for i in range(0, N, args.batch):
                idx = perm[i:i + args.batch]
                zx = student.adapt(Xt[idx])
                loss = nn.functional.mse_loss(zx, Zt[idx])
                if args.w_act > 0:
                    with torch.no_grad():
                        a_t = student.act(Xt[idx], Zt[idx])
                    loss = loss + args.w_act * nn.functional.mse_loss(
                        student.act(Xt[idx], zx), a_t)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss))
        print(f"[{it+1:2d}/{args.iters}] z손실 {np.mean(losses):.5f} | "
              f"수집 {N} | 에피 {eps} 성공 {succ/max(eps,1):.0%} | "
              f"beta {beta:.2f} | {time.time()-t0:.0f}s")

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapt": student.adapt.state_dict(),
                "teacher": args.teacher, "vecnorm": args.vecnorm}, out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()

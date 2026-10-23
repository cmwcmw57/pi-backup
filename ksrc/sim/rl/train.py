#!/usr/bin/env python3
"""PPO 학습.

  python3 train.py --steps 2000000 --envs 16 --terrain sand

알고리즘은 SB3 PPO 를 쓴다. 환경뿐 아니라 정책 업데이트 크기와 체크포인트
선택도 성능에 영향을 주므로, 최종 모델을 순수 IK 및 중간 모델과 비교한다.
"""
import argparse, os, pathlib, re, shutil
import hashlib, json, subprocess
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import HISTORY, RoverEnv
import terrain as terr
import path as pth
from config import RoverCfg
from reward import RewardCfg, VelRewardCfg


# gamma 만 CLI 로 받는다.  나머지를 여기 모아둔 이유는 **런 매니페스트에
# 기록하기 위해서**다.  s9~s18 캠페인에서 PPO 하이퍼가 어디에도 저장되지
# 않아 "한 번에 한 변수" 를 사후 검증할 수 없었다.
PPO_KW = dict(n_steps=512, batch_size=4096, n_epochs=6,
              learning_rate=3e-4, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.002)

# runs/<name>/src/ 에 복사해 두는 파일.  **환경 의미를 바꾸는 모듈 전부**가
# 들어가야 한다.  이전에는 reward.py/env.py 둘만 저장해서 config.py(잔차 권한·
# 잡음 모델), path.py(경로 생성기), terramech.py(토양역학), train.py(PPO 하이퍼),
# 그리고 측정도구(compare_policy.py/baseline.py) 가 추적되지 않았다 -- 캠페인
# 중간에 측정도구가 수정됐고 그래서 s9~s11 과 s12~s18 의 숫자를 한 줄에
# 세울 수 없다.
SNAPSHOT = ("reward.py", "env.py", "config.py", "path.py", "terrain.py",
            "terramech.py", "mjcf.py", "nets.py", "train.py",
            "compare_policy.py", "baseline.py")
# 이어 학습 때 "환경이 바뀌었다" 로 막아야 하는 파일.  train.py 와 측정도구는
# 환경 의미를 바꾸지 않으므로 제외한다 (해시는 매니페스트에 남는다).
SNAPSHOT_GUARD = ("reward.py", "env.py", "config.py", "path.py", "terrain.py",
                  "terramech.py", "mjcf.py")


def _git_rev():
    """재현을 위한 커밋 해시.  dirty 여부까지 남긴다."""
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=pathlib.Path(__file__).parent)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, timeout=5,
                               cwd=pathlib.Path(__file__).parent)
        if rev.returncode:
            return None
        return {"head": rev.stdout.strip(),
                "dirty": bool(dirty.stdout.strip()),
                "dirty_files": dirty.stdout.strip().splitlines()[:40]}
    except (OSError, subprocess.SubprocessError):
        return None


def write_manifest(out, args, resume_model, resume_stats):
    """런을 사후 감사할 수 있게 인자·하이퍼·소스해시를 한 파일에 남긴다."""
    here = pathlib.Path(__file__).parent
    digests = {}
    for name in SNAPSHOT:
        f = here / name
        if f.is_file():
            digests[name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    man = {
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, pathlib.Path) else v)
                 for k, v in vars(args).items()},
        "ppo": dict(PPO_KW, gamma=args.gamma,
                    effective_learning_rate=(args.resume_lr if args.resume_lr is not None
                                             else PPO_KW["learning_rate"]),
                    target_kl=args.resume_target_kl),
        "rollout_steps": PPO_KW["n_steps"] * args.envs,
        # 롤아웃이 batch_size 로 나눠지지 않으면 SB3 가 마지막 미니배치를 작게
        # 만든다.  s13(envs 14 -> 7168) 이 여기 걸려 있었고 s18(16 -> 8192) 과의
        # 비교를 교란했다.
        "minibatch_even": (PPO_KW["n_steps"] * args.envs) % PPO_KW["batch_size"] == 0,
        "updates_per_1M": 1_000_000 / (PPO_KW["n_steps"] * args.envs),
        "git": _git_rev(),
        "src_sha256": digests,
        "resume": {"model": str(resume_model) if resume_model else None,
                   "vecnorm": str(resume_stats) if resume_stats else None},
        "reward_cfg": vars(RewardCfg()),
        "vel_reward_cfg": vars(VelRewardCfg(w_tail=args.w_tail)),
    }
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return man


def stage1_paths(kind, seed, n, min_slope_deg=0.0, d_ep=0.0,
                 min_cross_slope_deg=0.0):
    """Stage 1 고정 경로 세트.  학습/평가는 **다른 seed** 여야 한다.

    2026-10-04: 예전에는 `drive, _ = drivable_mask(...)` 로 **경사 배열을 버려서**
    `make_path` 의 급사면 선호 분기(`slope_deg`/`d_ep`)가 한 번도 실행되지 않았다.
    그 결과 경로 경사 중앙값이 0도였고 목적 현상이 발생하지 않았다
    (`pth.make_path_set` docstring 참고).
    """
    Z = terr.load_arena(kind); ex, ey = terr.eval_extent(kind)
    cfg = RoverCfg()
    drive, slope = terr.drivable_mask(Z, ex, ey, max_slope_deg=cfg.max_slope_deg)
    route = terr.center_clearance_mask(drive, ex, ey, cfg.route_clearance)
    return pth.make_path_set(n, route, Z, ex, ey, seed=seed, slope=slope,
                             min_mean_slope_deg=min_slope_deg,
                             min_cross_slope_deg=min_cross_slope_deg,
                             slope_deg=(slope if d_ep > 0 else None), d_ep=d_ep,
                             cross_slope=min_cross_slope_deg > 0.0)


def resume_files(target):
    """Resolve a completed run or a checkpoint to its matching normalization."""
    path = pathlib.Path(target)
    if path.is_dir():
        final = path / "final.zip"
        if final.is_file() and (path / "vecnorm.pkl").is_file():
            return final, path / "vecnorm.pkl"
        checkpoints = []
        for model in path.glob("ppo_*_steps.zip"):
            match = re.fullmatch(r"ppo_(\d+)_steps\.zip", model.name)
            if match:
                stats = path / f"ppo_vecnormalize_{match.group(1)}_steps.pkl"
                if stats.is_file():
                    checkpoints.append((int(match.group(1)), model, stats))
        if checkpoints:
            _, model, stats = max(checkpoints, key=lambda item: item[0])
            return model, stats
        raise ValueError(f"복원할 모델·VecNormalize 쌍이 없습니다: {path}")
    model = path if path.suffix == ".zip" else path.with_suffix(".zip")
    if not model.is_file():
        raise ValueError(f"모델 파일이 없습니다: {model}")
    if model.name in ("final.zip", "model.zip"):
        stats = model.parent / "vecnorm.pkl"
    else:
        match = re.fullmatch(r"ppo_(\d+)_steps\.zip", model.name)
        if not match:
            raise ValueError("--resume 모델은 final.zip, model.zip 또는 ppo_N_steps.zip 이어야 합니다")
        stats = model.parent / f"ppo_vecnormalize_{match.group(1)}_steps.pkl"
    if not stats.is_file():
        raise ValueError(f"모델과 짝인 VecNormalize 파일이 없습니다: {stats}")
    return model, stats


def make_env(rank, args):
    def _f():
        paths = None
        # cmdvel 은 경로를 명령에 쓰지 않는다. 원본 규사맵 경로를 8x8m
        # 물리맵의 스폰으로 재사용하면 좌표가 어긋나므로 절차지형에서 생성한다.
        if args.stage1 and args.task == "path":
            # Stage 1: 경로 세트 고정 + 랜덤화/교란/커리큘럼 OFF.
            # "3D 차체 잔차가 경로추종 자체를 배우나" 만 본다.
            paths = stage1_paths(args.terrain if args.terrain != "proc" else "sand",
                                 args.path_seed, args.n_paths,
                                 min_slope_deg=args.min_slope_deg,
                                 d_ep=args.path_d_ep)
        _cfg_kw = {}
        if args.d_rate_max is not None: _cfg_kw["d_rate_max"] = args.d_rate_max
        if args.wheel_closed_loop: _cfg_kw["wheel_closed_loop"] = True
        env = RoverEnv(cfg=(RoverCfg(**_cfg_kw) if _cfg_kw else None),
                       difficulty=args.d0,
                       difficulty_min_frac=args.difficulty_min_frac,
                       vel_rew=VelRewardCfg(w_tail=args.w_tail),
                       episode_s=args.episode_s,
                       privileged=args.teacher or args.teacher_direct,
                       seed=args.seed + rank,
                       # 대회맵으로 학습할 때: 거시 구조 고정 + 미세 요철 교란.
                       # 맵 하나에 고정하면 그 한 장의 요철까지 외우는데, 모래는
                       # 대회 당일까지 움직인다 (다른 팀 주행/갈퀴질/습도).
                       arena_eval=(args.terrain != "proc" and args.task == "path"),
                       eval_kind=(args.terrain if args.terrain != "proc" else "sand"),
                       perturb=(0.0 if args.terrain == "proc" else args.perturb),
                       paths=paths,
                       drive_align_gate_deg=args.drive_align_gate_deg,
                       discount_gamma=args.gamma,
                       randomize=((not args.stage1) if args.randomize is None
                                  else args.randomize),
                       soil_amp=args.soil_amp,
                       push_force_max=args.push_force,
                       cmd_filter_tau=args.cmd_filter_tau,
                       task=args.task, action_mode=args.action_mode,
                       velocity_obs=args.velocity_obs,
                       history_frames=args.history_frames)
        # Monitor 가 있어야 SB3 가 rollout/ep_rew_mean, ep_len_mean 을 찍는다.
        # make_vec_env() 헬퍼는 자동으로 씌워주지만 SubprocVecEnv 를 직접 만들면
        # 안 씌워진다 -- 그래서 터미널에 손실만 나오고 보상이 안 나왔다.
        return Monitor(env)
    return _f


class CurriculumCallback(BaseCallback):
    """에피소드 성공률이 임계를 넘으면 지형 난이도를 올린다.

    30도 경사에서 시작하면 정책이 단 한 번도 성공을 못 해 보상 신호가 0 이고
    영원히 학습이 안 된다.  쉬운 데서 시작해 서서히 올리는 게 필수다."""

    def __init__(self, d0=0.0, step=0.05, thresh=0.60, window=100, verbose=0):
        super().__init__(verbose)
        self.d = d0; self.step = step; self.thresh = thresh
        self.window = window; self.buf = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.buf.append(float(info.get("success", 0.0)))
        if len(self.buf) >= self.window:
            sr = float(np.mean(self.buf)); self.buf.clear()
            if sr >= self.thresh and self.d < 1.0:
                self.d = min(1.0, self.d + self.step)
                self.training_env.env_method("set_difficulty", self.d)
                if self.verbose:
                    print(f"[curriculum] 성공률 {sr:.0%} -> 난이도 {self.d:.2f}")
            self.logger.record("curriculum/difficulty", self.d)
            self.logger.record("curriculum/success_rate", sr)
        return True


class ActionStdCallback(BaseCallback):
    """롤아웃마다 현재 탐색 노이즈 std 를 모든 env 에 밀어넣는다.

    > [!caution] 2026-10-04 **비활성.**  구현이 틀렸다 -- 행동 클리핑을 빼먹었다
    > SB3 는 행동을 `[-1,1]` 로 **클리핑해서** env 에 넘기므로 실제 `E[a^2]` 가
    > `std^2` 보다 훨씬 작다 (std 1.0 에서 0.516, 0.63 에서 0.322).  `std^2` 를
    > 그대로 차감하면 **과차감**이 되어 벌점이 보너스로 바뀐다:
    >     std 1.00 -> 0.484 과차감 -> w_resid 0.2 x 0.484 = +0.097/스텝 보너스
    > 실측 `s17` 의 `reward/resid` 가 +40.66 (ep_len 460 -> +0.088/스텝) 로
    > 정확히 그 값이었고, s13 대비 e_y 가 0.0213 -> 0.0262 로 나빠졌다.
    >
    > 제대로 하려면 **클리핑된 정규분포의 2차 모멘트**를 써야 하는데, 정책 평균이
    > 0 이 아니면 클리핑이 평균과 상호작용해서 닫힌 형태가 없다 (수치적으로
    > 구해야 한다).  또는 애초에 정책 평균을 env 에 따로 넘겨야 한다.
    > 지금은 `reward.py` 를 s13 스냅샷으로 되돌렸고 이 콜백은 등록하지 않는다.
    > 원래 문제(resid/smooth 가 노이즈를 벌한다)는 **여전히 유효한 미해결 항목**이다.
    

    왜 필요한가 (2026-10-04): PPO 가 env 에 넘기는 행동은 **정책 평균이 아니라
    샘플**이라, `reward` 의 `resid`(mean(a^2))·`smooth`(mean(d2a^2)) 가 평균 0
    정책에서도 각각 `std^2`, `6*std^2` 만큼 벌점을 냈다.  실측에서 `resid` 벌점의
    **58%** 가 이 노이즈 기여분이었고, std 가 학습 중 0.98 -> 0.42 로 줄면서
    벌점이 저절로 약해졌다.  env 에 std 를 알려주면 그만큼 차감해 **정책 자신의
    행동 크기**만 벌할 수 있다 (`reward.tracking_terms` docstring 참고).
    """

    def _on_rollout_start(self) -> None:
        try:
            std = float(th.exp(self.model.policy.log_std).mean().item())
        except AttributeError:          # 상태 의존 std 등 log_std 가 없는 정책
            return
        self.model.get_env().env_method("set_action_std", std)

    def _on_step(self) -> bool:
        return True


class MetricsCallback(BaseCallback):
    """에피소드가 끝날 때 info 를 모아 텐서보드에 올린다.

    SB3 기본 스칼라(loss/entropy/...)만으로는 "정책이 왜 못 가는지"를 못 본다.
    슬립·침하·배걸림·고착원인까지 같이 봐야 지형이 어려운 건지 정책이 못 하는
    건지 구분된다.
    """

    # **에피소드 전 구간 집계만 쓴다.**  예전엔 종료 스텝의 순간값을 모았고
    # (slip/sink/e_y/yaw/energy), 그러면 누적 비율인 f_belly 와 의미가 섞인다.
    # 실측 차이: 종료스텝 slip 0.149 vs 에피소드 평균 0.482 (3.2배).
    KEYS = ("success", "stuck", "tip", "oob", "goals", "dist", "frac", "goal_dist",
            # 경로추종 정확도 -- 논문과 같은 지표
            "e_y_m", "e_y_p90", "e_y_max", "e_psi_m", "e_psi_p90",
            "course_m", "course_p90", "t_elapsed",
            # 3D 잔차 진단. a_use/a_p95 가 ±1 에 박혀 있으면 정책 출력
            # 포화·학습 불안정·권한 부족을 구분해야 한다. cmd_sat/wheel_desat 은
            # 실현 가능한 차체/바퀴 명령 제한에 걸린 비율이다.
            "a_use", "a_p95", "wheel_use", "cmd_sat", "wheel_desat",
            # cmdvel 과제의 **본 지표**.  무차원(v_max / om_max 로 나눈 값).
            # 없으면 reward/track_* 에서 역산해야 하는데, 에피소드 합이라
            # Jensen 때문에 평균과 일치하지 않는다.
            "v_err_m", "v_err_p90", "om_err_m", "om_err_p90",
            "slip_m", "slip_p90", "sink_m", "prog_m", "pw_m", "e_J", "e_per_m",
            "f_belly", "f_lifted", "duty_sat", "duty_use", "drive_gate", "steer_use")
    CAUSES = ("배걸림-하중상실", "배걸림-접촉", "막힘", "슬립고착")
    TAGS = {"배걸림-하중상실": "lost_load", "배걸림-접촉": "belly",
            "막힘": "blocked", "슬립고착": "slip_stuck"}

    def __init__(self, window=200, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.buf = {k: [] for k in self.KEYS}
        self.rterm = {}          # 보상 항별 에피소드 합
        self.cause = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            for k in self.KEYS:
                if k in info:
                    self.buf[k].append(float(info[k]))
            for k, v in info.items():
                if k.startswith("rterm_"):
                    self.rterm.setdefault(k[6:], []).append(float(v))
            self.cause.append(info.get("cause", "-"))
        if len(self.cause) >= self.window:
            for k, v in self.buf.items():
                if v:
                    # e_per_m 은 경로 진행이 0.3m 미만인 에피소드에서 **의도적으로**
                    # nan 이다 (미터당 소비가 정의되지 않음).  np.mean 으로 묶으면
                    # 그 한 건이 배치 전체를 nan 으로 만든다.  유효한 것만 평균하고,
                    # 전부 nan 이면 그 주기에는 기록하지 않는다.
                    a = np.asarray(v, dtype=float)
                    ok = np.isfinite(a)
                    if ok.any():
                        self.logger.record(f"rollout/{k}", float(a[ok].mean()))
                    v.clear()
            # **조건부 평균 버그 (2026-10-04 수정).**  env 는 tip/stuck/oob 를
            # 그 일이 일어난 에피소드에만 terms 에 넣는다 (env.py 의 `if stuck:`).
            # np.mean(v) 은 그 에피소드들만 평균하므로 reward/stuck 이 항상
            # 정확히 r_stuck = -5.00 으로 찍혔고, 캠페인 내내 고착 벌점을
            # **40배 과대평가**해서 읽었다 (실제 기여 = -5.0 x 고착률 ~ -0.12).
            # 성분합이 ep_rew_mean 과 5~15 어긋났던 원인도 이것이다.
            # 없었던 에피소드는 0 으로 세야 하므로 합을 에피소드 수로 나눈다.
            n = len(self.cause)
            for k, v in self.rterm.items():
                if v:
                    self.logger.record(f"reward/{k}", float(np.sum(v) / n))
                    v.clear()
            for c in self.CAUSES:
                self.logger.record(f"stuck_cause/{self.TAGS[c]}",
                                   self.cause.count(c) / n)
            self.cause.clear()
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4_000_000)
    p.add_argument("--envs", type=int, default=os.cpu_count())
    p.add_argument("--d0", type=float, default=None,
                   help="시작 난이도. 기본 path=0, cmdvel=0.8 (속도추종은 고정 난이도)")
    p.add_argument("--difficulty-min-frac", type=float, default=0.6,
                   help="에피소드 난도 하한 비율; d0=1에서 0.9면 U(0.9,1.0). "
                        "기본 0.6은 기존 분포 유지")
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--checkpoint-every", type=int, default=200_000,
                   help="모델·VecNormalize 저장 간격 (전체 환경 스텝)")
    p.add_argument("--resume", metavar="RUN_OR_MODEL",
                   help="기존 run 디렉터리 또는 체크포인트 .zip에서 이어 학습")
    p.add_argument("--allow-source-change", action="store_true",
                   help="저장된 reward.py/env.py와 현재 코드가 달라도 이어 학습")
    p.add_argument("--reset-optimizer-on-resume", action="store_true",
                   help="모방학습으로 actor 가 바뀐 모델을 이어 학습할 때 Adam 상태 초기화")
    p.add_argument("--reset-reward-norm-on-resume", action="store_true",
                   help="보상이 바뀐 재학습에서 과거 VecNormalize 반환 통계를 초기화")
    p.add_argument("--resume-lr", type=float, default=None,
                   help="이어 학습 시 PPO 학습률 재설정 (모방 정책 미세조정용)")
    p.add_argument("--resume-target-kl", type=float, default=None,
                   help="이어 학습 시 근사 KL 상한; 큰 정책 갱신을 조기 중단")
    p.add_argument("--drive-align-gate-deg", type=float, default=10.0,
                   help="네 조향각 공통 구동 허용 오차 [deg], 0=게이트 끔")
    p.add_argument("--w-energy", type=float, default=None,
                   help="현재 미지원: 차축 부하 모델을 검증하기 전에는 보상에 넣지 않음")
    # gamma 는 **시간 지평**으로 정한다.  gamma = exp(-dt/tau).
    # 0.99 는 50Hz 에서 시상수 2초라 20초 과제의 완주 보너스가 739 스텝 뒤
    # 0.0006 배로 보인다 -- 사실상 없는 항이 된다.  0.998 = 시상수 10초.
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--randomize", dest="randomize", action="store_true",
                   default=None,
                   help="센서·구동·측위 잡음을 켠다. --stage1 이 경로 고정과 잡음 "
                        "끄기를 묶어버리므로, '고정 경로 + 잡음' 을 쓰려면 둘을 "
                        "같이 준다. 2026-10-04 측정: 잡음만으로 순수 IK 횡오차가 "
                        "15.99 -> 21.68mm (+36%%) 로 악화된다 -- 지형 외란(모래 "
                        "비대칭)과 달리 실제로 작동하는 유일한 외란이다")
    p.add_argument("--no-randomize", dest="randomize", action="store_false",
                   help="--stage1 없이도 잡음을 끈다")
    p.add_argument("--min-slope-deg", type=float, default=0.0,
                   help="경로가 지나는 평균 지형경사의 하한 [deg]. 0 이면 기존 동작(경사 무시). 2026-10-04 측정: 하한이 없으면 경로 경사 중앙값이 0도라 견인력 수요가 R/N=0.057 뿐이고, 바퀴별 mu 를 +-16%% 흔들어도 목적 현상이 발생하지 않는다. 8 이면 실제 평균 9.5도(수요 0.165)로 가용 mu(i) 0.17~0.39 와 같은 범위가 된다")
    p.add_argument("--path-d-ep", type=float, default=0.0,
                   help="make_path 의 급사면 선호 확률. min-slope-deg 와 함께 쓴다 (0.9 권장)")
    p.add_argument("--soil-amp", type=float, default=None,
                   help="모래 비대칭 장(바퀴별 mu 배율) 진폭. **이 RL 의 과제 본체**다 (terrain.soil_field 참고). --stage1 은 randomize 를 통째로 꺼서 이것까지 끄므로, Stage 1 에서 목적 현상을 쓰려면 이 값을 명시한다. 미지정이면 기존 동작(randomize 에 종속). 0.25 권장, 0.5 는 장이 음수가 된다")
    p.add_argument("--stage1", action="store_true",
                   help="Stage 1 sanity check: 경로 세트 고정, 랜덤화·교란·노이즈·"
                        "커리큘럼 전부 OFF.  3D 잔차가 경로추종을 배우는지만 본다")
    p.add_argument("--path-seed", type=int, default=1234, help="학습 경로 세트 seed")
    p.add_argument("--n-paths", type=int, default=10)
    p.add_argument("--terrain", default="proc", choices=["proc", "sand", "rock"],
                   help="path: proc=절차지형, sand/rock=대회맵. cmdvel: 넓은 절차지형에 "
                        "규사 물리를 적용하고 대회맵은 별도 평가")
    p.add_argument("--perturb", type=float, default=0.020,
                   help="대회맵 학습 시 미세 요철 진폭(표준편차) [m], 난이도에 비례")
    p.add_argument("--task", default="path", choices=("path", "cmdvel"),
                   help="path: pure pursuit 경로추종 (align/prog/gate 보상). "
                        "cmdvel: **밑단** 과제 -- 랜덤 차체속도를 변형지형에서 "
                        "달성한다.  경로/lookahead/진행량이 보상에서 빠지므로 "
                        "lookahead 튜닝과 경로 생성기 기하가 결과를 교란하지 "
                        "않는다 (2026-10-05: 경로추종에서 lookahead 를 0.264 -> "
                        "0.20 으로 맞추자 RL 의 횡오차 이득이 -1.88mm -> +0.18mm "
                        "로 소멸했다).")
    p.add_argument("--action-mode", default="body",
                   choices=("body", "wheel", "body+wheel", "res8", "direct8"),
                   help="IK 잔차 3/4/7D, res8=IK+조향·속도 잔차(8D), "
                        "direct8=IK 없이 조향·속도 직접(8D)")
    p.add_argument("--wheel-closed-loop", action="store_true",
                   help="바퀴 속도 폐루프(엔코더 PI) 켬. 안 켜면 개루프 duty")
    p.add_argument("--velocity-obs", action="store_true",
                   help="진단 전용: MuJoCo 참 차체속도를 관측에 추가. 실기 배포 금지")
    p.add_argument("--push-force", type=float, default=0.0,
                   help="에피소드마다 0.3초간 가하는 무작위 평면 외력 상한 [N]")
    p.add_argument("--cmd-filter-tau", type=float, default=0.28,
                   help="cmdvel 목표속도 필터 시정수 [s]; 0.28=기존 학습 동특성, "
                        "0.10=ROS 브리지 기본값")
    p.add_argument("--w-tail", type=float, default=0.0,
                   help="cmdvel 한정: 큰 정규화 속도오차의 이차 비용 계수 (기본 0)")
    p.add_argument("--history-frames", type=int, default=HISTORY,
                   help="관측 이력 프레임 수 (1..50, 기본 50=1초)")
    p.add_argument("--d-rate-max", type=float, default=None,
                   help="잔차 변화율 제한 [스텝당 |da|]. 속도 액션은 급격한 변화를 요구할 수 있어 실기에서 깨진 사례가 있다 (arXiv:2504.02420). 0.25 = 80ms 에 -1 -> +1. 미지정이면 제한 없음(기존 동작)")
    p.add_argument("--log-std-init", type=float, default=None,
                   help="정책 초기 log std. cmdvel 기본 -2.0 (std 0.135), "
                        "path 기본 SB3 값(0 -> std 1.0). "
                        "cmdvel 에서는 action scale 이 0.11/0.085/0.45 로 커서 "
                        "std 1.0 이면 초기 오차가 운용점의 3배까지 벌어지고 "
                        "exp(-(e/sigma)^2) 보상이 평평해진다.")
    p.add_argument("--resume-log-std", type=float, default=None,
                   help="이어 학습 시 정책 행동 탐색 log(std) 재설정; "
                        "행동 모방 초기값을 좁은 분포로 미세 조정할 때 사용")
    p.add_argument("--teacher", action="store_true",
                   help="특권 관측으로 teacher 학습 (이후 distill.py 로 student 증류)")
    p.add_argument("--teacher-direct", action="store_true",
                   help="진단 전용: 43D 특권정보를 잠재 압축 없이 정책에 직접 입력")
    args = p.parse_args()
    if args.task == "cmdvel" and args.log_std_init is None and not args.resume:
        args.log_std_init = -2.0
    if args.d0 is None:
        # 경로 완주율 기반 커리큘럼은 속도추종 과제에서 정의되지 않는다.
        # d=0 은 흙 비대칭과 경사가 모두 0 이라 학습할 외란도 사라진다.
        args.d0 = 0.8 if args.task == "cmdvel" and not args.stage1 else 0.0
    if not 0.0 <= args.d0 <= 1.0:
        p.error("--d0 는 [0, 1] 범위여야 합니다")
    if not np.isfinite(args.difficulty_min_frac) or not 0 <= args.difficulty_min_frac <= 1:
        p.error("--difficulty-min-frac 은 [0,1] 범위여야 합니다")
    if args.teacher and (args.action_mode != "body" or args.velocity_obs):
        p.error("teacher 네트워크는 현재 36D 프레임/3D 차체 행동 전용입니다")
    if args.teacher_direct and args.teacher:
        p.error("--teacher 와 --teacher-direct 중 하나만 사용하세요")
    if args.teacher_direct and args.velocity_obs:
        p.error("--teacher-direct 에 참속도를 중복 입력할 수 없습니다")
    if not 1 <= args.history_frames <= HISTORY:
        p.error(f"--history-frames 는 1..{HISTORY} 이어야 합니다")
    if not np.isfinite(args.w_tail) or args.w_tail < 0:
        p.error("--w-tail 은 0 이상의 유한한 값이어야 합니다")
    if args.task != "cmdvel" and args.w_tail != 0:
        p.error("--w-tail 은 cmdvel 과제에서만 사용합니다")
    if args.reset_optimizer_on_resume and not args.resume:
        p.error("--reset-optimizer-on-resume 은 --resume 과 함께 사용하세요")
    if args.reset_reward_norm_on_resume and not args.resume:
        p.error("--reset-reward-norm-on-resume 은 --resume 과 함께 사용하세요")
    if args.resume_lr is not None and (
            not args.resume or not np.isfinite(args.resume_lr) or args.resume_lr <= 0):
        p.error("--resume-lr 은 --resume 과 함께 쓰는 양의 유한한 값이어야 합니다")
    if args.resume_target_kl is not None and (
            not args.resume or not np.isfinite(args.resume_target_kl) or
            args.resume_target_kl <= 0):
        p.error("--resume-target-kl 은 --resume 과 함께 쓰는 양의 유한한 값이어야 합니다")
    if args.resume_log_std is not None and (
            not args.resume or not np.isfinite(args.resume_log_std)):
        p.error("--resume-log-std 는 --resume 과 함께 쓰는 유한한 값이어야 합니다")
    if args.steps < 1 or args.envs < 1 or args.checkpoint_every < 1:
        p.error("--steps, --envs, --checkpoint-every 는 1 이상이어야 합니다")
    if not np.isfinite(args.drive_align_gate_deg) or args.drive_align_gate_deg < 0:
        p.error("--drive-align-gate-deg 는 0 이상의 유한한 숫자여야 합니다")
    if not np.isfinite(args.push_force) or args.push_force < 0:
        p.error("--push-force 는 0 이상의 유한한 숫자여야 합니다")
    if not np.isfinite(args.cmd_filter_tau) or args.cmd_filter_tau < 0:
        p.error("--cmd-filter-tau 는 0 이상의 유한한 숫자여야 합니다")
    if args.w_energy is not None:
        p.error("--w-energy 는 현재 미지원입니다. 차축 부하 모델 검증 후 활성화하세요")
    if not np.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        p.error("--gamma 는 (0, 1] 범위의 유한한 값이어야 합니다")

    out = pathlib.Path(args.out)
    resume_model = resume_stats = None
    if args.resume:
        try:
            resume_model, resume_stats = resume_files(args.resume)
        except ValueError as exc:
            p.error(str(exc))
        source_dir = resume_model.parent
        if out.resolve() == source_dir.resolve():
            p.error("이어 학습 결과는 기존 run과 다른 --out 디렉터리에 저장하세요")
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            p.error(f"이어 학습 출력 디렉터리가 비어 있지 않습니다: {out}")
        for name in SNAPSHOT_GUARD:
            saved = source_dir / "src" / name
            current = pathlib.Path(__file__).with_name(name)
            if (not saved.is_file() or saved.read_bytes() != current.read_bytes()) and \
                    not args.allow_source_change:
                p.error(f"저장된 {name}와 현재 코드가 다릅니다. 변경된 환경/보상으로 "
                        "계속 학습하려면 --allow-source-change 를 명시하세요")
        if not args.stage1 and args.task != "cmdvel":
            print("[warn] 커리큘럼 상태는 체크포인트에 저장되지 않습니다. "
                  "--d0 로 재개 난이도를 지정해야 합니다.")
        print(f"[resume] model={resume_model}, vecnorm={resume_stats}")
    out.mkdir(parents=True, exist_ok=True)
    # 보상 프리셋 대신 실행 당시 공식을 보존한다. 가중치/게이트는 reward.py,
    # 진행량·종료 보너스 계산은 env.py 에 있으므로 둘 다 저장한다.
    snapshot = out / "src"
    snapshot.mkdir(exist_ok=True)
    for name in SNAPSHOT:
        f = pathlib.Path(__file__).with_name(name)
        if f.is_file():
            shutil.copy2(f, snapshot / name)
    man = write_manifest(out, args, resume_model, resume_stats)
    if not man["minibatch_even"]:
        print(f"[warn] 롤아웃 {man['rollout_steps']} 이 batch_size "
              f"{PPO_KW['batch_size']} 로 나눠지지 않습니다 -> 마지막 미니배치가 "
              f"작아집니다.  --envs 를 조정하면 비교가 깨끗해집니다.")
    print(f"[manifest] {out/'manifest.json'}  (1M 스텝당 업데이트 "
          f"{man['updates_per_1M']:.1f}회)")
    env_fns = [make_env(i, args) for i in range(args.envs)]
    raw_venv = (DummyVecEnv(env_fns) if args.envs == 1 else SubprocVecEnv(env_fns))
    if resume_stats is None:
        venv = VecNormalize(raw_venv, norm_obs=True, norm_reward=True, clip_obs=10.0,
                            gamma=args.gamma)
    else:
        venv = VecNormalize.load(str(resume_stats), raw_venv)
        venv.training = True
        venv.norm_reward = True
        if args.reset_reward_norm_on_resume:
            from stable_baselines3.common.running_mean_std import RunningMeanStd
            venv.ret_rms = RunningMeanStd(shape=())
            venv.returns[:] = 0.0
    # teacher 는 특권정보를 z 로 압축하는 인코더를 앞에 단다 (nets.TeacherExtractor).
    # student 는 이 z 를 고유수용감각 이력만으로 맞히도록 나중에 증류한다.
    policy_kwargs = {}
    if args.log_std_init is not None:
        if not np.isfinite(args.log_std_init):
            p.error("--log-std-init 는 유한한 값이어야 합니다")
        policy_kwargs["log_std_init"] = float(args.log_std_init)
    if args.teacher:
        from nets import TeacherExtractor, LATENT_DIM
        policy_kwargs.update(features_extractor_class=TeacherExtractor,
                             features_extractor_kwargs=dict(latent_dim=LATENT_DIM))

    # MLP 정책은 CPU 가 더 빠르다 (GPU 전송 오버헤드가 연산보다 큼). SB3 도 경고함.
    try:
        import tensorboard  # noqa: F401
        tb = str(out / "tb")
        print(f"[tb] tensorboard --logdir {out/'tb'}  ->  http://localhost:6006")
    except ImportError:
        tb = None
        print("[warn] tensorboard 없음 (pip install tensorboard) -> stdout 만")

    if resume_model is None:
        model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu",
                    gamma=args.gamma, **PPO_KW,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                       **policy_kwargs),
                    tensorboard_log=tb)
    else:
        model = PPO.load(str(resume_model), env=venv, device="cpu", verbose=1,
                         tensorboard_log=tb)
        if args.resume_lr is not None:
            from stable_baselines3.common.utils import get_schedule_fn
            model.learning_rate = args.resume_lr
            model.lr_schedule = get_schedule_fn(args.resume_lr)
        if args.resume_target_kl is not None:
            model.target_kl = args.resume_target_kl
        if args.reset_optimizer_on_resume:
            model.policy.optimizer = model.policy.optimizer_class(
                model.policy.parameters(), lr=model.lr_schedule(1.0),
                **model.policy.optimizer_kwargs)
        if args.resume_log_std is not None:
            with th.no_grad():
                model.policy.log_std.fill_(args.resume_log_std)
            print(f"[resume] exploration std={np.exp(args.resume_log_std):.4f}")
        if not np.isclose(model.gamma, args.gamma) or not np.isclose(venv.gamma, args.gamma):
            p.error("저장된 PPO/VecNormalize gamma 와 --gamma 가 다릅니다")
        print(f"[resume] {model.num_timesteps:,} 스텝부터 {args.steps:,} 스텝 추가 학습")
    model.learn(total_timesteps=args.steps, reset_num_timesteps=resume_model is None,
                callback=[
                    # Stage 1 은 난이도를 고정한다.
                    *([] if args.stage1 or args.task == "cmdvel" else [CurriculumCallback(d0=args.d0,
                                                                  verbose=1)]),
                    MetricsCallback(),
                    # 체크포인트의 관측 정규화 통계도 함께 저장한다.
                    CheckpointCallback(save_freq=max(args.checkpoint_every // args.envs, 1),
                                       save_path=str(out), name_prefix="ppo",
                                       save_vecnormalize=True),
                ])
    model.save(out / "final")
    venv.save(str(out / "vecnorm.pkl"))
    print("saved ->", out)


if __name__ == "__main__":
    main()

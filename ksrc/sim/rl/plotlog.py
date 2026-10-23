#!/usr/bin/env python3
"""SB3 stdout 로그를 그림으로. 텐서보드 없이 돌린 학습을 사후에 보기 위한 것.

  python3 plotlog.py <로그파일> [-o out.png]
"""
import argparse, re, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SB3 표는 "| rollout/ |  |" 섹션 헤더 아래에 "|    ep_rew_mean | 1.23 |" 으로
# 들여쓴다.  섹션을 따로 잡아서 앞에 붙여야 키가 안 겹친다.
SEC = re.compile(r"^\|\s*([a-z_]+/)\s*\|\s*\|")
ROW = re.compile(r"^\|\s+(\w+)\s+\|\s+([-\d.e+]+)\s*\|")
CUR = re.compile(r"\[curriculum\] 성공률 (\d+)% -> 난이도 ([\d.]+)")


def parse(path):
    recs = []; cur = {}; sec = ""
    curric = []
    for line in pathlib.Path(path).read_text(errors="ignore").splitlines():
        m = CUR.search(line)
        if m:
            curric.append((int(m.group(1)), float(m.group(2))))
            continue
        m = SEC.match(line)
        if m:
            sec = m.group(1); continue
        m = ROW.match(line)
        if m:
            try:
                cur[sec + m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        elif line.startswith("---") and cur:
            if any(k.endswith("total_timesteps") for k in cur):
                cur["total_timesteps"] = next(v for k, v in cur.items()
                                              if k.endswith("total_timesteps"))
                recs.append(cur)
            cur = {}; sec = ""
    if cur.get("total_timesteps"):
        recs.append(cur)
    return recs, curric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("-o", "--out", default="trainlog.png")
    a = p.parse_args()
    recs, curric = parse(a.log)
    if not recs:
        print("파싱된 행이 없다"); return
    x = np.array([r["total_timesteps"] for r in recs])
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    panels = [("rollout/ep_rew_mean", "에피소드 보상"),
              ("rollout/ep_len_mean", "에피소드 길이"),
              ("train/value_loss", "가치 손실"),
              ("train/entropy_loss", "엔트로피"),
              ("train/explained_variance", "설명분산"),
              ("train/approx_kl", "근사 KL")]
    have = [(k, t) for k, t in panels if any(k in r for r in recs)]
    n = len(have) + (1 if curric else 0)
    fig, ax = plt.subplots((n + 1) // 2, 2, figsize=(11, 2.6 * ((n + 1) // 2)))
    ax = np.atleast_1d(ax).ravel()
    for i, (k, t) in enumerate(have):
        xs = [r["total_timesteps"] for r in recs if k in r]
        ys = [r[k] for r in recs if k in r]
        ax[i].plot(xs, ys, lw=1.0)
        ax[i].set_title(t, fontsize=10); ax[i].grid(alpha=0.3)
    if curric:
        i = len(have)
        sr = [c[0] for c in curric]; dd = [c[1] for c in curric]
        ax[i].plot(dd, marker="o", ms=3, label="난이도")
        ax[i].plot(np.array(sr) / 100.0, marker=".", ms=3, alpha=0.6, label="성공률")
        ax[i].set_title("커리큘럼 (승급 시점마다)", fontsize=10)
        ax[i].legend(fontsize=8); ax[i].grid(alpha=0.3)
    for j in range(n, len(ax)):
        ax[j].axis("off")
    for a_ in ax[:n]:
        a_.set_xlabel("스텝" if a_ is not ax[len(have)] else "승급 횟수", fontsize=8)
    fig.suptitle(f"{a.log}   (최종 {int(x[-1]):,} 스텝)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"저장: {a.out}   ({len(recs)} 행, 커리큘럼 승급 {len(curric)}회)")


if __name__ == "__main__":
    main()

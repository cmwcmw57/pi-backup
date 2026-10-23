# ================================================================
# trace_io.py
# 설명: 실기 Pi run_policy npz(wheel_mps, steer_meas, host_ns, acc_g, slip)와 Isaac/run_real trace.npz(trace·columns·labels)를 같은 RunTrace로 읽는다.
# 사용법:
#   from experiments.yeouido_analysis.trace_io import load_run_trace
#   run = load_run_trace(Path("ik_1.npz"))   # npz 키로 형식을 자동 판별
# ================================================================
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WHEEL_RADIUS_M = 0.0475
PITCH_SMOOTH_S = 0.5
MODULE_ORDER = "FL, FR, RL, RR"


@dataclass(frozen=True)
class RunTrace:
    """시각(s), 바퀴 속도(rad/s)·조향각(rad) 4개와 선택 pitch·위치·구간 라벨·펌웨어 slip을 보관한다."""

    t_s: np.ndarray
    wheel_rad_s: np.ndarray
    steer_rad: np.ndarray
    pitch_rad: np.ndarray | None = None
    xyz_m: np.ndarray | None = None
    labels: np.ndarray | None = None
    log_slip: np.ndarray | None = None


def load_sim_trace(path: Path) -> RunTrace:
    """trace·columns(·labels) npz에서 t, wheel_vel_0-3, steer_0-3(+선택 x,y,z, pitch)를 읽는다."""
    data = np.load(path, allow_pickle=False)
    names = [str(name) for name in data["columns"]]
    table = data["trace"]
    column = lambda name: table[:, names.index(name)]  # noqa: E731
    missing = [n for n in ["t"] + [f"wheel_vel_{i}" for i in range(4)] + [f"steer_{i}" for i in range(4)]
               if n not in names]
    if missing:
        raise ValueError(f"trace npz missing columns: {missing}")
    has_xyz = all(n in names for n in ("x", "y", "z"))
    return RunTrace(
        t_s=column("t"),
        wheel_rad_s=np.stack([column(f"wheel_vel_{i}") for i in range(4)], axis=1),
        steer_rad=np.stack([column(f"steer_{i}") for i in range(4)], axis=1),
        pitch_rad=column("pitch") if "pitch" in names else None,
        xyz_m=np.stack([column("x"), column("y"), column("z")], axis=1) if has_xyz else None,
        labels=data["labels"].astype(str) if "labels" in data.files else None,
    )


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """행 방향 이동 평균(가장자리는 가용 표본만)을 계산한다."""
    kernel = np.ones(window)
    return np.stack([np.convolve(v, kernel, "same") / np.convolve(np.ones(len(v)), kernel, "same")
                     for v in values.T], axis=1)


def load_pi_npz(path: Path) -> RunTrace:
    """Pi run_policy 로그 npz를 읽는다: wheel_mps/r→rad/s, steer_meas, host_ns→t_s, acc_g→pitch, slip."""
    data = np.load(path, allow_pickle=True)
    t_s = (data["host_ns"] - data["host_ns"][0]) * 1e-9
    window = max(1, round(PITCH_SMOOTH_S / float(np.median(np.diff(t_s)))))
    acc = _moving_average(np.asarray(data["acc_g"], dtype=float), window)
    # 기체 x가 위로 들리면 ax>0. 시뮬 pitch 부호(코 들림 = 음수)에 맞춘다. 동적 가속이 섞인 값이다.
    pitch = -np.arctan2(acc[:, 0], np.hypot(acc[:, 1], acc[:, 2]))
    return RunTrace(
        t_s=t_s,
        wheel_rad_s=np.asarray(data["wheel_mps"], dtype=float) / WHEEL_RADIUS_M,
        steer_rad=np.asarray(data["steer_meas"], dtype=float),
        pitch_rad=pitch,
        log_slip=np.asarray(data["slip"], dtype=float) if "slip" in data.files else None,
    )


def load_run_trace(path: Path) -> RunTrace:
    """npz 키(trace/columns 또는 wheel_mps/host_ns)로 형식을 판별해 RunTrace를 반환한다."""
    keys = set(np.load(path, allow_pickle=True).files)
    if {"trace", "columns"} <= keys:
        run = load_sim_trace(path)
    elif {"wheel_mps", "steer_meas", "host_ns"} <= keys:
        run = load_pi_npz(path)
    else:
        raise ValueError(f"unknown trace npz format (keys: {sorted(keys)})")
    if len(run.t_s) < 2 or np.any(np.diff(run.t_s) <= 0):
        raise ValueError("trace time must be strictly increasing")
    return run


def with_label_table(run: RunTrace, path: Path) -> RunTrace:
    """label,t_start_s,t_end_s CSV로 구간 라벨을 붙인 복사본을 만든다(라벨 없는 실기 npz용)."""
    labels = np.full(len(run.t_s), "(none)", dtype=object)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if {"label", "t_start_s", "t_end_s"} - set(reader.fieldnames or []):
            raise ValueError("label CSV needs columns: label, t_start_s, t_end_s")
        for row in reader:
            inside = (run.t_s >= float(row["t_start_s"])) & (run.t_s < float(row["t_end_s"]))
            labels[inside] = row["label"]
    return RunTrace(run.t_s, run.wheel_rad_s, run.steer_rad, run.pitch_rad, run.xyz_m,
                    labels.astype(str), run.log_slip)

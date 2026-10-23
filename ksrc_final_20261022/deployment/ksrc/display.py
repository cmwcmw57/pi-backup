# ================================================================
# display.py
# 설명: 제어 명령 없이 시험 상태와 저장된 부하·정책 로그를 한국어로 요약한다.
# 사용법: ./ksrc --state-dir state/S1_01 view --watch; ./ksrc --state-dir state/S1_01 report
# ================================================================
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys
import time


def read_json(path: Path) -> dict:
    """생성 전이거나 기록 중인 파일은 빈 상태로 처리한다."""
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def tail_text(path: Path, limit: int = 65536) -> str:
    """큰 원시 로그를 화면 갱신마다 전부 읽지 않는다."""
    try:
        with path.open('rb') as source:
            source.seek(0, 2)
            source.seek(max(0, source.tell() - limit))
            return source.read().decode(errors='replace')
    except OSError:
        return ''


def view_text(state: Path) -> str:
    """실행 상태·준비 조건·로그 저장 상태를 행렬 없이 보여준다."""
    status = read_json(state / 'status.json')
    stopped = (state / 'stopped.json').exists()
    mode = 'FAKE · 모터 미사용' if (state / 'fake.json').exists() else '모드 미확인'
    owned = read_json(state / 'processes.json')
    command = owned.get('runner', {}).get('command', [])
    if '--config' in command:
        config = command[command.index('--config') + 1]
        mode = 'FAKE · 모터 미사용' if Path(config).name == 'fake.json' else 'REAL · 실기 스택'
    if not status and not stopped and 'BlockingIOError' in tail_text(state / 'supervisor.log'):
        return '\n'.join([f'[시작 실패] {state}', '상태: 시작 실패 — 다른 실기 스택이 전역 lock을 쥐고 있어 이 폴더는 시작되지 않았다.',
                          "살아 있는 세션 찾기: ps -eo pid,args | grep _supervise | grep -v grep  (--state-dir 값이 그 폴더)",
                          '그 세션을 estop·stop으로 끈 뒤 다시 시작한다. 이 폴더의 모드 정보는 없다.'])
    lines = [f'[{mode}] {state}']
    if not status:
        lines += ['상태: 종료됨' if stopped else '상태: 초기화 중 또는 상태 파일 없음',
                  '2초 뒤 다시 확인. 계속 같으면 supervisor.log와 runner.log 확인.']
        return '\n'.join(lines)
    age = max(0., time.time() - (state / 'status.json').stat().st_mtime)
    stale = age > 2.
    active = status.get('active', False)
    recovering = status.get('recovering', False)
    label = ('종료됨 · 아래는 마지막 기록' if stopped else
             '오래된 상태 · 현재 실행 여부 확인 필요' if stale else
             '복구 중' if recovering else 'AUTO' if active else
             '임무 종료' if status.get('done') else '대기 / TELEOP')
    lines.append(f'상태: {label} | run={status.get("run", 0)} | 단계={status.get("stage", "?")} | 기록 {age:.1f}초 전')
    sensor = '정상' if status.get('orb_ok') else '대기/불량'
    def age_ms(key: str) -> str:
        value = status.get(key)
        return '--' if value is None else f'{value * 1000:.0f} ms'
    lines.append(f'센서: {sensor} | pose age={age_ms("pose_age_s")} | IMU age={age_ms("imu_age_s")} | 복구 {status.get("recovery_count", 0)}회')
    velocity = status.get('command', [0., 0., 0.])
    lines.append(f'명령: 전진 {velocity[0]:+.3f} m/s | 횡이동 {velocity[1]:+.3f} m/s | 회전 {velocity[2]:+.3f} rad/s')
    stationary = status.get('stationary_s', 0.)
    aligned = status.get('in_region1', False) and status.get('start_aligned', False)
    if active:
        lines.append('출발 조건은 재평가하지 않음: AUTO 중 출발점 밖이어도 정상일 수 있음.')
    else:
        ready = not (stopped or stale or recovering or status.get('done')) and status.get('orb_ok') and aligned and stationary >= 3.
        lines.append(f'시작 조건: 센서={sensor} / 정지={stationary:.1f}초(필요 3초) / 출발좌표={"정렬" if aligned else "미정렬"} → {"START 가능" if ready else "START 대기"}')
    lines.append(f'이유: {status.get("reason", "?")} | 실제 맵 일치 판정은 아님. 실기 배치는 사람이 확인.')
    load = state / 'load'
    load_label = '완료' if (load / 'finished.txt').exists() else '수집 중/중단 여부 확인' if (load / 'meta.txt').exists() else '미시작'
    lines.append(f'로그: 부하={load_label} | 정책 NPZ={"저장됨" if (state / "policy.npz").exists() else "아직 없음(정상 stop 후 확인)"}')
    events = []
    for line in tail_text(state / 'navigation.jsonl').splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') != 'tick':
            events.append(f'{event.get("type", "?")} (t={event.get("stamp_s", "?")})')
    if events:
        lines.append('최근 이벤트: ' + ' / '.join(events[-3:]))
    lines.append('읽기 전용 화면. Ctrl+C는 보기만 종료하며 로버를 정지시키지 않음.')
    return '\n'.join(lines)


def watch(state: Path) -> None:
    """JSON 전체를 흘려보내지 않고 같은 화면을 1초마다 갱신한다."""
    try:
        while True:
            if sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(view_text(state), flush=True)
            time.sleep(1.)
    except KeyboardInterrupt:
        pass


def report_text(state: Path) -> str:
    """정책 주기·코어별 CPU·온도를 요약하며 stop 이후 부하 표본을 제외한다."""
    import numpy as np
    lines = [f'시험 결과: {state}',
             f'실행 스택: {"종료 확인" if (state / "stopped.json").exists() else "종료 미확인"}']
    load = state / 'load'
    finished = (load / 'finished.txt').exists()
    lines.append('부하 기록: ' + ('수집 완료' if finished else '수집 중 또는 중단됨 · load_collector.log 확인'))
    cutoff = read_json(state / 'stopped.json').get('time', float('inf'))
    meta = tail_text(load / 'meta.txt').splitlines()
    base = None
    if meta:
        try:
            base = datetime.fromisoformat(meta[0])
        except ValueError:
            pass
    if base:
        samples = defaultdict(list)
        text = (load / 'top_threads.log').read_text(errors='replace') if (load / 'top_threads.log').exists() else ''
        previous = base.replace(hour=0, minute=0, second=0, microsecond=0)
        for index, frame in enumerate(re.split(r'(?m)^top - ', text)[1:]):
            clock = re.match(r'(\d\d):(\d\d):(\d\d)', frame)
            if not clock:
                continue
            stamp = previous.replace(hour=int(clock[1]), minute=int(clock[2]), second=int(clock[3]))
            if stamp < previous:
                stamp += timedelta(days=1)
            previous = stamp
            if index == 0 or stamp.timestamp() > cutoff:
                continue
            for core, values in re.findall(r'%Cpu(\d+)\s*:\s*(.*?)(?=%Cpu|\n)', frame):
                idle = re.search(r'([\d.]+)\s+id\b', values)
                if idle:
                    samples[int(core)].append(100. - float(idle[1]))
        lines.append('CPU: 첫 프레임 제외 / 종료 표시 이후 표본 제외 / 100%=코어 하나')
        for core, values in sorted(samples.items()):
            lines.append(f'CPU{core}: 평균 {np.mean(values):.1f}% / 최대 {max(values):.1f}% / {len(values)}개 표본')
        if not samples:
            lines.append('CPU: 분석 가능한 표본 없음')
    temperatures, throttle, stamp = [], set(), None
    for line in tail_text(load / 'thermal.log', 1024 * 1024).splitlines():
        try:
            stamp = datetime.fromisoformat(line).timestamp()
            continue
        except ValueError:
            pass
        if stamp is None or stamp > cutoff:
            continue
        if line.startswith('temperature_mC='):
            temperatures.append(float(line.split('=')[1]) / 1000.)
        elif line.startswith('throttled='):
            throttle.add(line.split('=')[1])
    if temperatures:
        lines.append(f'온도: 시작 {temperatures[0]:.1f}°C / 최대 {max(temperatures):.1f}°C / 마지막 {temperatures[-1]:.1f}°C')
    if throttle:
        lines.append('throttle 표본: ' + ', '.join(sorted(throttle)))
    path = state / 'policy.npz'
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                for key, label in (('period_ms', '하위 제어 주기'), ('compute_ms', '하위 계산 시간')):
                    values = np.asarray(data[key], dtype=float).ravel()
                    values = values[np.isfinite(values)]
                    if not len(values):
                        lines.append(f'{label}: 표본 없음')
                        continue
                    lines.append(f'{label}: 중앙 {np.median(values):.2f} / p99 {np.quantile(values, .99):.2f} / 최대 {max(values):.2f} ms ({len(values)}개)')
                    if key == 'period_ms':
                        lines.append(f'주기 초과: 25 ms 초과 {int((values > 25).sum())}/{len(values)} | 40 ms 초과 {int((values > 40).sum())}')
        except (OSError, ValueError, KeyError) as error:
            lines.append(f'정책 NPZ 읽기 실패: {error}')
    else:
        lines.append('정책 NPZ 없음: 실행 중이면 정상 stop 후 확인. 정책 미설치/저장 실패는 policy.log 확인.')
    events = []
    path = state / 'navigation.jsonl'
    if path.exists():
        with path.open() as source:
            for line in source:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('type') in ('recovery_start', 'recovery_done', 'start_rejected'):
                    events.append(event)
    for kind in ('recovery_start', 'recovery_done', 'start_rejected'):
        lines.append(f'{kind}: {sum(event.get("type") == kind for event in events)}회')
    for event in events[-3:]:
        lines.append(f'  t={event.get("stamp_s", "?")}초 {event.get("type")} / 이유={event.get("reason", "-")} / 단계={event.get("stage", "-")}')
    lines.append('정지 시험은 이동 중 MPPI 부하·완주 검증이 아님. 부하 수집 시간과 실제 스택 실행 시간은 다를 수 있음.')
    return '\n'.join(lines)

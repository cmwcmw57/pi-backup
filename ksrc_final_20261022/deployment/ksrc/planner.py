# ================================================================
# planner.py
# 설명: 최종 MPPI 계산을 별도 프로세스로 실행하고 오래된 결과·다른 단계 결과를 버린다.
# 사용법: AsyncPlanner(runtime, '0'); adapter.apply(now_s)
# ================================================================
from __future__ import annotations
import multiprocessing as mp
import os
import signal


def zero_plan(reason: str) -> dict:
    """계산 대기 중 임무를 실패 처리하지 않고 정지시킨다."""
    return dict(valid=True, reason=reason, command=dict(vx_m_s=0., vy_m_s=0., yaw_rate_rad_s=0.),
                selected_trajectory=[], candidate_trajectories=[])


def work(connection, planner, cpus: str) -> None:
    """요청 하나씩 처리하며 기존 MPPI 난수 상태를 유지한다."""
    # fork가 실행기의 SIGTERM handler를 물려받아 worker 종료를 무시하지 않게 한다.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.sched_setaffinity(0, {int(v) for v in cpus.split(',')})
    try:
        while True:
            request = connection.recv()
            if request is None:
                break
            key, stamp, args, kwargs = request
            connection.send((key, stamp, planner.plan(*args, **kwargs)))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class AsyncPlanner:
    """제어 tick은 기다리지 않고 현재 단계의 최신 MPPI 결과만 수용한다."""

    def __init__(self, runtime, cpus: str) -> None:
        self.runtime = runtime
        context = mp.get_context('fork')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=work, args=(child, runtime.planner, cpus), daemon=True)
        self.process.start()
        child.close()
        self.stamp, self.accepted_at, self.generation = 0., -1e9, 0
        self.inflight = False
        self.key = None
        self.result = zero_plan('plan_pending')
        runtime.planner.plan = self.plan

    def phase(self) -> tuple:
        """이전 단계·복구 전 요청과의 혼용을 막는다."""
        return self.runtime.stage, self.runtime.state, self.runtime.route, self.generation

    def invalidate(self) -> None:
        """이전 결과가 나중에 도착해도 다시 사용하지 않는다."""
        self.generation += 1
        self.key = None
        self.result = zero_plan('plan_pending')
        self.accepted_at = -1e9

    def apply(self, now: float) -> None:
        """완료 결과를 비차단 수집하고 300 ms 넘게 묵으면 0으로 바꾼다."""
        self.stamp = now
        if self.connection.poll():
            try:
                key, stamp, result = self.connection.recv()
            except (EOFError, OSError):
                self.inflight = False
            else:
                self.inflight = False
                if key == self.phase() and now - stamp <= .3:
                    self.key, self.result, self.accepted_at = key, result, stamp
                    self.runtime.last_plan = result
        if self.key != self.phase() or now - self.accepted_at > .3:
            self.result = zero_plan('plan_stale' if self.key else 'plan_pending')
            if self.runtime.last_plan:
                self.runtime.last_plan = self.result

    def plan(self, *args, **kwargs) -> dict:
        """계산 요청을 보낸 뒤 직전 유효 결과 또는 정지 결과를 반환한다."""
        self.apply(self.stamp)
        if not self.inflight and self.process.is_alive():
            self.connection.send((self.phase(), self.stamp, args, kwargs))
            self.inflight = True
        return self.result

    def close(self) -> None:
        """계산 중에도 제어 종료를 막지 않는다."""
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=.5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        self.connection.close()

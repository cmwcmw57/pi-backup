# ================================================================
# estimator_log.py
# 설명: 추정기에 실제 들어간 IMU·영상·pose 조회 이벤트를 파일 한 개에 순서대로 기록한다. 기록 실패는 시뮬을 멈추지 않는다.
# 사용법: log=EstimatorLog(output_dir); log.write({'kind':'imu',...}); log.close()
# ================================================================
from __future__ import annotations

import json
from pathlib import Path


class EstimatorLog:
    """estimator_events.jsonl을 한 번만 열고 줄 단위로 쓴다. 직렬화·디스크 오류는 기록만 끈다."""

    def __init__(self, output: Path) -> None:
        self.stream = (Path(output)/'estimator_events.jsonl').open('a')
        self.failed = False

    def write(self, event: dict) -> None:
        if self.failed:
            return
        try:
            self.stream.write(json.dumps(event, allow_nan=False)+'\n')
        except (ValueError, TypeError, OSError) as error:
            self.failed = True
            print(f'SITE_SIM_WARNING estimator log disabled: {type(error).__name__}: {error}', flush=True)

    def close(self) -> None:
        self.stream.close()

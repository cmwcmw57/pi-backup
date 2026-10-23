# ================================================================
# async_region_observation.py
# 설명: 픽셀/depth 관측을 별도 CPU 프로세스에서 계산하고 최신 입력·결과 한 개만 전달한다.
# 사용법: worker=AsyncRegionObservation(site, calibration, regions, mode='pixel'); worker.submit(left, right, pose, stamp)
# ================================================================
from __future__ import annotations

import multiprocessing as mp
import math
from pathlib import Path
from queue import Empty, Full
import time

import cv2
import numpy as np


def _read_gray(value: str | bytes):
    """임시 파일이 덮어써져도 안전한 영상 바이트 또는 기존 재생 경로를 읽는다."""
    return (cv2.imdecode(np.frombuffer(value,np.uint8),cv2.IMREAD_GRAYSCALE)
            if isinstance(value,bytes) else cv2.imread(value,cv2.IMREAD_GRAYSCALE))


def next_observation_stamp(stamp_s: float, hz: float) -> float:
    """영상 tick 양자화로 요청 주기가 누적해서 느려지지 않게 고정 위상을 유지한다."""
    return (math.floor(stamp_s*hz+1e-8)+1)/hz


def latest_observation(previous: dict | None, incoming: dict) -> dict:
    """원래 관측 시각으로 역순 결과를 거르고 새 관측의 시각을 덮어쓰지 않는다."""
    return incoming if previous is None or incoming['stamp_s'] >= previous['stamp_s'] else previous


def _replace(q, value) -> None:
    """producer가 소비자를 기다리지 않도록 최신 메시지 한 개만 남긴다."""
    try:
        q.put_nowait(value)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(value)
        except Full:
            pass  # feeder가 이전 메시지를 전달 중이면 이번 입력을 버린다.


def _calculate(site, calibration: dict, regions: list[dict], mode: str, inputs, outputs) -> None:
    """하위 프로세스에서 영상 경로/바이트를 디코딩하고 관측을 계산한다."""
    cv2.setNumThreads(1)
    if mode == 'pixel':
        from experiments.handheld_mapping.pixel_region_observation import PixelRegionObservation
        observer = PixelRegionObservation(site, calibration, regions)
    else:
        from experiments.handheld_mapping.region_observation import RegionObservation
        observer = RegionObservation(site, calibration, regions)
    while True:
        request = inputs.get()
        if request is None:
            break
        left, right, pose, stamp = request
        begin = time.monotonic()
        try:
            image = _read_gray(left)
            other = _read_gray(right) if mode == 'depth' else None
            result = observer.update(image, other, pose, stamp)
        except Exception as error:
            result = dict(stamp_s=stamp, valid=False, reason='observation_error', error=repr(error),
                          visible_regions=[], obstacles_xy=[])
        result.update(processing_ms=(time.monotonic()-begin)*1000, observation_mode=mode)
        _replace(outputs, result)


class AsyncRegionObservation:
    """비차단 submit/poll과 명시적인 하위 프로세스 종료를 제공한다."""

    def __init__(self, site, calibration: dict, regions: list[dict], *, mode: str) -> None:
        if mode not in ('pixel', 'depth'):
            raise ValueError('mode must be pixel or depth')
        context = mp.get_context('spawn')
        self.inputs, self.outputs = context.Queue(maxsize=1), context.Queue(maxsize=1)
        self.process = context.Process(target=_calculate,
                                       args=(site, calibration, regions, mode, self.inputs, self.outputs), daemon=True)
        self.process.start()
        self.latest = None

    def submit(self, left: Path | bytes, right: Path | bytes, pose, stamp_s: float) -> None:
        """영상 경로/불변 바이트와 영상 시각의 추정 pose를 전달하고 즉시 반환한다."""
        _replace(self.inputs, (left if isinstance(left,bytes) else str(left),
                               right if isinstance(right,bytes) else str(right),pose.copy(),float(stamp_s)))

    def poll(self) -> dict | None:
        """새 결과가 있을 때만 반환한다. 마지막 관측의 시각은 유지한다."""
        incoming = None
        while True:
            try:
                value = self.outputs.get_nowait()
            except Empty:
                break
            newer = latest_observation(self.latest, value)
            if newer is not self.latest:
                self.latest = incoming = newer
        if not self.process.is_alive():
            raise RuntimeError('region observation worker exited')
        return incoming

    def close(self) -> None:
        """이 인스턴스의 하위 프로세스만 종료한다."""
        _replace(self.inputs, None)
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3)
        self.inputs.cancel_join_thread()
        self.outputs.cancel_join_thread()
        self.inputs.close()
        self.outputs.close()

# ================================================================
# recording.py
# 설명: 노트북 임시 디스크 인덱스로 keyframe을 정렬하고 depth를 순차 매칭한다.
# 사용법: from experiments.handheld_mapping.recording import matched_depths
# ================================================================
from contextlib import contextmanager
import csv
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterator


@contextmanager
def disk_index(parent: Path) -> Iterator[sqlite3.Connection]:
    """SQLite page cache를 2MiB로 제한한 소유 임시 디스크 인덱스를 만든다."""
    with tempfile.TemporaryDirectory(prefix='.rover_index_', dir=parent) as directory:
        database = sqlite3.connect(str(Path(directory)/'index.sqlite'))
        database.execute('PRAGMA cache_size=-2048')
        database.execute('PRAGMA temp_store=FILE')
        try:
            yield database
        finally:
            database.close()


def ordered_keyframes(path: Path, map_id: int) -> Iterator[tuple[int, dict]]:
    """native pointer 순서 CSV를 원본 수정 없이 timestamp 순서로 전달한다."""
    with disk_index(path.parent) as database:
        database.execute('CREATE TABLE frames(stamp INTEGER PRIMARY KEY, row TEXT NOT NULL)')
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if str(row['map_id']) != str(map_id):
                    raise ValueError('keyframe map identity mismatch')
                stamp = int((Decimal(row['source_stamp'])*1_000_000_000).to_integral_value())
                if not 0 <= stamp < 2**63:
                    raise ValueError('keyframe stamp outside signed64 range')
                try:
                    database.execute('INSERT INTO frames VALUES (?,?)', (stamp, json.dumps(row)))
                except sqlite3.IntegrityError as error:
                    raise ValueError('duplicate keyframe stamp') from error
        database.commit()
        for stamp, row in database.execute('SELECT stamp,row FROM frames ORDER BY stamp'):
            yield stamp, json.loads(row)


def depth_frames(run: Path, frame_id: str) -> Iterator[dict]:
    """전체 촬영 인덱스를 읽지 않고 단조 증가 depth 레코드만 전달한다."""
    previous = -1
    with (run/'frames.jsonl').open() as stream:
        for line in stream:
            frame = json.loads(line)
            if frame['stream'] != 'depth':
                continue
            stamp = frame['stamp_ns']
            if not isinstance(stamp, int) or stamp <= previous or frame['frame_id'] != frame_id:
                raise ValueError('invalid depth stamp/frame identity')
            previous = stamp
            yield frame


def matched_depths(run: Path, keyfile: Path, map_id: int, frame_id: str,
                   tolerance_ns: int = 2_000_000) -> Iterator[tuple[dict, dict | None, int | None]]:
    """현재/직전 depth 레코드만 유지하며 정렬된 keyframe과 일대일 매칭한다."""
    frames = depth_frames(run, frame_id)
    current, previous = next(frames, None), None
    last_used = -1
    for stamp, row in ordered_keyframes(keyfile, map_id):
        while current is not None and current['stamp_ns'] < stamp:
            previous, current = current, next(frames, None)
        candidates = [frame for frame in (previous, current) if frame is not None]
        matched = min(candidates, key=lambda frame: abs(frame['stamp_ns']-stamp)) if candidates else None
        gap = abs(matched['stamp_ns']-stamp) if matched else None
        if matched is None or gap > tolerance_ns or matched['stamp_ns'] == last_used:
            yield row, None, None
        else:
            last_used = matched['stamp_ns']
            yield row, matched, gap

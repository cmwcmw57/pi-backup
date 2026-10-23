# ================================================================
# storage.py
# 설명: 전달받은 IR·depth 원본과 metadata를 새 노트북 폴더에만 저장한다.
# 사용법: from ksrc_runtime.handheld_mapping.storage import RunWriter
# ================================================================
import hashlib
import json
from pathlib import Path
import re
import time
import zlib

import cv2
import numpy as np


def atomic_json(path: Path, data: dict, *, indent: int | None=2) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=indent, allow_nan=False) + "\n")
    temporary.replace(path)


def decode_image(header: dict, payload: bytes) -> np.ndarray:
    width, height, step = (int(header[key]) for key in ("width", "height", "step"))
    encoding = header["encoding"]
    if encoding not in ("mono8", "16UC1", "rgb8", "bgr8"):
        raise ValueError("unsupported image encoding")
    pixel = {"mono8": 1, "16UC1": 2, "rgb8": 3, "bgr8": 3}[encoding]
    if not (0 < width <= 4096 and 0 < height <= 2160 and width * pixel <= step <= width * pixel + 4096):
        raise ValueError("invalid image dimensions/stride")
    size = height * step
    if size > 32 * 1024 * 1024 or header.get("compression") != "zlib":
        raise ValueError("invalid image size/compression")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, size + 1)
    if len(raw) != size or not decoder.eof or decoder.unused_data:
        raise ValueError("invalid image payload length")
    dtype = np.dtype(">u2" if header.get("is_bigendian") else "<u2") if pixel == 2 else np.dtype("u1")
    if pixel == 3:
        image = np.ndarray((height, width, 3), dtype=dtype, buffer=raw, strides=(step, 3, 1))
        return image[:, :, ::-1].copy() if encoding == "rgb8" else image.copy()
    return np.ndarray((height, width), dtype=dtype, buffer=raw, strides=(step, pixel)).astype(
        np.uint16 if pixel == 2 else np.uint8, copy=True)


class RunWriter:
    """노트북 원본과 최종 ORB 산출물을 기록하고 경로·checksum을 검증한다."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "orb").mkdir()
        self.index = (self.path / "frames.jsonl").open("w", buffering=1)
        self.events = (self.path / "events.jsonl").open("w", buffering=1)
        self.counts = {key: 0 for key in ("left", "right", "depth")}
        self.last_stamp: dict[str, int] = {}
        self.image_timings = {side: {"images": 0, "decode_ms_total": 0., "decode_ms_max": 0.,
                                    "png_write_ms_total": 0., "png_write_ms_max": 0.}
                              for side in self.counts}

    def accept(self, header: dict, payload: bytes) -> np.ndarray | None:
        kind = header["kind"]
        if kind == "image":
            side, stamp = header["stream"], header["stamp_ns"]
            if side not in ("left", "right", "depth", "color") or not isinstance(stamp, int) or stamp <= self.last_stamp.get(side, -1):
                raise ValueError("invalid/non-increasing image stream or stamp")
            expected = {"left": ("mono8",), "right": ("mono8",), "depth": ("16UC1",),
                        "color": ("rgb8", "bgr8")}
            if header.get("encoding") not in expected[side]:
                raise ValueError("image encoding disagrees with stream")
            decode_start = time.monotonic_ns()
            image = decode_image(header, payload)
            decode_ms = (time.monotonic_ns()-decode_start)/1e6
            relative = f"raw/{side}/{stamp}.png"
            target = self.path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            png_start = time.monotonic_ns()
            if not cv2.imwrite(str(target), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError("원본 영상 저장 실패")
            saved_ns = time.monotonic_ns()
            png_ms = (saved_ns-png_start)/1e6
            self.index.write(json.dumps({**header, "path": relative, "laptop_decode_ms": decode_ms,
                                        "laptop_png_write_ms": png_ms,
                                        "laptop_image_saved_monotonic_ns": saved_ns}) + "\n")
            if side not in self.counts:
                self.counts[side] = 0
                self.image_timings[side] = {"images": 0, "decode_ms_total": 0., "decode_ms_max": 0.,
                                            "png_write_ms_total": 0., "png_write_ms_max": 0.}
            stats = self.image_timings[side]
            stats["images"] += 1
            for key, value in (("decode_ms", decode_ms), ("png_write_ms", png_ms)):
                stats[key+"_total"] += value
                stats[key+"_max"] = max(stats[key+"_max"], value)
            self.counts[side] += 1
            self.last_stamp[side] = stamp
            return image
        if kind == "calibration":
            target = self.path / "capture_calibration.json"
            if target.exists():
                raise ValueError("calibration changed during capture")
            atomic_json(target, header)
        elif kind == "artifact":
            name = header["name"]
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.(csv|json|yaml|osa|ply|tum|log)", name):
                raise ValueError("invalid artifact path")
            if hashlib.sha256(payload).hexdigest() != header.get("sha256"):
                raise ValueError("artifact checksum mismatch")
            target = self.path / "orb" / name
            if target.exists():
                raise ValueError("duplicate artifact")
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
        self.events.write(json.dumps(header, allow_nan=False) + "\n")
        return None

    def close(self) -> None:
        self.index.close()
        self.events.close()

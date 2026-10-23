# ================================================================
# protocol.py
# 설명: SSH를 통해 JSON metadata와 binary 원본을 길이 제한으로 전달한다.
# 사용법: from experiments.handheld_mapping.protocol import read_packet, write_packet
# ================================================================
import json
import struct
from typing import BinaryIO

MAX_HEADER = 64 * 1024
MAX_PAYLOAD = 128 * 1024 * 1024


def _read(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            raise EOFError("전송 도중 연결이 끊겼습니다")
        chunks.extend(block)
    return bytes(chunks)


def write_packet(stream: BinaryIO, header: dict, payload: bytes = b"") -> None:
    data = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    if not 0 < len(data) <= MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ValueError("packet size exceeds limit")
    stream.write(struct.pack(">II", len(data), len(payload)))
    stream.write(data)
    stream.write(payload)
    stream.flush()


def read_packet(stream: BinaryIO) -> tuple[dict, bytes] | None:
    first = stream.read(1)
    if not first:
        return None
    header_size, payload_size = struct.unpack(">II", first + _read(stream, 7))
    if not 0 < header_size <= MAX_HEADER or payload_size > MAX_PAYLOAD:
        raise ValueError("invalid packet size / SSH stdout contamination")
    header = json.loads(_read(stream, header_size))
    if not isinstance(header, dict) or not isinstance(header.get("kind"), str):
        raise ValueError("invalid packet header")
    return header, _read(stream, payload_size)

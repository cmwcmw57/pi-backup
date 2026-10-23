# ================================================================
# transport.py
# 설명: 패킷 개수와 압축 전 byte 수를 동시에 제한하는 센서 전송 큐다.
# 사용법: from experiments.handheld_mapping.transport import ByteQueue
# ================================================================
import json
import queue
import threading
import time
from collections import deque


class ByteQueue:
    """전송 대기의 payload·header byte를 제한하고 포화 누락을 집계한다."""

    def __init__(self, max_items: int = 24, max_bytes: int = 4*1024*1024) -> None:
        if not isinstance(max_items, int) or not isinstance(max_bytes, int) or min(max_items, max_bytes) <= 0:
            raise ValueError('invalid queue limits')
        self.max_bytes = max_bytes
        self.queue: queue.Queue = queue.Queue(maxsize=max_items)
        self.lock = threading.Lock()
        self.queued_bytes = self.peak_bytes = self.dropped = 0

    def put_nowait(self, item: tuple[dict, bytes]) -> None:
        self.put_many_nowait([item])

    def put_many_nowait(self, items: list[tuple[dict, bytes]]) -> None:
        """한 시각의 영상 묶음을 전부 넣거나 전부 거절한다."""
        sized = [(item, len(item[1])+len(json.dumps(item[0], allow_nan=False,
                  separators=(',', ':')).encode())+8) for item in items]
        total = sum(size for _, size in sized)
        with self.lock:
            if total+self.queued_bytes > self.max_bytes or self.queue.qsize()+len(items) > self.queue.maxsize:
                self.dropped += len(items)
                raise queue.Full
            for item in sized:
                self.queue.put_nowait(item)
            self.queued_bytes += total
            self.peak_bytes = max(self.peak_bytes, self.queued_bytes)

    def get(self) -> tuple[dict, bytes] | None:
        item, size = self.queue.get()
        with self.lock:
            self.queued_bytes -= size
        return item

    def finish(self, timeout: float = 10.) -> None:
        self.queue.put((None, 0), timeout=timeout)

    def snapshot(self) -> dict:
        with self.lock:
            return {'queued_bytes': self.queued_bytes, 'peak_bytes': self.peak_bytes,
                    'max_bytes': self.max_bytes, 'dropped': self.dropped,
                    'scope': 'queued raw payload/header bytes; excludes ROS/current frame and Python object overhead'}


class FrameBundler:
    """stereo exact + depth 2ms를 같은 묶음으로 유지하는 유한 콜백 버퍼."""

    def __init__(self, emit, discard, max_pending=3, tolerance_ns=2_000_000, max_wait_s=.25):
        self.emit, self.discard = emit, discard
        self.max_pending, self.tolerance_ns, self.max_wait_ns = max_pending, tolerance_ns, int(max_wait_s*1e9)
        if max_pending < 1 or tolerance_ns < 0 or max_wait_s <= 0:
            raise ValueError('invalid bundle limits')
        self.pending = {side: deque() for side in ('left', 'right', 'depth')}
        self.groups = self.orphans = self.peak_bytes = 0

    def _drop(self, side):
        item = self.pending[side].popleft()
        self.discard(item)
        self.orphans += 1

    def push(self, item):
        header, _ = item
        self.pending[header['stream']].append(item)
        now = header['pi_receipt_monotonic_ns']
        for side, entries in self.pending.items():
            while entries and (len(entries) > self.max_pending or
                    now-entries[0][0]['pi_receipt_monotonic_ns'] > self.max_wait_ns):
                self._drop(side)
        left, right, depth = (self.pending[side] for side in ('left', 'right', 'depth'))
        while left and right and depth:
            ls, rs, ds = (q[0][0]['stamp_ns'] for q in (left, right, depth))
            if ls != rs:
                self._drop('left' if ls < rs else 'right')
            elif ds < ls-self.tolerance_ns:
                self._drop('depth')
            elif ds > ls+self.tolerance_ns:
                self._drop('left')
                self._drop('right')
            else:
                self.emit([left.popleft(), right.popleft(), depth.popleft()])
                self.groups += 1
        self.peak_bytes = max(self.peak_bytes, sum(len(payload) for q in self.pending.values() for _, payload in q))

    def finish(self):
        for side in self.pending:
            while self.pending[side]:
                self._drop(side)

    def snapshot(self):
        return {'complete_groups': self.groups, 'orphan_images': self.orphans,
                'pending_images': sum(len(q) for q in self.pending.values()),
                'peak_raw_bytes': self.peak_bytes, 'max_pending_per_stream': self.max_pending,
                'depth_tolerance_ns': self.tolerance_ns, 'max_wait_ms': self.max_wait_ns/1e6,
                'scope': 'partial capture groups only; excludes sender queue/current callbacks'}


class PacketReader:
    """네트워크 수신과 PNG/SLAM 처리를 분리하며 포화 시 누락 없이 역압을 건다."""

    def __init__(self, stream, max_bytes=32*1024*1024, max_items=128):
        from experiments.handheld_mapping.protocol import read_packet
        if max_bytes <= 0 or max_items <= 0:
            raise ValueError('invalid reader limits')
        self.read_packet, self.stream = read_packet, stream
        self.max_bytes, self.max_items = max_bytes, max_items
        self.condition = threading.Condition()
        self.pending = deque()
        self.bytes = self.peak = self.packets = self.oversize = 0
        self.producer_wait_ms = self.max_queue_wait_ms = 0.
        self.done = self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._read, name='mapping-packet-reader', daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while True:
                start = time.monotonic_ns()
                packet = self.read_packet(self.stream)
                receipt = time.monotonic_ns()
                if packet is None:
                    break
                header, payload = packet
                header = {**header, 'laptop_receipt_monotonic_ns': receipt,
                          'laptop_read_wait_ms': (receipt-start)/1e6}
                size = len(payload)+len(json.dumps(header).encode())+8
                with self.condition:
                    wait_start = time.monotonic_ns()
                    # One protocol-limited large artifact can pass when the buffer is empty.
                    while not self.closed and self.pending and (self.bytes+size > self.max_bytes
                                                               or len(self.pending) >= self.max_items):
                        self.condition.wait()
                    self.producer_wait_ms += (time.monotonic_ns()-wait_start)/1e6
                    if self.closed:
                        return
                    self.pending.append((header, payload, size))
                    self.bytes += size
                    self.peak = max(self.peak, self.bytes)
                    self.oversize += size > self.max_bytes
                    self.packets += 1
                    self.condition.notify_all()
        except Exception as error:
            self.error = error
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def get(self):
        with self.condition:
            while not self.pending and not self.done and not self.closed:
                self.condition.wait()
            if self.pending:
                header, payload, size = self.pending.popleft()
                self.bytes -= size
                delay = (time.monotonic_ns()-header['laptop_receipt_monotonic_ns'])/1e6
                header['laptop_receive_queue_wait_ms'] = delay
                self.max_queue_wait_ms = max(self.max_queue_wait_ms, delay)
                self.condition.notify_all()
                return header, payload
            if self.error is not None:
                raise self.error
            return None

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {'queued_bytes': self.bytes, 'peak_bytes': self.peak, 'max_bytes': self.max_bytes,
                    'max_items': self.max_items, 'packets': self.packets, 'oversize_artifacts': self.oversize,
                    'producer_wait_ms': self.producer_wait_ms, 'max_queue_wait_ms': self.max_queue_wait_ms,
                    'drops': 0, 'scope': 'compressed receive buffer; excludes one packet being read, '
                    'writer and object overhead; one large protocol-limited artifact may exceed max_bytes'}

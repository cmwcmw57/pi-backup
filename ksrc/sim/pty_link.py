"""
pty 기반 범용 바이트 소스.

pty 를 열고, 텔레옵 클라이언트가 --port 로 가리킬 수 있는 슬레이브 경로를
노출하고, 들어온 바이트를 콜백으로 넘긴다. 모든 시뮬 백엔드
(virtual_nucleo.py 의 2D 추측항법 시뮬, mujoco/mujoco_sim.py 의 3D 물리 시뮬)가
공유해서, 까다로운 이 배관 코드가 **한 곳에만** 존재하도록 한다.

주의: 리눅스에서 슬레이브가 한 번도 열리지 않은 상태로 pty 마스터를 읽으면
블로킹이 아니라 OSError(EIO) 가 난다. 예전에 pty.openpty() 직후 우리 쪽
slave_fd 를 닫았더니, 리더 스레드가 "다음에 슬레이브를 여는 클라이언트"와
경합해서 클라이언트가 붙기도 전에 리더가 죽는 일이 있었다. 그러면 그때까지
보낸 바이트가 **에러 없이 전부 사라진다**. 해결: (a) 이 객체가 살아 있는 동안
slave_fd 를 계속 열어두고, (b) EIO 를 치명적 오류가 아니라 일시적/재시도
가능으로 처리한다 (클라이언트가 끊었다 다시 붙는 경우 대비).
"""
import errno
import os
import pty
import threading
import time


class PtyByteSource:
    def __init__(self, on_bytes):
        """pty 에 바이트가 들어올 때마다 리더 스레드에서
        on_bytes(data: bytes, now_ms: int) 를 호출한다."""
        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd
        self.slave_fd = slave_fd
        self.slave_name = os.ttyname(slave_fd)
        self.stop_flag = threading.Event()

        def reader():
            while not self.stop_flag.is_set():
                try:
                    data = os.read(self.master_fd, 4096)
                except OSError as e:
                    if e.errno == errno.EIO:
                        time.sleep(0.05)  # 지금은 슬레이브를 연 클라이언트가 없음
                        continue
                    break
                if not data:
                    break
                on_bytes(data, int(time.monotonic() * 1000))

        self.thread = threading.Thread(target=reader, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_flag.set()
        for fd in (self.master_fd, self.slave_fd):
            try:
                os.close(fd)
            except OSError:
                pass

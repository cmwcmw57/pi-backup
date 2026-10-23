# IMU 평균 표본 읽기 시각 패치

## 2026-10-16 최신 KSRC 통합

기존 `ksrc_imu_mean_time.patch`와 `BASELINE.json`은 배포했던 40499bc용으로 보존한다. 최신 KSRC `07161349eb1cfd765d3efff1da43a5cbac57e24f`에는 **별도** `ksrc_imu_mean_time_latest.patch`와 `LATEST_BASELINE.json`을 사용한다.

최신 upstream이 byte140~141을 `wpi_corr`로 사용하므로 이 위치를 예약 영역으로 취급하면 안 된다. latest patch는 bit12로 구분한다: bit12=0이면 기존 unsigned PI readback, bit12=1이면 signed IMU offset이다. 새 Python decoder는 timing frame에서 `wpi_corr=None`으로 표시하고 IMU_VALID까지 있어야 timing을 유효하게 취급한다. **두 값을 같은 frame에 동시에 제공하지 않는다.** 구형 KSRC decoder는 timing offset을 PI 설정값으로 오해하므로 latest timing firmware와 혼용하면 안 된다. PI 설정 명령 자체는 바뀌지 않지만 timing 모드에서는 PI readback 검증을 수행할 수 없다.

latest patch는 common뿐 아니라 실제 board의 SPI source/header 사본과 Python ctypes CTlm 두 정의도 함께 갱신한다. C 구조체 확장 후 기존 ctypes 정의를 사용하면 buffer 밖 접근이 생긴다. upstream wire test도 timing/PI flag 의미를 구분하도록 수정한다.

```bash
python experiments/imu_timing_patch/build_patch.py /path/to/ksrc-0716134 --profile latest
KSRC_LATEST_BASELINE=/path/to/ksrc-0716134 \
  .venv-mapping/bin/python -m pytest tests/imu_timing/test_ksrc_latest.py -q
```

generator는 baseline hash를 검사하고 원본을 바꾸지 않는다. `--output NEW_PATCH`로 별도 출력할 수 있다. `tools/mapping_imu_bridge.sh`는 Pi 원본 소스 대신 RAM 사본에 검토된 decoder를 사용하며 metadata 불일치 시 실행을 거부한다. 이번 통합은 실제 flash·SPI·모터 검증이 아니다.

아래는 **기존 40499bc 패치의 역사적 계약·적용 결과**다. 최신 patch에 구형 decoder 공존 주장을 적용하지 않는다.

현재 STM이 보내는 IMU 평균값에, 그 평균에 포함된 표본들의 대표 **MCU 읽기 완료 시각**을 붙이는 패치다. 2026-10-15에 ARM 전체 build, STM flash/verify 및 Pi decoder 적용 후 정지 상태 POLL-only 수신을 확인했다. ORB IMU fusion은 아직 수행하지 않았다. [실기 적용 기록](../../docs/experiments/2026-10-15-imu-timestamp-bench-deployment.md)

## 선택한 방식과 트레이드오프

기존 IMU 평균 처리를 유지하면서 각 성공한 읽기의 `HAL_GetTick()`을 기록하고, 같은 표본 집합의 읽기 시각 산술평균을 보낸다. 불규칙한 읽기 간격도 반영한다. 기존 센서 읽기와 144-byte SPI 패킷 길이는 유지한다.

| 항목 | 이점 | 한계·비용 |
|---|---|---|
| 평균값 + 대표 읽기 시각 | 제어 loop 시작 시각을 IMU 시각으로 대신 쓰던 오차를 관측할 수 있다 | 평균으로 사라진 고주파 움직임은 복구하지 못한다 |
| 남은 2-byte와 flag 사용 | 추가 전송량 없이 기존 decoder와 공존한다 | offset 표현 범위는 −32768~32767µs다 |
| `HAL_GetTick()` 사용 | 추가 배선·센서 읽기가 필요 없다 | 원래 시계는 1ms 단위다. µs 단위 필드가 1µs 정확도를 뜻하지 않는다 |
| MCU에서 누적 | Pi 저장량은 거의 늘지 않는다 | 누적 상태·정수 연산이 추가된다. target RAM·실행 시간은 미측정이다 |

읽기 완료 시각은 센서 내부 acquisition 시각과 다르다. 이 패치만으로 카메라와 동기화가 완료되거나 `imu_clock_verified`가 true가 되지 않는다. 더 정밀한 방식은 센서 data-ready와 고해상도 MCU timer를 함께 검토해야 한다. 전체 raw 표본 전송은 별도 프로토콜과 대역폭 검토가 필요하다.

## Wire 계약

- version 2, 길이 144 bytes 유지.
- byte 140~141: little-endian signed int16 `imu_mean_read_offset_us`.
- flags bit 12: `SPI_TLM_FLAG_IMU_READ_TIME_VALID`.
- 유효 조건: 기존 `IMU_VALID`와 새 timing flag가 모두 설정됨.
- 같은 MCU clock에서 `mean_read_ms = t_ms + imu_mean_read_offset_us / 1000`. `t_ms` wrap을 고려해야 한다.
- 범위를 벗어나거나 표본이 없거나 시각이 뒤로 가면 timing을 무효화한다. clipping하지 않는다.
- 평균 표본이 없어 순간값을 읽는 기존 `imu_n=0` 경로도 성공 시 읽기 시각을 보낸다. 실패하면 timing을 무효화한다.
- 기존 CRC는 새 2-byte까지 보호한다. command encoding은 바뀌지 않는다.

새 Python decoder는 `imu_timing_valid`, `imu_mean_read_offset_us`, `imu_time_source`를 추가한다. 구버전 firmware에서는 timing flag가 없으므로 offset/source가 `None`이다. mapper에서 offset을 사용하는 연결은 별도 작업이다.

## 대상과 재현

대상 KSRC commit은 `40499bcb1acaabf6bbbf604795caf28b6a952739`이다. `BASELINE.json`에 여섯 원본 파일의 SHA256을 고정했다. 다른 revision이나 이 저장소의 오래된 `teleop/ksrc_snapshot`에 그대로 적용하지 않는다.

baseline checkout을 준비한 뒤 아래 명령으로 패치를 재생성한다. baseline은 읽기만 하고, 이 폴더의 patch를 출력한다.

```bash
PYTHONNOUSERSITE=1 /home/su/rover/.venv-mapping/bin/python \
  experiments/imu_timing_patch/build_patch.py /path/to/ksrc-baseline
```

테스트는 baseline의 임시 복사본에 `patch --fuzz=0`으로 적용한다. 실제 Pi source에는 적용하지 않는다.

```bash
KSRC_IMU_TIMING_BASELINE=/path/to/ksrc-baseline \
PYTHONNOUSERSITE=1 /home/su/rover/.venv-mapping/bin/python \
  -m pytest tests/imu_timing -q
```

baseline 환경변수가 없으면 portable C 산술 테스트 6개만 실행하고 통합 테스트 5개는 skip한다. C compiler `cc`가 필요하다.

## 검증과 적용 순서

native C/UBSan 및 실제 driver 함수·SPI pack/parse·신구 Python decoder를 사용하는 테스트 **11개 통과**. 규칙/불규칙 간격, clock wrap, 범위 초과, 빈 표본, 역행, count overflow, 평균값 보존, 순간값 fallback, CRC 및 기존 command 호환성을 확인했다.

ARM 전체 firmware build와 정지 상태 벤치 적용도 확인했다. 원본 대비 text +656 bytes, bss +48 bytes이며, 30초 시험에서 유효 IMU 고유 표본 1,190개 모두 timing flag가 설정됐다. 디코딩 실패·MCU 시각 역행은 없었다. 추가 연산의 최악 실행 시간과 카메라 동시 운용 부하는 아직 측정하지 않았다. 적용 및 후속 검증 순서는 다음과 같다.

1. 대상 revision에서 ARM target build와 정적 확인.
2. 실제 구동 전 simulation/mock 검증을 우선 권고한다. 현재 변경은 IMU·telemetry 경로에 한정되며 native mock 검증은 통과했다.
3. 모터 구동을 차단한 bench에서 POLL-only로 값·flag·offset·loop 주기·reset 동작 확인.
4. 새 시각을 Pi 기록 및 시간 정렬에 연결하고 기존 기록 방식과 비교.
5. 카메라/IMU offset을 재측정하고 offline stereo-inertial 비교. 기존 기록으로 하는 실험은 firmware 적용과 독립적으로 진행할 수 있다.

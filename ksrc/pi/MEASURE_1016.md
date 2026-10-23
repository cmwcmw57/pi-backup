# 10-16 실측 절차 (바닥, 모래 없음)

목적: (1) RL 정책이 실물에서 안정적으로 도는지, (2) student v2(가속도 관측) 학습에 넣을 **IMU 잡음·진동·바이어스·기울기**와 **Pi 루프 주기**를 실측.
모든 주행은 `run_policy.py --log` 로 원시 텔레메트리(IMU·바퀴·duty·조향·수신 시각)를 남긴다. 분석은 `pi/common/imu_noise.py`.

## 준비 (한 번)
- 펌웨어 0x03 굽기: `cd ~/ksrc/firmware/src2026-rover-firmware && ./build.sh --flash` (노트북 USB)
- 라파: `cd ~/ksrc && git pull && bash sim/build.sh`, torch venv (`~/venv/bin/python`)
- 정책 파일: `res8_mlp_v1_rc4000.{pt,json}` 를 `~/ksrc/pi/policy/` 로
- 배터리 완충, Pi 전원 5 V 5 A(PD 정격 확인), D435 는 USB3
- 바닥: 평평하고 단단한 곳, 직선 2 m 이상, 출발선·1.5 m 지점 테이프
- 로그 폴더: `mkdir -p ~/ksrc/logs/1016`
- 아래 `RUN` = `~/venv/bin/python pi/policy/run_policy.py --policy pi/policy/res8_mlp_v1_rc4000 --rt 50 --cpu 3 --gc-off`

## 측정 순서 (총 약 40분)

| # | 무엇 | 환경·방법 | 명령 | 보는 것 |
|---|---|---|---|---|
| M0 | 링크·IMU 기본 | 로버 수평, 바퀴 띄움 | `python3 pi/common/bringup.py link` | 텔레메트리 정상, `imu_n` 3–4 |
| M1 | **정지 IMU 기준** | 로버 바닥에 수평, 모터 전원 ON, 손 대지 않기, 60 s | `$RUN --residual-off --cmd 0 0 0 --seconds 60 --log logs/1016/still.npz` | 자이로 바이어스·잡음, 가속도 잡음, 정지 기울기, imu_n |
| M2 | **축 부호** (10 s씩) | 손으로: 위에서 반시계로 돌리기 / 앞 들기 / 왼쪽 들기 | `$RUN --residual-off --cmd 0 0 0 --seconds 10 --log logs/1016/axes_yaw.npz` (pitch, roll 도 같은 식) | 반시계 → gyro z +, 앞 들기 → gravity x −, 왼쪽 들기 → gravity y − |
| M3 | **주행 진동, IK** | 바닥 직진 1.5 m, 잔차 끔, 속도 3단계 | `$RUN --residual-off --cmd 0.10 0 0 --seconds 12 --log logs/1016/ik_v010.npz` · `0.20` → 7 s · `0.30` → 5 s | 속도별 가속도 잡음·진동 주파수(그라우저 14 × 회전수), 루프 주기 |
| M4 | 회전 진동, IK | 제자리 회전 0.6 rad/s, 잔차 끔 | `$RUN --residual-off --cmd 0 0 0.6 --seconds 8 --log logs/1016/ik_w06.npz` | 회전 중 잡음, 조향 반전·게이트 |
| M5 | **RL 안정성** | M3·M4 와 같은 명령, **잔차 켬** | 위 명령에서 `--residual-off` 빼고 파일명 `rl_*.npz` | 잔차가 ±1 에 붙는지, 조향 떨림(jerk), 바퀴 명령 흔들림, 루프 p99 |
| M6 | **Pi 부하** (전원 OK 일 때만) | M5 의 0.20 m/s 를 ORB 동시 실행으로 | ORB 띄운 뒤 `$RUN --cmd 0.20 0 0 --seconds 7 --log logs/1016/rl_v020_orb.npz` + `vcgencmd get_throttled` · `vcgencmd measure_temp` | 루프 p99·25 ms 초과 횟수, 스로틀링, ORB Hz |

- M3·M5 는 휴대폰으로 옆에서 촬영(테이프 1.5 m 기준) → 실제 속도 기준.
- 이상하면 즉시 멈춤: 조이스틱/스페이스. 정책은 끝나면 0 명령 5번 보내고 멈춘다.

## 끝나고 (노트북)
```bash
scp pi@raspberrypi.local:~/ksrc/logs/1016/*.npz ~/ksrc/sysid_logs/1016/
python3 pi/common/imu_noise.py ~/ksrc/sysid_logs/1016/{still,ik_v010,ik_v020,ik_v030,ik_w06}.npz --out ~/ksrc/sysid_logs/1016/imu_noise.json
```
`suggest` 의 값(자이로 잡음·바이어스, 정지 기울기, 주행 중 가속도 잡음)을 시뮬 `STUDENT_NOISE`·`GRAV_ERR`·v2a 가속도 잡음에 넣는다.
M5 는 잔차 포화·jerk·루프를 따로 본다 (Claude 에게 npz 경로를 주면 분석).

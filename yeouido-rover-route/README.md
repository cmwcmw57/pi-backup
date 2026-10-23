# yeouido-rover-route

여의도 모래 경사 램프(A 10°, B 30°, C 15°) 주행 시험용 경로 실행기와 실기 연결·분석 도구.

## 구성
| 경로 | 내용 |
|---|---|
| `docs/yeouido_route_plan.json` | STL 좌표계 경로: A 직진 → U턴 → B 시도(막히면 후퇴) → C 오르기 |
| `docs/yeouido_run_sheet.md` | 실행 구성(정책, 속도, 위치 추정)과 남은 틈 |
| `docs/yeouido-data-collection-checklist.md` | 현장 데이터 수집 체크리스트 |
| `experiments/orbslam3_ros2/yeouido_route.py` | `RouteRunner`: pose를 받아 (vx, vy, ω) 생성, 체크포인트 정지, B 정체 시 포기, IMU 기울기 중단 |
| `experiments/orbslam3_ros2/stl_to_usda.py` | binary STL(mm) → 미터 단위 USDA |
| `experiments/yeouido_real/` | 실기 어댑터: pose 추정, KSRC UDP 명령 송수신, 실행 루프 (`--fake` 지원) |
| `experiments/yeouido_analysis/` | 슬립, 카메라-IMU 시각차 스윕, ORB 유지율 분석 |
| `experiments/yeouido_real/open_loop.py` | 열린 루프: 경로를 명령표(CSV)로 만들고 50 Hz로 재생 |
| `pi_patch/` | Pi `run_policy.py` 텔레메트리 복사 패치(적용은 사용자가 직접) |
| `docs/yeouido_field_procedure.md` | 현장 실행 절차 |
| `tests/` | pytest |

## 실행
```bash
python3 -m pytest -q
python3 -m experiments.yeouido_real.run_real --fake
```

## 상태
- 이상적 운동학 시뮬과 합성 입력으로만 검증했다. 실기체 검증은 하지 않았다.
- 실기 SPI·모터·CAN 쓰기 코드는 없다. 명령은 UDP 패킷(`b"KSRC"`, seq, vx, vy, ω)으로만 내보낸다.
- Isaac Sim 실행 스크립트와 팀원 저장소 코드는 포함하지 않았다.
- 라이선스는 아직 정하지 않았다.

# Isaac Lab 이주 계획 검토 메모

2026-10-08 작성, 2026-10-09 순수 IK 비교 결과 반영. `ISAAC_MIGRATION.md`의 구현 단계에서 확인할 불일치다.

1. **컨테이너에 `PYTHONPATH`가 전달되지 않았음 — 수정·검증 완료.** 계획 §5.3의 초기 `PYTHONPATH=/workspace/source bash run.sh ...`는 호스트에만 적용됐다. `sim/isaac/ksrc_rover/run.sh`에 컨테이너 `-e PYTHONPATH=/workspace/source`와 `-w /workspace`를 넣었고, 서버 Direct8/Residual3/Res8 task 생성 smoke에서 import가 통과했다.

2. **IK 구현 설명 충돌 — 문서 수정·수치 검증 완료.** 계획 §5.4는 Isaac 측 배치 IK를 PyTorch로, 펌웨어 C/ctypes를 로컬 검증 오라클로 정했지만 과거 Phase 2에는 `.so` 공통 사용으로 적혀 있었다. Phase 2 문구를 고쳤고 `sim/rl/ik_validate.py` 1000개 연속 명령에서 최대 절대오차 `6e-8`을 확인했다.

3. **실측 조향오차 게이트가 현재 펌웨어 경로에서 도달 불가다.** `firmware/src2026-rover-firmware/Core/Src/main.c`의 `Swerve_Control()`은 `s_gate.pending`일 때만 `drive_gate_allow()`를 호출한다. 반면 `firmware/common/drive_align_gate.c`의 실측오차 트리거는 `!g->pending && max_err > g->trigger_rad`에서 발동한다. 목표각이 매 주기 조금씩 변해 목표 점프 트리거를 피하면서 실제 서보만 크게 뒤처지면, 해당 보호 분기가 호출되지 않는다. 기본 조향과 큰 목표점프 게이트는 별개로 동작하지만, 이 실측오차 보호는 수정 후 통합 테스트가 필요하다.

4. **관측 차원 계약이 계획과 현재 MuJoCo에서 다르다.** 계획 §4는 Actor 26D×이력으로 적었지만 현재 `sim/rl/env.py`의 `OBS_PER_FRAME`은 36이다. Isaac 이주 task의 첫 관측도 29D로 시작한다. 이 세 숫자가 다르면 체크포인트와 sim2sim 입력을 공유할 수 없다. Phase 2에서 실기 센서로 구성 가능한 항목을 확정하고 단일 `contract.py`를 양쪽에 적용해야 한다.

5. **초기 Isaac task는 학습 검증 완료 상태가 아니다.** 현재 평지 bring-up은 변환 USD를 임시 사용한다. 바퀴 DCMotor 속도 루프 게인은 1.0에서 0.20으로 조정했고, 로커 소프트 구속은 K=20,D=0.3으로 강화했으며, 평지 순수 IK 기준선을 실제 대조했다. 그러나 전체 명령 시퀀스의 곡선 횡속도 차이는 29.9%로 Phase 3 관문 미통과다. 기수 오프셋·terrain loader·관측 계약·모래 전이·실기 토크 모델도 미검증이다. 상세 자료는 [벤치마크](../isaac/ksrc_rover/benchmarks/2026-10-09_pure_ik/RESULTS.md).

6. **액션별 보상 페널티 비교 주의.** 3D/8D raw action에 같은 `action_rate_l2`를 쓰면 차원과 액션 단위가 달라 서로 다른 최적화 문제가 된다. bring-up 보상은 공통 물리량인 속도·각속도 추종 항만 두고, 잔차·smooth 항은 단위와 정규화를 확정한 뒤 모든 모드에서 같은 물리적 의미로 넣어야 한다.

7. **순수 IK 대조에서 기하·액추에이터 계약 오류 발견 및 수정.** 초기 Isaac 계약의 휠 반지름 0.0405 m, 조향축 (±0.105,±0.100) m는 현재 CAD/서버 URDF의 0.047503 m, (±0.102431,±0.1325) m와 달랐다. 고친 뒤 펌웨어 C↔Torch IK 1000개 연속 명령 최대 오차는 약 `9e-8`. 5 ms 물리 스텝에서 DCMotor 속도 게인 1.0은 무명령 휠 속도 평균 약 2.52 rad/s 진동을 냈고, 0.20에서는 그 진동이 사라졌다. Isaac 조향 목표에도 5.45 rad/s slew 제한을 넣었다. 이 값들은 평지 기준선 보정이며 실측 모터 시스템 식별을 대체하지 않는다.

8. **남은 Phase 3 차이를 IK 버그로 단정하면 안 된다.** 조향·휠 목표가 Isaac과 MuJoCo에서 일치하고 단독 좌·우 곡선은 20% 이내다. 이전 동작을 거친 전체 시퀀스의 곡선 횡속도만 정상구간 29.9% 차이가 난다. 로커 구속 형태(Isaac 소프트, MuJoCo 하드), 바퀴 속도 응답, 접촉 이력 등이 후보지만 원인을 분리하지 못했다. 로커 합 오차도 Isaac 0.18°, MuJoCo 0.002°로 남아 있다.

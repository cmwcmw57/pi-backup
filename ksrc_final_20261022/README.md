# ksrc_final_20261022 (Pi 배포: 새 맵 f 설정 최종 조합 중 지금 있는 부분)
- experiments/: f 설정 코드 스냅샷(cl1022_ksrcrun1f_depth_s2 기준) + control_gateway.py(2026-10-22, 커밋 b2081cf)
- mission/: 새 맵 mission(scenario.json, regions.json, site 지도, 경로). sha256은 로컬 번들과 일치 확인
- deps/: scipy 1.14.1(격리 설치본, 기존 upper_control_probe_20261021_01/deps 복사). 사용: sys.path.insert(0, deps)
- 아직 없는 것: ORB pose+IMU→ESKF→RegionMission→run_policy(--cmd-udp 5005, --vo-port) 를 잇는 실기 실행기. 모터·SPI는 이 폴더의 어떤 것도 건드리지 않는다.

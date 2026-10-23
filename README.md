# Raspberry Pi 프로젝트 백업

2026-10-23에 Raspberry Pi의 프로젝트 파일을 복사한 스냅샷입니다.

| 폴더 | Pi의 원본 경로 | 설명 |
| --- | --- | --- |
| `ksrc/` | `/home/pi/ksrc` | 펌웨어, Pi 실행 코드, 모델, 지도 및 실행 로그 |
| `ksrc_final_20261022/` | `/home/pi/ksrc_final_20261022` | 10월 22일 최종 배포본, 실험, mission 및 의존성 |
| `yeouido-rover-route/` | `/home/pi/yeouido-rover-route` | 여의도 경로 프로젝트 |

각 프로젝트의 Git 메타데이터(`.git`)를 제외한 파일을 포함합니다. 원본 Pi의 파일은 변경하지 않았습니다. 실행 방법과 요구 사항은 각 폴더의 README 및 문서를 참고하세요.

10 MiB 이상인 파일은 Git LFS로 보관합니다. 전체 파일을 받으려면 Git LFS를 설치한 뒤 다음을 실행하세요.

```bash
git lfs install
git clone https://github.com/cmwcmw57/pi-backup.git
cd pi-backup
git lfs pull
```

`ksrc` 원본 저장소: https://github.com/YunBeeeen/ksrc

`yeouido-rover-route` 원본 저장소: https://github.com/lee-robot-lab/yeouido-rover-route

원본 프로젝트의 라이선스와 저작권 표기는 각 프로젝트 폴더를 참고하세요.

# 공유 서버(hpi-lab, sumee@) RL 학습 사용 규칙

RL 주행보조 학습이 공유 서버를 쓸 때 지키는 규칙. **동료 작업과 충돌 안 나게** 하는 게 목적.

## GPU
- **(임시 규칙) `run.sh`의 `GPU=1` 그대로, GPU 1이 비어 있을 때만 실행.** 차 있으면 기다린다(임의로 0번으로 바꾸지 않음).
- **어떤 실행이든(학습·eval·분석) 띄우기 전 항상 `nvidia-smi`로 GPU 확인.**
- **학습은 1개씩, 기본 4096env.** (VRAM ≈ 2048env 11.6GB / 4096env 12.7GB)

## 동료 자원 보호
- **동료 컨테이너 절대 stop/rm 안 함**: `oclat-dev`, orbslam3, isaac-sim 관련 등.
- **동료 terrain 패키지 수정 안 함**: `/terrain`(= `experiments.rover_training_terrains`)·`/assets`는 **읽기전용 마운트**로만 사용.

## 컨테이너/이미지
- 우리 이미지: **`ksrc-rover:lab2.3.2-rsl312`** (원본 `lab2.3.2`에 rsl-rl-lib 3.1.2 설치 후 commit). **원본 보존**.
- 텐서보드 컨테이너 이름 = **`ksrc_tb`** (같은 이미지 사용).
- ⚠ **`docker stop --filter ancestor=ksrc-rover...` 금지** — TB(`ksrc_tb`)도 같은 이미지라 같이 꺼진다. **반드시 컨테이너 ID로 지정해서 stop.**

## 실행 규칙
- **중복 실행 방지(가드)**: 띄우기 전
  `docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb`
  로 **학습 컨테이너 개수 확인** → 기대값일 때만 실행. (ssh가 끊겨도 서버에선 실행돼 중복 OOM 잘 남)
- **긴 학습은 detached**: `setsid nohup bash run.sh ... </dev/null > logs/xxx.log 2>&1 & disown`. → VPN 끊겨도 계속 돈다.
- **여러 개 순차**는 체인 스크립트로 (`run.sh`는 `docker run --rm` 포그라운드라 각 학습이 끝날 때까지 블록 → 겹침 없음). 예: `chain3.sh`.
- **`timeout`으로 docker run 감싸지 말 것** — 클라이언트만 죽고 컨테이너는 살아남아 겹친다.

## 파일 위치
- 코드/데이터 루트: **`~/gyusa/ksrc_rover`** (노트북 `deploy.sh`로 동기화되는 미러 — 서버에서 직접 수정 금지, 다음 deploy에 덮임).
- 학습/eval 산출물: `logs/` (rsl_rl 런dir, eval npz 등).
- ⚠ **rsl_rl 런dir 타임스탬프: 2026-10-10 22시 이전 런은 UTC(KST−9h), 이후는 KST.** 이미지에 tzdata가 없어 `TZ=Asia/Seoul`이 무시됐음 → `run.sh`에서 `TZ=KST-9`(시간대 파일 불필요)로 수정.

## 접속
- `ssh hpi-lab`. SSH 간헐 끊김 잦음 → `~/.ssh/config`에 ControlMaster(다중화) 설정으로 완화. 재시도 루프 권장.

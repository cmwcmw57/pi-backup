/*
 * wheel_pi.h
 *
 * 구동 바퀴 속도 제어: 전향보상(FF) + 기준 모델 PI.  바퀴 하나당 상태 하나.
 * 순수 C (HAL 없음) -> 호스트 gcc 테스트, 시뮬(ctypes) 에서 같은 코드를 돈다.
 * Isaac 은 GPU 배치라 ctypes 를 못 써서 torch 로 옮기고, 같은 입력열로 이 C 와 비교 검증한다.
 *
 * 설계 근거: firmware/MOTOR_MODELING_PLAN.md §8 (2026-10-13).  플랜트는 실측
 * (sysid_logs/20261013_102224_drive): duty 1 당 0.0097 rad/s, 시정수 40 ms,
 * 운동마찰 ~5 duty, 정지마찰 기동 35–52 duty, 저속에서 ~33 duty 아래면 멈춤.
 *
 * 한 제어주기 (dt = 20 ms, main.c 50 Hz 루프):
 *   r_m  += (w_ref - r_m) * dt / (tau_m + dt)       기준 모델: FF 만으로 바퀴가 따라갈 궤적
 *   e     = r_m - w_meas                            PI 는 그 궤적과의 차이만 고친다 (계단 오버슈트 없음)
 *   ff    = w_ref / k_duty + sgn(w_ref) * f_kin     (+ sgn(w_ref) * kick  바퀴가 멈춰 있으면: 정지마찰 넘기)
 *   duty  = clip(ff + kp*e + i, ±duty_max)
 *   i    += ki * e * dt   (출력이 e 방향으로 포화면 안 쌓음),  |i| <= i_max
 *   |w_ref| < w_zero 이면 duty 0 (MDD3A 브레이크), 상태 초기화.
 *
 * 보호 (2026-10-13, 모래 지면에서 PI 가 끝까지 밀어붙이는 것 방지):
 *   1) 보정량 상한: c = kp*e + i 를 ±corr_max 로 자른다.  최악에도 duty <= FF + corr_max
 *      (개루프보다 조금 센 정도).  c 가 상한에 붙은 방향으로는 적분하지 않는다.
 *   2) 스톨: c 가 진행 방향 상한에 붙어 있고 |w_meas| < stall_w 인 상태가 stall_t 이상 -> stall=1:
 *      FF 만 (킥 없음, 적분 0) = 개루프처럼 양보.  바퀴가 |w_ref| 의 절반 이상으로 같은 방향으로
 *      돌면 PI 로 돌아가고, 정지 명령(reset)에도 풀린다.  엔코더가 죽어 틱이 0 이어도 여기로 온다.
 *   3) 엔코더 방향 불일치 (ENC_DIR): |duty| >= enc_duty 인데 바퀴가 반대로 |w_meas| >= enc_w 로 도는
 *      상태가 enc_t 이상.  엔코더 A/B·모터 +/- 가 바뀌었거나 MOTOR_DRIVE_SIGN·커넥터가 틀리면 PI 는
 *      양의 되먹임으로 폭주한다 (기어비 131 이라 주행 중 진짜로 거꾸로 밀리는 일은 드묾).
 *      재시도: 걸리면 enc_retry_t 동안 FF 만, 그 뒤 PI 다시.  구동 중 enc_window_t 안에 enc_trips_max
 *      번 걸리면 enc_dir=1 로 잠근다 (FF 만, reset 으로 안 풀림 -> wheel_pi_clear_fault, WPI ON).
 *      재시도·창 시간은 구동 명령이 있는 주기만 센다.
 *
 * 전압 처짐·모터 개체차·경사 부하는 적분항이 메운다 (FF 는 고정 기울기).
 * 바퀴가 헛돌면 PI 는 회전속도를 목표에 묶을 뿐 헛돎 자체는 못 없앤다 -> 정책 몫.
 * 호출하는 쪽 책임:
 *   - 정렬 게이트가 구동을 막는 동안은 wheel_pi_reset() 하고 duty 0.
 *   - w_meas 는 이번 주기 엔코더 차분 (20 ms 창, 1틱 = 0.0375 rad/s).
 */

#ifndef WHEEL_PI_H
#define WHEEL_PI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float dt;          /* 제어주기 [s] */
    float k_duty;      /* FF 기울기: duty 1 당 정상속도 [rad/s] */
    float f_kin;       /* FF 운동마찰 보상 [duty] */
    float kick;        /* 멈춰 있을 때 더하는 정지마찰 킥 [duty] */
    float w_still;     /* |w_meas| 가 이보다 작으면 "멈춤" [rad/s] (1틱 0.0375 의 절반 아래) */
    float w_zero;      /* |w_ref| 가 이보다 작으면 정지 명령 [rad/s] */
    float kp;          /* [duty / (rad/s)] */
    float ki;          /* [duty / rad] */
    float i_max;       /* 적분항 상한 [duty] */
    float tau_m;       /* 기준 모델 시정수 [s] (0 = 끔, 오차를 w_ref 에 직접) */
    float duty_max;    /* PWM 상한 (799) */
    float corr_max;    /* 보호 1: |kp*e + i| 상한 [duty] */
    float stall_w;     /* 보호 2: 이보다 느리면 "안 돎" [rad/s] */
    float stall_t;     /* 보호 2: 그 상태 지속 시간 [s] */
    float enc_duty;    /* 보호 3: 이 이상 duty 일 때만 검사 */
    float enc_w;       /* 보호 3: 반대 방향 속도 문턱 [rad/s] */
    float enc_t;       /* 보호 3: 지속 시간 [s] */
    float enc_retry_t; /* 보호 3: 걸린 뒤 FF 만 내는 시간 [s], 그다음 PI 재시도 */
    float enc_window_t;/* 보호 3: 이 시간 안에 */
    float enc_trips_max; /* 보호 3: 이 횟수 걸리면 잠금 */
} wheel_pi_cfg_t;

typedef struct {
    float r_m;         /* 기준 모델 출력 [rad/s] */
    float i;           /* 적분항 [duty] */
    uint16_t stall_n;  /* 스톨 조건 연속 주기 수 */
    uint16_t enc_n;    /* 방향 불일치 연속 주기 수 */
    uint16_t enc_hold; /* 재시도 대기: 남은 FF 전용 주기 수 */
    uint16_t enc_win;  /* 첫 번째 걸림 뒤 지난 주기 수 (창) */
    uint8_t enc_trips; /* 창 안에서 걸린 횟수 */
    uint8_t stall;     /* 1 = 스톨로 FF 만 내는 중 */
    uint8_t enc_dir;   /* 1 = 방향 불일치로 잠김, FF 만 (clear_fault 전까지) */
} wheel_pi_state_t;

/* 2026-10-13 설계값 (MOTOR_MODELING_PLAN.md §8). */
void wheel_pi_default_cfg(wheel_pi_cfg_t *cfg);

/* 적분·기준모델·스톨을 0 으로. 방향 불일치 상태(enc_dir·재시도·창)는 그대로 둔다. */
void wheel_pi_reset(wheel_pi_state_t *s);

/* 전부 0 (enc_dir 포함). 부팅·WPI ON 때. */
void wheel_pi_clear_fault(wheel_pi_state_t *s);

/* 한 주기. 반환 = 부호 있는 duty (+ = 바퀴 전진, 모터 부호 보정 전), 정수로 자른 값. */
int16_t wheel_pi_step(const wheel_pi_cfg_t *cfg, wheel_pi_state_t *s, float w_ref, float w_meas);

#ifdef __cplusplus
}
#endif

#endif /* WHEEL_PI_H */

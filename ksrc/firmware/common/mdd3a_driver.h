/*
 * mdd3a_driver.h
 *
 * Cytron MDD3A 구동모터 듀티 계산. 순수 C 라 HAL 의존성이 없다.
 * PWM 입력별로 [0,1] 정규화 듀티를 내주고, 실제 펌웨어가 타이머 ARR 을 곱해
 * 쓴다 (pulse = duty * ARR).
 *
 * 근거: MDD3A 는 Cytron 의 다른 보드 대부분과 달리 **PWM+DIR 방식이 아니다**.
 * Cytron 자체 아두이노 라이브러리에 PWM_PWM 모드 호환으로 명시돼 있다
 * (github.com/CytronTechnologies/CytronMotorDriver, CytronMotorDriver.cpp,
 * 2026-09-13 확인). 채널마다 PWM 입력이 둘이다 (A=정방향, B=역방향):
 *
 *   speed >= 0:  A = PWM(speed),  B = 0
 *   speed <  0:  A = 0,           B = PWM(|speed|)
 *
 * 즉 방향은 별도 DIR 선이 아니라 **어느 핀에 PWM 이 실리느냐**로 정해진다.
 * 아래 mdd3a_speed_to_duty() 가 이 로직을 그대로 재현한다
 * (CytronMotorDriver.cpp 의 setSpeed() 와 대조 검증).
 * 2채널 MDD3A 한 장에 PWM 출력 4개가 필요하고, 로버는 보드 2장(구동모터 4개)
 * 이라 **독립 PWM 타이머 채널이 총 8개** 필요하다.
 *
 * 미해결: 아래 `max_speed_mps` 는 바퀴별 캘리브레이션 상수로, 부하 상태에서
 * 듀티 100% 일 때의 선속도(m/s)다. 정격 전압에서의 무부하 RPM, 기어비,
 * 바퀴 지름이 정해진 뒤 벤치 측정이 필요하다. 그 전까지 이 파일은 [-1,1]
 * 정규화 영역만 다룬다.
 */

#ifndef MDD3A_DRIVER_H
#define MDD3A_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float duty_a; /* [0,1], "정방향" 입력의 PWM 듀티 */
    float duty_b; /* [0,1], "역방향" 입력의 PWM 듀티 */
} mdd3a_channel_cmd_t;

/* speed_norm 은 부호 있는 값이며 사용 전에 [-1, 1] 로 클램프된다.
 * duty_a 와 duty_b 가 동시에 0 이 아닌 경우는 구조적으로 없다
 * (참조 라이브러리의 상호배타성과 동일). */
mdd3a_channel_cmd_t mdd3a_speed_to_duty(float speed_norm);

/* 편의 함수: 물리 속도(m/s)를 바퀴별 최고속도 캘리브레이션 상수로 나눠
 * [-1,1] 로 클램프한다 (위 주의사항 참고). max_speed_mps <= 0 이면
 * "아직 캘리브레이션 안 됨" 으로 보고 0 을 돌려준다. */
float mdd3a_normalize_speed(float speed_mps, float max_speed_mps);

#ifdef __cplusplus
}
#endif

#endif /* MDD3A_DRIVER_H */

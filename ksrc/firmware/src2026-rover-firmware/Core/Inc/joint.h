/**
 * @file    joint.h
 * @brief   관절(Joint) 각도 관리 모듈
 *
 * STS3215 서보 모터의 물리적 각도(°)와 서보 위치값(0~4095)의
 * 변환, 범위 제한(클램프), 다축 동시 제어를 담당합니다.
 *
 * 핵심 설계: 입력 소스(터미널/SPI) 분리
 *   on_target_angles_received() 함수가 각도를 받아 SYNC_WRITE로
 *   서보를 이동시킵니다. 입력 소스와 모터 제어 로직이 완전히 분리되어,
 *   나중에 SPI 수신 콜백에서 같은 함수를 호출하면 됩니다.
 */

#ifndef JOINT_H
#define JOINT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>

/* ========================== 서보 모터 수 설정 ========================== */
/**
 * 연결된 조향 서보 모터 수.
 * 나중에 4개로 늘릴 때 이 값만 변경하고 joint_config[] 배열에 항목을 추가하면 됩니다.
 */
#define SERVO_COUNT  4

/* ========================== 관절 설정 구조체 ========================== */
/**
 * @brief  각 관절(서보)의 물리적 설정
 *
 * offset, dir, min/max는 캘리브레이션 후 실제 로봇에 맞게 수정합니다.
 */
typedef struct {
    uint8_t  id;        /**< 서보 버스 ID (1~253) */
    int16_t  offset;    /**< 0도에 해당하는 위치값 (기본 2048) */
    int8_t   dir;       /**< 회전 방향 (+1 또는 -1) */
    float    min_deg;   /**< 허용 최소 각도 (도) */
    float    max_deg;   /**< 허용 최대 각도 (도) */
} JointConfig_t;

/**
 * @brief  서보 한 개의 실측값 (레지스터 56~65 를 한 번에 읽은 것)
 *
 * 각도·속도·부하는 offset/dir 을 적용한 **관절 기준** 값이다 (T 명령과 같은 부호).
 */
typedef struct {
    float    angle_deg;   /**< 현재 각도 (도) */
    int16_t  speed;       /**< 현재 속도 step/s (4096 step = 360도) */
    int16_t  load;        /**< 부하 0.1% 단위, -1000..1000 */
    uint8_t  voltage;     /**< 전압 0.1V 단위 */
    uint8_t  temp;        /**< 온도 섭씨 */
    uint8_t  status;      /**< 상태 비트 (레지스터 65) */
} JointFeedback_t;

/* ========================== 함수 프로토타입 ========================== */

/**
 * @brief  관절 모듈 초기화
 *
 * STS3215 드라이버가 이미 sts_init()으로 초기화된 상태에서 호출합니다.
 * 각 서보에 가속도 설정 + 토크 ON을 수행합니다.
 */
void joint_init(void);

/**
 * @brief  ★ 핵심 콜백 함수 — 목표 각도를 받아 서보를 이동시킴
 *
 * 현재는 PC 터미널 파서가 호출합니다.
 * 나중에 SPI 수신 콜백에서도 동일하게 호출하면 입력 소스 교체 완료.
 *
 * @param  theta  목표 각도 배열 (도 단위, 예: {10.5, -20.0})
 * @param  count  배열 크기 (SERVO_COUNT와 일치해야 함)
 * @return 가동범위 경계로 잘린 각도 개수 (0 = 전부 범위 안)
 */
int on_target_angles_received(float *theta, int count);

/**
 * @brief  관절 가동범위 (도). 기구학이 조향각을 이 범위 안으로 접을 때 쓴다.
 * @param  idx  관절 인덱스 (0 ~ SERVO_COUNT-1)
 */
void joint_get_limits_deg(int idx, float *min_deg, float *max_deg);

/**
 * @brief  현재 각도 읽기 (실패해도 출력하지 않음 -- 제어주기용)
 * @return 1 = 성공, 0 = 읽기 실패
 */
int joint_read_angle(int idx, float *deg);

/**
 * @brief  위치·속도·부하·전압·온도·상태를 한 번의 READ 로 읽기 (10바이트)
 *
 * 위치만 읽는 것과 비교해 서보당 약 80us 더 걸린다 (1Mbps).  한 번 읽기의
 * 대부분은 서보 반환 지연(기본 500us)이다.
 * @return 1 = 성공, 0 = 실패 (출력 없음)
 */
int joint_read_feedback(int idx, JointFeedback_t *fb);

/**
 * @brief  특정 관절의 현재 각도 읽기
 * @param  idx  관절 인덱스 (0 ~ SERVO_COUNT-1)
 * @return 현재 각도 (도), 에러 시 -999.0f
 */
float joint_get_angle(int idx);

/**
 * @brief  모든 서보의 이동 속도 설정
 * @param  speed  속도 값 (step/s, 0=최대속도)
 */
void joint_set_speed(uint16_t speed);

/** 지금 설정된 목표 속도 (S 명령 값, 0 = 최대) */
uint16_t joint_get_speed(void);

/**
 * @brief  모든 서보의 토크 ON/OFF
 * @param  on  1=ON, 0=OFF
 */
void joint_set_torque_all(uint8_t on);

#ifdef __cplusplus
}
#endif

#endif /* JOINT_H */

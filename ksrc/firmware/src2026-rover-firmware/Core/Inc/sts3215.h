/**
 * @file    sts3215.h
 * @brief   Feetech STS3215 시리얼 버스 서보 프로토콜 드라이버
 *
 * Dynamixel 1.0 호환 프로토콜을 사용하는 STS3215 서보 모터를 제어합니다.
 * "Servo Bus to TTL" 어댑터 보드를 경유하여 일반 전이중 UART로 통신합니다.
 *
 * 패킷 구조: 0xFF 0xFF ID LEN INSTR PARAM... CHECKSUM
 *   - LEN = 파라미터 수 + 2 (INSTR + CHECKSUM 포함)
 *   - CHECKSUM = ~(ID + LEN + INSTR + PARAM 합) & 0xFF
 *   - 16비트 값은 리틀엔디안
 */

#ifndef STS3215_H
#define STS3215_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>
#include "stm32f4xx.h"

/* ========================== 레지스터 주소 ========================== */
/* 주소 | 크기 | 설명 */
#define STS_REG_TORQUE_ENABLE     0x28  /* 1B  토크 ON/OFF (0=OFF, 1=ON) */
#define STS_REG_ACCELERATION      0x29  /* 1B  가속도 (0~254, 값 클수록 급가속) */
#define STS_REG_GOAL_POSITION     0x2A  /* 2B  목표 위치 (0~4095, 중앙=2048) */
#define STS_REG_GOAL_SPEED        0x2E  /* 2B  목표 속도 (step/s) */
#define STS_REG_PRESENT_POSITION  0x38  /* 2B  현재 위치 (읽기 전용) */

/* ========================== 프로토콜 상수 ========================== */
#define STS_HEADER          0xFF        /**< 패킷 헤더 바이트 */
#define STS_BROADCAST_ID    0xFE        /**< 브로드캐스트 ID (응답 없음) */

/* 명령어(Instruction) 코드 */
#define STS_INST_PING       0x01        /**< 핑 (연결 확인) */
#define STS_INST_READ       0x02        /**< 레지스터 읽기 */
#define STS_INST_WRITE      0x03        /**< 레지스터 쓰기 */
#define STS_INST_SYNC_WRITE 0x83        /**< 다축 동시 쓰기 */

/* 위치 범위 */
#define STS_POS_MIN         0           /**< 최소 위치값 */
#define STS_POS_MAX         4095        /**< 최대 위치값 */
#define STS_POS_CENTER      2048        /**< 중앙 위치값 (0도) */

/* ========================== 에러 코드 ========================== */
typedef enum {
    STS_OK = 0,             /**< 성공 */
    STS_TIMEOUT,            /**< 응답 타임아웃 */
    STS_CHECKSUM_ERR,       /**< 체크섬 불일치 */
    STS_ID_MISMATCH,        /**< 응답 ID 불일치 */
    STS_RX_ERROR            /**< UART 수신 에러 */
} STS_Status;

/* ========================== 함수 프로토타입 ========================== */

/**
 * @brief  드라이버 초기화 (USART3 레지스터 사용)
 */
void sts_init(void);

/**
 * @brief  서보에 PING 패킷 전송, 응답 확인
 * @param  id  서보 ID (1~253)
 * @return STS_OK이면 연결 정상
 */
STS_Status sts_ping(uint8_t id);

/**
 * @brief  레지스터에 데이터 쓰기
 * @param  id    서보 ID
 * @param  addr  시작 레지스터 주소
 * @param  data  쓸 데이터 포인터
 * @param  len   데이터 바이트 수
 * @return STS_OK이면 성공
 */
STS_Status sts_write(uint8_t id, uint8_t addr, uint8_t *data, uint8_t len);

/**
 * @brief  레지스터에서 데이터 읽기 (타임아웃 + 체크섬 검증)
 * @param  id    서보 ID
 * @param  addr  시작 레지스터 주소
 * @param  len   읽을 바이트 수
 * @param  buf   읽은 데이터를 저장할 버퍼
 * @return STS_OK이면 성공
 */
STS_Status sts_read(uint8_t id, uint8_t addr, uint8_t len, uint8_t *buf);

/**
 * @brief  토크 ON/OFF
 * @param  id  서보 ID
 * @param  on  1=ON, 0=OFF
 */
STS_Status sts_set_torque(uint8_t id, uint8_t on);

/**
 * @brief  가속도 설정
 * @param  id   서보 ID
 * @param  acc  가속도 값 (0~254, 0=최대 가속)
 */
STS_Status sts_set_acceleration(uint8_t id, uint8_t acc);

/**
 * @brief  다축 동시 위치+속도 이동 (SYNC_WRITE)
 *
 * 한 사이클에 모든 서보를 동시에 이동시킴 (응답 없음).
 * Goal Position(2B) + Goal Speed(2B) = 4바이트를 SYNC_WRITE로 전송.
 *
 * @param  ids        서보 ID 배열
 * @param  positions  목표 위치 배열 (0~4095)
 * @param  speeds     목표 속도 배열 (step/s)
 * @param  count      서보 수
 */
STS_Status sts_sync_write_position(uint8_t *ids, uint16_t *positions,
                                   uint16_t *speeds, uint8_t count);

/**
 * @brief  현재 위치 읽기
 * @param  id  서보 ID
 * @return 위치값 (0~4095), 에러 시 -1
 */
int16_t sts_read_position(uint8_t id);

#ifdef __cplusplus
}
#endif

#endif /* STS3215_H */

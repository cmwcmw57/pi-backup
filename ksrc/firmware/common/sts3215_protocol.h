/*
 * sts3215_protocol.h
 *
 * FEETECH STS3215 조향 서보 버스 프로토콜: 명령 패킷 조립기와 응답 파서.
 * 순수 C 라 HAL 의존성이 없다 (swerve_kinematics.h / teleop_protocol.h 와
 * 같은 호스트 테스트 가능 패턴).
 *
 * 근거: 역공학이나 기억에 의존한 게 아니라 FEETECH 공식 아두이노 라이브러리
 * (github.com/ftservo/FTServo_Arduino, src/SCS.cpp + SMS_STS.cpp,
 * 2026-09-13 확인) 를 바이트 단위로 옮긴 것이다. 여기 모든 패킷 배치는 그
 * 라이브러리 수식을 파이썬으로 처음부터 재구현해 교차검증했다. 그 결과로
 * 나온 바이트 단위 벡터는 test_sts3215_protocol.c 에 있다.
 *
 * 와이어 포맷 (반이중 TTL 버스, STS3215 기본 1,000,000 baud.
 * 여러 서보가 서로 다른 ID 로 버스를 공유한다):
 *
 *   요청:  0xFF 0xFF ID LEN INST [MEM_ADDR] [PARAMS...] CHECKSUM
 *   응답:  0xFF 0xFF ID LEN STATUS [PARAMS...] CHECKSUM
 *
 *   LEN = LEN 뒤에 오는 바이트 수, 즉 CHECKSUM 까지 포함한 전부.
 *         요청이면 1(INST) + 1(MEM_ADDR, 있는 경우) + param_len + 1(CHECKSUM).
 *         응답이면 1(STATUS) + param_len + 1(CHECKSUM).
 *   CHECKSUM = ~(ID + LEN + INST/STATUS + [MEM_ADDR] + sum(PARAMS)) & 0xFF
 *
 * 엔디언: 서보 기본값인 "End=0" 바이트 순서를 가정한다 (거의 모든 FEETECH
 * 예제가 쓰는 `SMS_STS()` 무인자 생성자와 동일). 16비트 레지스터 값은
 * 하위 바이트를 먼저 쓴다. 예: Goal_Position_L(42) 다음 Goal_Position_H(43).
 *
 * *** 0/4095 경계 문제 — 2026-09-21 해결. 근거는 FEETECH 공식 메모리표
 * (舵机协议内存表-磁编码版本.xlsx) ***
 * Goal_Position(레지스터 42/43) 은 모듈러가 아니라 **선형** 절대 스텝 목표다.
 * 서보의 위치루프 PID 가 error = target - current 를 레지스터 원값으로 그냥
 * 계산하며, 최단경로 개념이 아예 없다. 단일회전 모드(Operating Mode 0, 기본값.
 * 조향은 1회전을 넘길 일이 없으니 우리도 이걸 쓴다) 에서는 최소/최대 각도제한
 * 레지스터(9/11) 가 0/4095 로 기본 설정되어 목표를 그 창 안으로 클램프한다.
 * 따라서 순진하게 "0..4095 로 wrap" 하면, 명령각이 0/4095 이음매를 넘을 때마다
 * 서보가 **먼 쪽으로 최대 360도 돌아간다**. 예: 현재 4090 -> 목표 10 이면
 * 서보는 error = 10-4090 = -4080 으로 계산한다. 사람이 기대하는 +6 이 아니다.
 *
 * 해결: Goal_Position 의 실제 레지스터 범위는 부호 있는 -32766..32766 이다
 * (메모리표 참고. "다회전 지원, 전원 차단 시 회전수 미보존" 이 이 얘기다).
 * 다회전 모드 자체가 필요한 건 아니다 (Operating Mode 는 0 으로 두고,
 * 레지스터 18 의 bit4 도 레지스터 30 의 회전수 스케일도 안 건드린다).
 * 필요한 건 **목표값이 가끔 0..4095 를 조금 벗어나도 되는 것**뿐이다.
 * 그래야 이음매를 넘는 이동이 모듈러 wrap 이 아니라 작은 델타가 된다.
 * 이를 위해 최소/최대 각도제한 레지스터(9, 11) 를 0 으로 지워 서보가 단일
 * 0..4095 창으로 클램프하지 않게 해야 한다. ServoBus_Init() 이 서보마다
 * 한 번 수행한다. 아래 sts3215_angle_rad_to_ticks_near() 는 목표각의 대표값
 * (base + k*4096) 중 마지막 명령 틱값에 수치적으로 가장 가까운 것을 고른다.
 * swerve_kinematics.h 가 라디안 영역에서 이미 쓰는 "직전 상태로부터 최단경로"
 * 패턴을 틱 변환 계층에서 한 번 더 적용하는 것이다. 서보가 대신 해주지 않으니까.
 */

#ifndef STS3215_PROTOCOL_H
#define STS3215_PROTOCOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 컨트롤 테이블 주소 (SMS/STS 계열, SMS_STS.h 출처) ---- */
#define STS3215_REG_ID                  5
#define STS3215_REG_MIN_ANGLE_LIMIT_L   9   /* 2바이트. 0 이면 최소 클램프 해제 */
#define STS3215_REG_MAX_ANGLE_LIMIT_L   11  /* 2바이트. 0 이면 최대 클램프 해제 (위 경계 문제 설명 참고) */
#define STS3215_REG_TORQUE_ENABLE       40
#define STS3215_REG_ACC                 41
#define STS3215_REG_GOAL_POSITION_L     42
#define STS3215_REG_GOAL_POSITION_H     43
#define STS3215_REG_GOAL_TIME_L         44
#define STS3215_REG_GOAL_SPEED_L        46
#define STS3215_REG_PRESENT_POSITION_L  56
#define STS3215_REG_PRESENT_SPEED_L     58

#define STS3215_INST_PING  0x01
#define STS3215_INST_READ  0x02
#define STS3215_INST_WRITE 0x03

#define STS3215_TICKS_PER_REV 4096
#define STS3215_BROADCAST_ID  0xFEu

/* --- 패킷 조립기 ------------------------------------------------------
 * 각 함수는 out[] 에 쓴 바이트 수를 돌려주며, out_size 가 부족하면 0. */

/* 범용 "mem_addr 부터 data_len 바이트 쓰기" 요청. */
int sts3215_build_write_packet(uint8_t id, uint8_t mem_addr,
                                const uint8_t *data, uint8_t data_len,
                                uint8_t *out, int out_size);

/* `position`(원시 틱, 아래 각도 헬퍼 참고) 으로 `speed`(틱/s, 서보 고유 단위)
 * 와 `acc`(가속도, 0 이 가장 빠름) 로 이동시킨다.
 * SMS_STS::WritePosEx 와 정확히 동일하다 (레지스터 41..47, ACC..GOAL_SPEED_H
 * 7바이트를 쓴다. 시간 지정 이동이 아니라 속도로 구동하므로 goal_time 은 0).
 * 먼저 토크를 켜지 않으면(아래) 아무 일도 일어나지 않는다. */
int sts3215_build_write_pos_packet(uint8_t id, int16_t position,
                                    uint16_t speed, uint8_t acc,
                                    uint8_t *out, int out_size);

/* 토크 on/off. WritePos 가 실제로 움직이려면 시작할 때 서보마다 한 번
 * 호출해야 한다. */
int sts3215_build_torque_enable_packet(uint8_t id, uint8_t enable,
                                        uint8_t *out, int out_size);

/* 생존 확인 / 버스 스캔. 정상 서보는 자기 ID 로 응답한다. */
int sts3215_build_ping_packet(uint8_t id, uint8_t *out, int out_size);

/* --- 각도 <-> 원시 틱 변환 ---------------------------------------- */

/* angle_rad 를 0..4095 한 바퀴 안으로 접는다. 이전 위치를 기억하지 않는다.
 * 단발/첫 이동에는 괜찮지만, 같은 서보에 반복 명령할 때는 아래
 * sts3215_angle_rad_to_ticks_near() 를 쓰는 게 낫다. 그래야 0/4095 이음매
 * 근처에서 연속 이동이 먼 쪽으로 돌지 않는다 (파일 상단 설명 참고). */
int16_t sts3215_angle_rad_to_ticks(float angle_rad);

/* 같은 변환이지만, angle_rad 의 대표값들(base_ticks + k*4096, k 는 임의 정수)
 * 중 last_ticks 에 가장 가까운 것을 고른다.
 * 서보의 최소/최대 각도제한 레지스터가 0 으로 지워져 있어야 한다
 * (ServoBus_Init() 이 처리). 안 그러면 0..4095 를 살짝 벗어난 목표를 서보가
 * 도로 클램프한다. last_ticks 에는 이 서보에 **마지막으로 명령한** 틱값을
 * 넘긴다 (아직 아무 명령도 안 보냈다면 중앙값 2048 이 무난한 초기값). */
int16_t sts3215_angle_rad_to_ticks_near(float angle_rad, int16_t last_ticks);

/* --- 응답 파싱 --------------------------------------------------------
 * 바이트 단위 상태기계 (인터럽트 안전, 쓰레기/CRC 오류 시 재동기).
 * teleop_protocol.h 의 teleop_parser_feed_byte() 와 같은 방식. */
#define STS3215_MAX_REPLY_PARAMS 8

typedef struct {
    uint8_t id;
    uint8_t status;      /* 서보 에러/상태 바이트. 0 이면 정상 */
    uint8_t param_len;
    uint8_t params[STS3215_MAX_REPLY_PARAMS];
} sts3215_reply_t;

typedef enum {
    STS3215_RSTATE_SYNC0 = 0,
    STS3215_RSTATE_SYNC1,
    STS3215_RSTATE_ID,
    STS3215_RSTATE_LEN,
    STS3215_RSTATE_STATUS,
    STS3215_RSTATE_PARAMS,
    STS3215_RSTATE_CRC,
} sts3215_reply_pstate_t;

typedef struct {
    sts3215_reply_pstate_t state;
    uint8_t  id;
    uint8_t  len;         /* 와이어 LEN 필드: param_len + 2 */
    uint8_t  status;
    uint8_t  param_len;
    uint8_t  param_idx;
    uint8_t  params[STS3215_MAX_REPLY_PARAMS];
    uint32_t crc_error_count;
    uint32_t framing_error_count;
} sts3215_reply_parser_t;

void sts3215_reply_parser_init(sts3215_reply_parser_t *p);

/* 이 바이트로 CRC 검증까지 통과한 응답 프레임이 완성되면 *out 에 쓰고 1 을
 * 돌려준다. 그 외에는 0. */
int sts3215_reply_parser_feed_byte(sts3215_reply_parser_t *p, uint8_t byte,
                                    sts3215_reply_t *out);

#ifdef __cplusplus
}
#endif

#endif /* STS3215_PROTOCOL_H */

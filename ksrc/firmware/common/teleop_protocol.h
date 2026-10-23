/*
 * teleop_protocol.h
 *
 * 라즈베리파이 <-> Nucleo 시리얼 링크: 프레이밍, CRC, 메시지 인코딩/디코딩.
 * 순수 C 라 HAL 의존성이 없다. 바이트 파서는 실제 펌웨어에서 UART RX
 * 인터럽트/DMA 콜백이 한 바이트씩 넣어주도록 설계됐지만, 먼저 gcc 로
 * 호스트에서 완전히 테스트할 수 있다.
 *
 * 와이어 포맷 (페이로드 float 은 리틀엔디언):
 *
 *   바이트 0   : SYNC0  = 0xAA
 *   바이트 1   : SYNC1  = 0x55
 *   바이트 2   : TYPE   (teleop_msg_type_t 참고)
 *   바이트 3   : LEN    (페이로드 바이트 수)
 *   바이트 4.. : PAYLOAD (LEN 바이트)
 *   마지막     : {TYPE, LEN, PAYLOAD...} 에 대한 CRC8 (sync 바이트는 제외)
 *
 * CRC8: poly 0x07, init 0x00, reflect 없음, 최종 xor 없음
 * (teleop_protocol.c 의 crc8() 참고). pi/common/nucleo_link.py 의 파이썬
 * 인코더와 **비트 단위로 일치**해야 하며, 이를 검사하는 교차언어 왕복
 * 테스트가 있다.
 *
 * TELEOP_MSG_VELOCITY_CMD 페이로드(12바이트): vx, vy, omega 를 float32 로.
 * 차체 좌표계 규약은 swerve_kinematics.h 와 동일 (x=전진, y=좌측, omega=+반시계).
 *
 * 실제 펌웨어 연동 스케치:
 *   - UART RX 인터럽트: 바이트마다 teleop_parser_feed_byte() 호출. 반환이 1 이고
 *     frame.msg_type == TELEOP_MSG_VELOCITY_CMD 이면 teleop_decode_velocity_cmd()
 *     로 디코딩해 "최신 명령" 으로 저장하고 last_valid_frame_ms = HAL_GetTick()
 *     을 기록한다.
 *   - 메인 제어 루프: swerve_ik_compute() 호출 전에
 *     teleop_link_is_stale(HAL_GetTick(), last_valid_frame_ms, timeout) 를 확인.
 *     stale 이면 마지막 수신 명령과 무관하게 vx=vy=omega=0 을 강제한다
 *     (링크 끊김 시 안전 정지).
 */

#ifndef TELEOP_PROTOCOL_H
#define TELEOP_PROTOCOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TELEOP_SYNC0 0xAAu
#define TELEOP_SYNC1 0x55u
#define TELEOP_MAX_PAYLOAD 16

typedef enum {
    TELEOP_MSG_VELOCITY_CMD = 0x01, /* 파이 -> Nucleo: vx, vy, omega (12바이트) */
} teleop_msg_type_t;

typedef struct {
    uint8_t msg_type;
    uint8_t len;
    uint8_t payload[TELEOP_MAX_PAYLOAD];
} teleop_frame_t;

typedef struct {
    float vx;    /* m/s, +가 전진 */
    float vy;    /* m/s, +가 좌측 */
    float omega; /* rad/s, +가 반시계 */
} teleop_velocity_cmd_t;

/* --- CRC ------------------------------------------------------------ */
uint8_t teleop_crc8(const uint8_t *data, int len);

/* --- 인코딩 ----------------------------------------------------------
 * 완성된 프레임을 buf 에 쓴다. 쓴 바이트 수를 돌려주며(성공 시 항상
 * TELEOP_VELOCITY_CMD_FRAME_LEN), buf_size 가 부족하면 0 을 돌려준다. */
#define TELEOP_VELOCITY_CMD_FRAME_LEN 17
int teleop_encode_velocity_cmd(float vx, float vy, float omega,
                                uint8_t *buf, int buf_size);

/* --- 파싱 ------------------------------------------------------------
 * 바이트 단위 상태기계. 인터럽트에서 호출해도 안전하다: 동적 할당이 없고,
 * 쓰레기 데이터나 CRC 실패 시 외부에서 리셋해주지 않아도 스스로 재동기한다. */
typedef enum {
    TELEOP_PSTATE_SYNC0 = 0,
    TELEOP_PSTATE_SYNC1,
    TELEOP_PSTATE_TYPE,
    TELEOP_PSTATE_LEN,
    TELEOP_PSTATE_PAYLOAD,
    TELEOP_PSTATE_CRC,
} teleop_parser_state_t;

typedef struct {
    teleop_parser_state_t state;
    uint8_t  msg_type;
    uint8_t  len;
    uint8_t  payload[TELEOP_MAX_PAYLOAD];
    uint8_t  payload_idx;
    uint32_t crc_error_count;   /* 진단용: CRC 불일치로 버린 프레임 수 */
    uint32_t framing_error_count; /* 진단용: sync/len 이상으로 버린 프레임 수 */
} teleop_parser_t;

void teleop_parser_init(teleop_parser_t *p);

/* 수신 바이트 하나를 넣는다. 이 바이트로 CRC 검증까지 통과한 프레임이
 * 완성되면 *out_frame 에 쓰고 1 을 돌려준다. 그 외에는 0
 * (아직 프레임 중간이거나, 이 바이트 때문에 재동기가 일어난 경우). */
int teleop_parser_feed_byte(teleop_parser_t *p, uint8_t byte,
                             teleop_frame_t *out_frame);

/* TELEOP_MSG_VELOCITY_CMD 프레임의 페이로드를 디코딩한다. 성공하면 1,
 * frame.msg_type/len 이 안 맞으면 0. */
int teleop_decode_velocity_cmd(const teleop_frame_t *frame,
                                teleop_velocity_cmd_t *out);

/* --- 링크 워치독 ------------------------------------------------------
 * 밀리초 카운터 롤오버에 안전하다: now_ms 가 last_valid_frame_ms 를 지나
 * 한 바퀴 돌았어도, 실제 경과시간이 약 24일(2^32 ms) 미만이면 정확히 동작한다. */
int teleop_link_is_stale(uint32_t now_ms, uint32_t last_valid_frame_ms,
                          uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* TELEOP_PROTOCOL_H */

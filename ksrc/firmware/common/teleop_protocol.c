/* 파이<->Nucleo 텔레옵 프레이밍/CRC/인코딩 구현. 와이어 포맷은 teleop_protocol.h 참고. */
#include "teleop_protocol.h"
#include <string.h>

/* IEEE-754 단정밀도를 가정한다 (x86 호스트와 Cortex-M4F 타깃 둘 다 해당).
 * 비트 재해석은 union 이 아니라 memcpy 로 한다. strict-aliasing 미정의
 * 동작을 피하기 위해서다. */
static void put_f32_le(float v, uint8_t *out)
{
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    out[0] = (uint8_t)(u & 0xFFu);
    out[1] = (uint8_t)((u >> 8) & 0xFFu);
    out[2] = (uint8_t)((u >> 16) & 0xFFu);
    out[3] = (uint8_t)((u >> 24) & 0xFFu);
}

static float get_f32_le(const uint8_t *in)
{
    uint32_t u = (uint32_t)in[0]
               | ((uint32_t)in[1] << 8)
               | ((uint32_t)in[2] << 16)
               | ((uint32_t)in[3] << 24);
    float v;
    memcpy(&v, &u, sizeof(v));
    return v;
}

uint8_t teleop_crc8(const uint8_t *data, int len)
{
    uint8_t crc = 0x00u;
    int i, b;
    for (i = 0; i < len; i++) {
        crc ^= data[i];
        for (b = 0; b < 8; b++) {
            if (crc & 0x80u) {
                crc = (uint8_t)((crc << 1) ^ 0x07u);
            } else {
                crc = (uint8_t)(crc << 1);
            }
        }
    }
    return crc;
}

int teleop_encode_velocity_cmd(float vx, float vy, float omega,
                                uint8_t *buf, int buf_size)
{
    uint8_t payload[12];

    if (buf_size < TELEOP_VELOCITY_CMD_FRAME_LEN) {
        return 0;
    }

    put_f32_le(vx, &payload[0]);
    put_f32_le(vy, &payload[4]);
    put_f32_le(omega, &payload[8]);

    buf[0] = TELEOP_SYNC0;
    buf[1] = TELEOP_SYNC1;
    buf[2] = (uint8_t)TELEOP_MSG_VELOCITY_CMD;
    buf[3] = (uint8_t)sizeof(payload);
    memcpy(&buf[4], payload, sizeof(payload));
    /* CRC 범위는 TYPE, LEN, PAYLOAD. 즉 sync 바이트 이후 전부 */
    buf[4 + sizeof(payload)] = teleop_crc8(&buf[2], 2 + (int)sizeof(payload));

    return TELEOP_VELOCITY_CMD_FRAME_LEN;
}

void teleop_parser_init(teleop_parser_t *p)
{
    p->state = TELEOP_PSTATE_SYNC0;
    p->msg_type = 0;
    p->len = 0;
    p->payload_idx = 0;
    p->crc_error_count = 0;
    p->framing_error_count = 0;
}

int teleop_parser_feed_byte(teleop_parser_t *p, uint8_t byte,
                             teleop_frame_t *out_frame)
{
    switch (p->state) {
    case TELEOP_PSTATE_SYNC0:
        if (byte == TELEOP_SYNC0) {
            p->state = TELEOP_PSTATE_SYNC1;
        }
        /* 아니면 SYNC0 에 머물며 시작 바이트를 계속 찾는다 */
        return 0;

    case TELEOP_PSTATE_SYNC1:
        if (byte == TELEOP_SYNC1) {
            p->state = TELEOP_PSTATE_TYPE;
        } else if (byte == TELEOP_SYNC0) {
            /* 여기 머문다: 예를 들어 0xAA 0xAA 0x55 도 올바르게 재동기된다 */
            p->state = TELEOP_PSTATE_SYNC1;
        } else {
            p->framing_error_count++;
            p->state = TELEOP_PSTATE_SYNC0;
        }
        return 0;

    case TELEOP_PSTATE_TYPE:
        p->msg_type = byte;
        p->state = TELEOP_PSTATE_LEN;
        return 0;

    case TELEOP_PSTATE_LEN:
        if (byte > TELEOP_MAX_PAYLOAD) {
            p->framing_error_count++;
            p->state = TELEOP_PSTATE_SYNC0;
            return 0;
        }
        p->len = byte;
        p->payload_idx = 0;
        p->state = (p->len == 0) ? TELEOP_PSTATE_CRC : TELEOP_PSTATE_PAYLOAD;
        return 0;

    case TELEOP_PSTATE_PAYLOAD:
        p->payload[p->payload_idx++] = byte;
        if (p->payload_idx >= p->len) {
            p->state = TELEOP_PSTATE_CRC;
        }
        return 0;

    case TELEOP_PSTATE_CRC:
    default:
        {
            uint8_t crc_buf[2 + TELEOP_MAX_PAYLOAD];
            uint8_t expected_crc;

            crc_buf[0] = p->msg_type;
            crc_buf[1] = p->len;
            memcpy(&crc_buf[2], p->payload, p->len);
            expected_crc = teleop_crc8(crc_buf, 2 + p->len);

            p->state = TELEOP_PSTATE_SYNC0; /* always resync after CRC byte */

            if (byte != expected_crc) {
                p->crc_error_count++;
                return 0;
            }

            out_frame->msg_type = p->msg_type;
            out_frame->len = p->len;
            memcpy(out_frame->payload, p->payload, p->len);
            return 1;
        }
    }
}

int teleop_decode_velocity_cmd(const teleop_frame_t *frame,
                                teleop_velocity_cmd_t *out)
{
    if (frame->msg_type != (uint8_t)TELEOP_MSG_VELOCITY_CMD || frame->len != 12) {
        return 0;
    }
    out->vx = get_f32_le(&frame->payload[0]);
    out->vy = get_f32_le(&frame->payload[4]);
    out->omega = get_f32_le(&frame->payload[8]);
    return 1;
}

int teleop_link_is_stale(uint32_t now_ms, uint32_t last_valid_frame_ms,
                          uint32_t timeout_ms)
{
    /* 부호 없는 뺄셈은 uint32 롤오버를 건너서도 올바르게 동작한다.
     * 실제 경과시간이 약 24일 미만이면 된다 */
    uint32_t elapsed = now_ms - last_valid_frame_ms;
    return elapsed > timeout_ms;
}

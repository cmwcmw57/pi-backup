/* STS3215 서보 버스 패킷 조립/응답 파싱 구현. 프로토콜 설명은 sts3215_protocol.h 참고. */
#include "sts3215_protocol.h"
#include <math.h>
#include <string.h>

#define STS3215_PI 3.14159265358979323846f

/* 체크섬 = ~(ID 부터 마지막 파라미터까지 모든 바이트의 합) & 0xFF */
static uint8_t checksum(const uint8_t *body, int len)
{
    uint8_t sum = 0;
    int i;
    for (i = 0; i < len; i++) {
        sum = (uint8_t)(sum + body[i]);
    }
    return (uint8_t)~sum;
}

/* End=0(기본값) 바이트 순서: 하위 바이트 먼저 */
static void put_u16_le(uint16_t v, uint8_t *out)
{
    out[0] = (uint8_t)(v & 0xFFu);
    out[1] = (uint8_t)((v >> 8) & 0xFFu);
}

int sts3215_build_write_packet(uint8_t id, uint8_t mem_addr,
                                const uint8_t *data, uint8_t data_len,
                                uint8_t *out, int out_size)
{
    uint8_t msg_len = (uint8_t)(2 + data_len + 1); /* INST + MEM_ADDR + data + CRC, minus CRC itself counted separately below like the reference lib */
    int frame_len = 7 + data_len; /* 2 sync + id + len + inst + memaddr + data + crc */
    uint8_t body[3 + 1 + 255]; /* id,len,inst + memaddr + data (data_len <= 255) */
    int body_len;

    if (out_size < frame_len) {
        return 0;
    }

    body[0] = id;
    body[1] = msg_len;
    body[2] = (uint8_t)STS3215_INST_WRITE;
    body[3] = mem_addr;
    memcpy(&body[4], data, data_len);
    body_len = 4 + data_len;

    out[0] = 0xFF;
    out[1] = 0xFF;
    memcpy(&out[2], body, (size_t)body_len);
    out[2 + body_len] = checksum(body, body_len);

    return frame_len;
}

int sts3215_build_write_pos_packet(uint8_t id, int16_t position,
                                    uint16_t speed, uint8_t acc,
                                    uint8_t *out, int out_size)
{
    uint8_t payload[7];
    payload[0] = acc;
    put_u16_le((uint16_t)position, &payload[1]); /* Goal_Position_L/H */
    put_u16_le(0, &payload[3]);                  /* Goal_Time_L/H = 0 (unused) */
    put_u16_le(speed, &payload[5]);              /* Goal_Speed_L/H */

    return sts3215_build_write_packet(id, STS3215_REG_ACC, payload, sizeof payload,
                                       out, out_size);
}

int sts3215_build_torque_enable_packet(uint8_t id, uint8_t enable,
                                        uint8_t *out, int out_size)
{
    uint8_t data = enable;
    return sts3215_build_write_packet(id, STS3215_REG_TORQUE_ENABLE, &data, 1,
                                       out, out_size);
}

int sts3215_build_ping_packet(uint8_t id, uint8_t *out, int out_size)
{
    uint8_t body[3];
    if (out_size < 6) {
        return 0;
    }
    body[0] = id;
    body[1] = 2; /* INST + CRC only, no mem_addr/params for PING */
    body[2] = (uint8_t)STS3215_INST_PING;

    out[0] = 0xFF;
    out[1] = 0xFF;
    memcpy(&out[2], body, sizeof body);
    out[5] = checksum(body, sizeof body);
    return 6;
}

int16_t sts3215_angle_rad_to_ticks(float angle_rad)
{
    float turns = angle_rad / (2.0f * STS3215_PI);
    float frac = turns - floorf(turns); /* wraps into [0, 1) */
    int ticks = (int)(frac * (float)STS3215_TICKS_PER_REV + 0.5f);
    if (ticks >= STS3215_TICKS_PER_REV) {
        ticks -= STS3215_TICKS_PER_REV;
    }
    if (ticks < 0) {
        ticks = 0;
    }
    return (int16_t)ticks;
}

int16_t sts3215_angle_rad_to_ticks_near(float angle_rad, int16_t last_ticks)
{
    int16_t base_ticks = sts3215_angle_rad_to_ticks(angle_rad); /* canonical 0..4095 */
    int delta = (int)last_ticks - (int)base_ticks;
    /* base_ticks 에 더할 추가 회전수(정수). 결과가 last_ticks 에 최대한
     * 가까워지도록 고른다 */
    int k = (int)floorf((float)delta / (float)STS3215_TICKS_PER_REV + 0.5f);
    int ticks = (int)base_ticks + k * STS3215_TICKS_PER_REV;
    return (int16_t)ticks;
}

void sts3215_reply_parser_init(sts3215_reply_parser_t *p)
{
    p->state = STS3215_RSTATE_SYNC0;
    p->id = 0;
    p->len = 0;
    p->status = 0;
    p->param_len = 0;
    p->param_idx = 0;
    p->crc_error_count = 0;
    p->framing_error_count = 0;
}

int sts3215_reply_parser_feed_byte(sts3215_reply_parser_t *p, uint8_t byte,
                                    sts3215_reply_t *out)
{
    switch (p->state) {
    case STS3215_RSTATE_SYNC0:
        if (byte == 0xFF) {
            p->state = STS3215_RSTATE_SYNC1;
        }
        return 0;

    case STS3215_RSTATE_SYNC1:
        if (byte == 0xFF) {
            p->state = STS3215_RSTATE_ID;
        } else {
            p->framing_error_count++;
            p->state = STS3215_RSTATE_SYNC0;
        }
        return 0;

    case STS3215_RSTATE_ID:
        p->id = byte;
        p->state = STS3215_RSTATE_LEN;
        return 0;

    case STS3215_RSTATE_LEN:
        if (byte < 2 || (byte - 2) > STS3215_MAX_REPLY_PARAMS) {
            p->framing_error_count++;
            p->state = STS3215_RSTATE_SYNC0;
            return 0;
        }
        p->len = byte;
        p->param_len = (uint8_t)(byte - 2);
        p->param_idx = 0;
        p->state = STS3215_RSTATE_STATUS;
        return 0;

    case STS3215_RSTATE_STATUS:
        p->status = byte;
        p->state = (p->param_len == 0) ? STS3215_RSTATE_CRC : STS3215_RSTATE_PARAMS;
        return 0;

    case STS3215_RSTATE_PARAMS:
        p->params[p->param_idx++] = byte;
        if (p->param_idx >= p->param_len) {
            p->state = STS3215_RSTATE_CRC;
        }
        return 0;

    case STS3215_RSTATE_CRC:
    default:
        {
            uint8_t body[3 + STS3215_MAX_REPLY_PARAMS];
            uint8_t expected;

            body[0] = p->id;
            body[1] = p->len;
            body[2] = p->status;
            memcpy(&body[3], p->params, p->param_len);
            expected = checksum(body, 3 + p->param_len);

            p->state = STS3215_RSTATE_SYNC0;

            if (byte != expected) {
                p->crc_error_count++;
                return 0;
            }

            out->id = p->id;
            out->status = p->status;
            out->param_len = p->param_len;
            memcpy(out->params, p->params, p->param_len);
            return 1;
        }
    }
}

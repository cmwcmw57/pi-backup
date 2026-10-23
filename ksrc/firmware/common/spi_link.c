/* Pi <-> Nucleo SPI 프레임 구현. 레이아웃은 spi_link.h 참고. */
#include "spi_link.h"
#include <string.h>

#define CMD_CRC_OFS  18   /* CRC 는 [2]..[17]   (타입 0x00·0x01) */
#define RES_CRC_OFS  50   /* CRC 는 [2]..[49]   (타입 0x02) */
#define TLM_CRC_OFS  (SPI_LINK_FRAME_LEN - 2)   /* CRC 는 [2]..[141] */

uint16_t spi_link_crc16(const uint8_t *data, uint32_t len)
{
    uint16_t crc = 0xFFFF;
    uint32_t i;
    int b;
    for (i = 0; i < len; i++) {
        crc ^= (uint16_t)((uint16_t)data[i] << 8);
        for (b = 0; b < 8; b++) {
            if (crc & 0x8000u) {
                crc = (uint16_t)((crc << 1) ^ 0x1021u);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }
    return crc;
}

/* --- 리틀엔디안 헬퍼 (호스트 엔디안과 무관하게 동작) --- */

static void put_u16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)(v >> 8);
}

static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)(v >> 24);
}

static void put_f32(uint8_t *p, float f)
{
    uint32_t v;
    memcpy(&v, &f, sizeof v);
    put_u32(p, v);
}

static uint16_t get_u16(const uint8_t *p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float get_f32(const uint8_t *p)
{
    uint32_t v = get_u32(p);
    float f;
    memcpy(&f, &v, sizeof f);
    return f;
}

/* --- 명령 프레임 --- */

void spi_link_pack_cmd(const spi_link_cmd_t *cmd, uint8_t frame[SPI_LINK_FRAME_LEN])
{
    memset(frame, 0, SPI_LINK_FRAME_LEN);
    frame[0] = SPI_LINK_SYNC0;
    frame[1] = SPI_LINK_SYNC1;
    frame[2] = SPI_LINK_VERSION;
    frame[3] = cmd->type;
    frame[4] = cmd->seq;
    frame[5] = 0;
    put_f32(&frame[6], cmd->vx);
    put_f32(&frame[10], cmd->vy);
    put_f32(&frame[14], cmd->omega);
    int ofs = CMD_CRC_OFS;
    if (cmd->type == SPI_LINK_TYPE_VEL_RES) {
        for (int i = 0; i < 4; i++) {
            put_f32(&frame[18 + 4 * i], cmd->d_steer[i]);
            put_f32(&frame[34 + 4 * i], cmd->wheel_gain[i]);
        }
        ofs = RES_CRC_OFS;
    }
    put_u16(&frame[ofs], spi_link_crc16(&frame[2], (uint32_t)ofs - 2));
}

spi_link_status_t spi_link_parse_cmd(const uint8_t frame[SPI_LINK_FRAME_LEN],
                                     spi_link_cmd_t *out)
{
    if (frame[0] != SPI_LINK_SYNC0 || frame[1] != SPI_LINK_SYNC1) return SPI_LINK_ERR_SYNC;
    if (frame[2] != SPI_LINK_VERSION) return SPI_LINK_ERR_VERSION;
    /* 타입 바이트도 CRC 안이라, 깨진 타입으로 고른 CRC 위치는 거의 확실히 CRC 에서 걸린다 */
    const int ofs = (frame[3] == SPI_LINK_TYPE_VEL_RES) ? RES_CRC_OFS : CMD_CRC_OFS;
    if (get_u16(&frame[ofs]) != spi_link_crc16(&frame[2], (uint32_t)ofs - 2)) {
        return SPI_LINK_ERR_CRC;
    }
    if (frame[3] != SPI_LINK_TYPE_POLL && frame[3] != SPI_LINK_TYPE_VEL_CMD &&
        frame[3] != SPI_LINK_TYPE_VEL_RES) {
        return SPI_LINK_ERR_TYPE;
    }
    out->type = frame[3];
    out->seq = frame[4];
    out->vx = get_f32(&frame[6]);
    out->vy = get_f32(&frame[10]);
    out->omega = get_f32(&frame[14]);
    for (int i = 0; i < 4; i++) {
        out->d_steer[i] = (frame[3] == SPI_LINK_TYPE_VEL_RES) ? get_f32(&frame[18 + 4 * i]) : 0.0f;
        out->wheel_gain[i] = (frame[3] == SPI_LINK_TYPE_VEL_RES) ? get_f32(&frame[34 + 4 * i]) : 1.0f;
    }
    return SPI_LINK_OK;
}

/* --- 텔레메트리 프레임 --- */

void spi_link_pack_telemetry(const spi_link_telemetry_t *t,
                             uint8_t frame[SPI_LINK_FRAME_LEN])
{
    int i;
    memset(frame, 0, SPI_LINK_FRAME_LEN);
    frame[0] = SPI_LINK_SYNC0;
    frame[1] = SPI_LINK_SYNC1;
    frame[2] = SPI_LINK_VERSION;
    frame[3] = SPI_LINK_TYPE_TELEMETRY;
    frame[4] = t->seq_echo;
    frame[5] = t->tick;
    put_u16(&frame[6], t->flags);
    put_u32(&frame[8], t->t_ms);
    put_f32(&frame[12], t->cmd_vx);
    put_f32(&frame[16], t->cmd_vy);
    put_f32(&frame[20], t->cmd_omega);
    for (i = 0; i < 4; i++) {
        put_u32(&frame[24 + 4 * i], (uint32_t)t->enc_ticks[i]);
        put_f32(&frame[40 + 4 * i], t->wheel_mps[i]);
        put_f32(&frame[56 + 4 * i], t->steer_rad[i]);
        put_u16(&frame[72 + 2 * i], (uint16_t)t->duty[i]);
    }
    for (i = 0; i < 6; i++) {
        put_u16(&frame[80 + 2 * i], (uint16_t)t->imu[i]);
    }
    put_u16(&frame[92], t->rx_err);
    for (i = 0; i < 4; i++) {
        put_f32(&frame[94 + 4 * i], t->steer_meas_rad[i]);
        put_u16(&frame[110 + 2 * i], (uint16_t)t->steer_speed[i]);
        put_u16(&frame[118 + 2 * i], (uint16_t)t->steer_load[i]);
        frame[126 + i] = t->steer_volt[i];
        frame[130 + i] = t->steer_temp[i];
        frame[134 + i] = t->steer_status[i];
    }
    frame[138] = t->servo_valid;
    frame[139] = t->imu_n;
    put_u16(&frame[140], (uint16_t)t->imu_mean_read_offset_us);
    put_u16(&frame[TLM_CRC_OFS], spi_link_crc16(&frame[2], TLM_CRC_OFS - 2));
}

spi_link_status_t spi_link_parse_telemetry(const uint8_t frame[SPI_LINK_FRAME_LEN],
                                           spi_link_telemetry_t *out)
{
    int i;
    if (frame[0] != SPI_LINK_SYNC0 || frame[1] != SPI_LINK_SYNC1) return SPI_LINK_ERR_SYNC;
    if (frame[2] != SPI_LINK_VERSION) return SPI_LINK_ERR_VERSION;
    if (get_u16(&frame[TLM_CRC_OFS]) != spi_link_crc16(&frame[2], TLM_CRC_OFS - 2)) {
        return SPI_LINK_ERR_CRC;
    }
    if (frame[3] != SPI_LINK_TYPE_TELEMETRY) return SPI_LINK_ERR_TYPE;
    out->seq_echo = frame[4];
    out->tick = frame[5];
    out->flags = get_u16(&frame[6]);
    out->t_ms = get_u32(&frame[8]);
    out->cmd_vx = get_f32(&frame[12]);
    out->cmd_vy = get_f32(&frame[16]);
    out->cmd_omega = get_f32(&frame[20]);
    for (i = 0; i < 4; i++) {
        out->enc_ticks[i] = (int32_t)get_u32(&frame[24 + 4 * i]);
        out->wheel_mps[i] = get_f32(&frame[40 + 4 * i]);
        out->steer_rad[i] = get_f32(&frame[56 + 4 * i]);
        out->duty[i] = (int16_t)get_u16(&frame[72 + 2 * i]);
    }
    for (i = 0; i < 6; i++) {
        out->imu[i] = (int16_t)get_u16(&frame[80 + 2 * i]);
    }
    out->rx_err = get_u16(&frame[92]);
    for (i = 0; i < 4; i++) {
        out->steer_meas_rad[i] = get_f32(&frame[94 + 4 * i]);
        out->steer_speed[i] = (int16_t)get_u16(&frame[110 + 2 * i]);
        out->steer_load[i] = (int16_t)get_u16(&frame[118 + 2 * i]);
        out->steer_volt[i] = frame[126 + i];
        out->steer_temp[i] = frame[130 + i];
        out->steer_status[i] = frame[134 + i];
    }
    out->servo_valid = frame[138];
    out->imu_n = frame[139];
    out->imu_mean_read_offset_us = (int16_t)get_u16(&frame[140]);
    return SPI_LINK_OK;
}

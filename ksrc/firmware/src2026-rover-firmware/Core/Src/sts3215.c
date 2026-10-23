/**
 * @file    sts3215.c
 * @brief   Feetech STS3215 시리얼 버스 서보 프로토콜 드라이버 구현 (USART3 레지스터 직접 제어)
 *
 * 통신 흐름:
 *   STM32 USART3(TX) → 어댑터 보드(반이중 변환) → STS3215 서보 버스
 *   STM32 USART3(RX) ← 어댑터 보드 ← STS3215 응답
 */

#include "sts3215.h"
#include <string.h>
#include <stdio.h>

/* HAL_GetTick 사용을 위해 선언 */
extern uint32_t HAL_GetTick(void);

/* ========================== 내부 상수 ========================== */
#define STS_TX_TIMEOUT_MS   10   /**< 송신 타임아웃 (ms) */
/** 수신 타임아웃 (ms). 서보 응답은 반환 지연 500us + 수십 us 안에 온다.
 *  15ms 였을 때는 서보 전원이 꺼진 벤치에서 4개 x 15ms = 60ms 동안 50Hz 제어
 *  루프가 멈췄다. HAL_GetTick 이 1ms 단위라 실제 대기는 2~3ms. */
#define STS_RX_TIMEOUT_MS   3
#define STS_MAX_PACKET_LEN  64   /**< 최대 패킷 크기 */

/* ========================== 레지스터 기반 UART 헬퍼 ========================== */

static void sts_uart_send_bytes(const uint8_t *data, uint16_t len)
{
    for (uint16_t i = 0; i < len; i++)
    {
        while (!(USART3->SR & USART_SR_TXE));
        USART3->DR = data[i];
    }
    while (!(USART3->SR & USART_SR_TC));
}

static uint8_t sts_uart_recv_byte(uint8_t *byte, uint32_t timeout_ms)
{
    uint32_t start = HAL_GetTick();
    while (!(USART3->SR & USART_SR_RXNE))
    {
        if (HAL_GetTick() - start >= timeout_ms)
        {
            return 0; /* 타임아웃 */
        }
    }
    /* 오버런/프레이밍/노이즈 오류: 인터럽트 등으로 바이트를 놓쳤다. 이 패킷은
     * 이미 깨졌으므로 남은 바이트를 타임아웃까지 기다리지 말고 바로 실패한다.
     * SR 을 읽은 뒤 DR 을 읽으면 ORE/FE/NE 가 지워진다. */
    uint32_t sr = USART3->SR;
    *byte = (uint8_t)(USART3->DR & 0xFF);
    if (sr & (USART_SR_ORE | USART_SR_FE | USART_SR_NE))
    {
        return 0;
    }
    return 1; /* 수신 성공 */
}

/**
 * @brief  RX 버퍼에 남아있는 쓰레기 데이터를 모두 비움
 */
static void sts_flush_rx(void)
{
    volatile uint8_t dummy;
    /* RXNE 뿐 아니라 남은 ORE 도 지운다 (SR 읽기 -> DR 읽기) */
    while (USART3->SR & (USART_SR_RXNE | USART_SR_ORE))
    {
        dummy = (uint8_t)(USART3->DR & 0xFF);
        (void)dummy;
    }
}



/**
 * @brief  패킷을 빌드하여 UART로 송신
 */
static uint8_t sts_build_and_send(uint8_t id, uint8_t inst,
                                   uint8_t *params, uint8_t param_len,
                                   uint8_t *tx_buf)
{
    uint8_t pkt_len = param_len + 6;
    uint8_t len_field = param_len + 2;

    tx_buf[0] = STS_HEADER;
    tx_buf[1] = STS_HEADER;
    tx_buf[2] = id;
    tx_buf[3] = len_field;
    tx_buf[4] = inst;

    if (params != NULL && param_len > 0)
    {
        memcpy(&tx_buf[5], params, param_len);
    }

    uint8_t sum = 0;
    for (uint8_t i = 2; i < pkt_len - 1; i++)
    {
        sum += tx_buf[i];
    }
    tx_buf[pkt_len - 1] = ~sum;

    sts_flush_rx();
    sts_uart_send_bytes(tx_buf, pkt_len);

    return pkt_len;
}

/**
 * @brief  서보 응답 패킷 수신 및 파싱
 */
static STS_Status sts_receive_response(uint8_t expected_id,
                                        uint8_t *out_params,
                                        uint8_t *out_param_len)
{
    uint8_t header[4];  /* 0xFF 0xFF ID LEN */

    for (int i = 0; i < 4; i++)
    {
        if (!sts_uart_recv_byte(&header[i], STS_RX_TIMEOUT_MS))
        {
            return STS_TIMEOUT;
        }
    }

    if (header[0] != STS_HEADER || header[1] != STS_HEADER)
    {
        return STS_RX_ERROR;
    }

    uint8_t id  = header[2];
    uint8_t len = header[3];

    if (id != expected_id)
    {
        return STS_ID_MISMATCH;
    }

    if (len < 2 || len > STS_MAX_PACKET_LEN)
    {
        return STS_RX_ERROR;
    }

    uint8_t remaining[STS_MAX_PACKET_LEN];
    for (int i = 0; i < len; i++)
    {
        if (!sts_uart_recv_byte(&remaining[i], STS_RX_TIMEOUT_MS))
        {
            return STS_TIMEOUT;
        }
    }

    uint8_t sum = id + len;
    for (uint8_t i = 0; i < len - 1; i++)
    {
        sum += remaining[i];
    }
    uint8_t calc_chk = ~sum;

    if (calc_chk != remaining[len - 1])
    {
        return STS_CHECKSUM_ERR;
    }

    uint8_t param_count = len - 2;

    if (out_params != NULL && param_count > 0)
    {
        memcpy(out_params, &remaining[1], param_count);
    }

    if (out_param_len != NULL)
    {
        *out_param_len = param_count;
    }

    return STS_OK;
}

/* ========================== 공개 API 구현 ========================== */

void sts_init(void)
{
    /* USART3 레지스터 제어 방식이므로 별도 핸들 저장이 필요 없음 */
}

STS_Status sts_ping(uint8_t id)
{
    uint8_t tx_buf[16];
    sts_build_and_send(id, STS_INST_PING, NULL, 0, tx_buf);
    sts_flush_rx();  /* 송신 후 메아리 플러시 */
    return sts_receive_response(id, NULL, NULL);
}

STS_Status sts_write(uint8_t id, uint8_t addr, uint8_t *data, uint8_t len)
{
    uint8_t tx_buf[STS_MAX_PACKET_LEN];
    uint8_t params[STS_MAX_PACKET_LEN];

    params[0] = addr;
    memcpy(&params[1], data, len);

    sts_build_and_send(id, STS_INST_WRITE, params, len + 1, tx_buf);
    sts_flush_rx();  /* 송신 후 메아리 플러시 */
    return sts_receive_response(id, NULL, NULL);
}

STS_Status sts_read(uint8_t id, uint8_t addr, uint8_t len, uint8_t *buf)
{
    uint8_t tx_buf[16];
    uint8_t params[2];

    params[0] = addr;
    params[1] = len;

    sts_build_and_send(id, STS_INST_READ, params, 2, tx_buf);
    sts_flush_rx();  /* 송신 후 메아리 플러시 */

    uint8_t param_len = 0;
    STS_Status ret = sts_receive_response(id, buf, &param_len);

    if (ret == STS_OK && param_len != len)
    {
        return STS_RX_ERROR;
    }

    return ret;
}

STS_Status sts_set_torque(uint8_t id, uint8_t on)
{
    uint8_t val = on ? 1 : 0;
    return sts_write(id, STS_REG_TORQUE_ENABLE, &val, 1);
}

STS_Status sts_set_acceleration(uint8_t id, uint8_t acc)
{
    return sts_write(id, STS_REG_ACCELERATION, &acc, 1);
}

STS_Status sts_sync_write_position(uint8_t *ids, uint16_t *positions,
                                    uint16_t *speeds, uint8_t count)
{
    (void)speeds;   /* 위치만 보낸다. 속도는 S 명령/joint_init 이 레지스터 0x2E 에 따로 쓴다 */
    uint8_t tx_buf[STS_MAX_PACKET_LEN];
    uint8_t data_per_servo = 2;
    uint8_t start_addr = STS_REG_GOAL_POSITION;

    uint8_t len_field = (data_per_servo + 1) * count + 4;
    uint8_t pkt_len = len_field + 4;

    if (pkt_len > STS_MAX_PACKET_LEN) return STS_RX_ERROR;

    tx_buf[0] = STS_HEADER;
    tx_buf[1] = STS_HEADER;
    tx_buf[2] = STS_BROADCAST_ID;
    tx_buf[3] = len_field;
    tx_buf[4] = STS_INST_SYNC_WRITE;
    tx_buf[5] = start_addr;
    tx_buf[6] = data_per_servo;

    uint8_t idx = 7;
    for (uint8_t i = 0; i < count; i++)
    {
        tx_buf[idx++] = ids[i];
        tx_buf[idx++] = (uint8_t)(positions[i] & 0xFF);
        tx_buf[idx++] = (uint8_t)((positions[i] >> 8) & 0xFF);
    }

    uint8_t sum = 0;
    for (uint8_t i = 2; i < pkt_len - 1; i++)
    {
        sum += tx_buf[i];
    }
    tx_buf[pkt_len - 1] = ~sum;

    sts_flush_rx();
    sts_uart_send_bytes(tx_buf, pkt_len);
    sts_flush_rx();  /* 송신 후 메아리 플러시 */

    return STS_OK;
}

int16_t sts_read_position(uint8_t id)
{
    uint8_t buf[2];
    STS_Status ret = sts_read(id, STS_REG_PRESENT_POSITION, 2, buf);
    if (ret != STS_OK)
    {
        return -1;
    }
    return (int16_t)(buf[0] | (buf[1] << 8));
}

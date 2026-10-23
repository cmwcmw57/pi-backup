/**
 * @file    pi_spi.c
 * @brief   Pi <-> Nucleo SPI2 슬레이브 링크 구현. 설명은 pi_spi.h 참고.
 */

#include "pi_spi.h"
#include "main.h"
#include <string.h>

/* SPI2 DMA 매핑 (RM0390 표 28): DMA1 Stream3 ch0 = SPI2_RX, Stream4 ch0 = SPI2_TX */
#define RX_STREAM  DMA1_Stream3
#define TX_STREAM  DMA1_Stream4

#define RX_FLAGS  (DMA_LIFCR_CTCIF3 | DMA_LIFCR_CHTIF3 | DMA_LIFCR_CTEIF3 | \
                   DMA_LIFCR_CDMEIF3 | DMA_LIFCR_CFEIF3)
#define TX_FLAGS  (DMA_HIFCR_CTCIF4 | DMA_HIFCR_CHTIF4 | DMA_HIFCR_CTEIF4 | \
                   DMA_HIFCR_CDMEIF4 | DMA_HIFCR_CFEIF4)

/* 삼중 버퍼: ISR 은 포인터만 바꾸고 memcpy 를 하지 않는다.
 * ISR 안의 144바이트 복사 두 번(바이트 루프 memcpy, 16MHz)은 ~130us 라, 그동안
 * 폴링으로 받는 USART3 서보 응답(1Mbps = 10us/바이트)이 오버런으로 깨졌다.
 *   RX: s_rx_dma (DMA 가 씀) / s_rx_ready (완성, 메인 대기) / s_rx_user (메인이 읽는 중)
 *   TX: s_tx_dma (DMA 가 읽음) / s_tx_pending (다음 전송용) / s_tx_user (메인이 채우는 중) */
static uint8_t s_buf_rx[3][SPI_LINK_FRAME_LEN];
static uint8_t s_buf_tx[3][SPI_LINK_FRAME_LEN];
static uint8_t *s_rx_dma   = s_buf_rx[0];
static uint8_t *s_rx_ready = s_buf_rx[1];
static uint8_t *s_rx_user  = s_buf_rx[2];
static uint8_t *s_tx_dma     = s_buf_tx[0];
static uint8_t *s_tx_pending = s_buf_tx[1];
static uint8_t *s_tx_user    = s_buf_tx[2];
static volatile uint8_t  s_rx_new = 0;
static volatile uint8_t  s_tx_new = 0;
static volatile uint32_t s_short = 0;
/* 배선 진단용 (SPI 명령) */
static volatile uint32_t s_cs_edges = 0;
static volatile uint32_t s_glitches = 0;
static volatile uint32_t s_full = 0;
static volatile uint32_t s_last_len = 0;
static volatile uint8_t  s_last_head[2] = {0, 0};

/* SPI2 를 리셋해 송신 버퍼에 남은 바이트까지 비운 뒤 슬레이브로 다시 설정한다.
 * SPE 만 내리면 DR 에 이미 올라간 이전 프레임의 바이트가 다음 전송 첫 바이트로
 * 나가 프레임이 한 칸 밀린다. */
static void spi2_reset_slave(void)
{
    RCC->APB1RSTR |= RCC_APB1RSTR_SPI2RST;
    RCC->APB1RSTR &= ~RCC_APB1RSTR_SPI2RST;
    /* 슬레이브, 모드 0, 8비트, MSB first, 소프트웨어 NSS (SSM=1, SSI=0 = 항상 선택).
     * 하드웨어 NSS 는 CS 선에 튀는 짧은 잡음(SCLK 옆선 크로스토크)에도 SCK 를 막아
     * 비트가 밀렸다. 프레임 경계는 EXTI 로 확인된 CS 상승 에지만 쓴다. */
    SPI2->CR1 = SPI_CR1_SSM;
    SPI2->CR2 = SPI_CR2_RXDMAEN | SPI_CR2_TXDMAEN;
}

static void dma_stop(DMA_Stream_TypeDef *s)
{
    s->CR &= ~DMA_SxCR_EN;
    for (uint32_t n = 0; (s->CR & DMA_SxCR_EN) && n < 10000U; n++) { }
}

static void arm(void)
{
    DMA1->LIFCR = RX_FLAGS;
    DMA1->HIFCR = TX_FLAGS;

    RX_STREAM->PAR  = (uint32_t)&SPI2->DR;
    RX_STREAM->M0AR = (uint32_t)s_rx_dma;
    RX_STREAM->NDTR = SPI_LINK_FRAME_LEN;
    RX_STREAM->CR   = DMA_SxCR_MINC | DMA_SxCR_PL_1;          /* ch0, 주변장치->메모리 */

    TX_STREAM->PAR  = (uint32_t)&SPI2->DR;
    TX_STREAM->M0AR = (uint32_t)s_tx_dma;
    TX_STREAM->NDTR = SPI_LINK_FRAME_LEN;
    TX_STREAM->CR   = DMA_SxCR_MINC | DMA_SxCR_DIR_0 | DMA_SxCR_PL_1; /* ch0, 메모리->주변장치 */

    RX_STREAM->CR |= DMA_SxCR_EN;
    TX_STREAM->CR |= DMA_SxCR_EN;
    SPI2->CR1 |= SPI_CR1_SPE;
}

/* CS 가 정말 High 인지 (잡음 스파이크는 ISR 진입 시점 ~1us 이면 이미 Low 로 돌아와 있다) */
static int cs_is_high(void)
{
    for (int i = 0; i < 4; i++)
    {
        if (!(GPIOB->IDR & GPIO_PIN_12)) return 0;
    }
    return 1;
}

/* CS 상승 에지: 한 프레임 전송이 끝났다.  포인터 교환만 한다 (수 us). */
static void on_cs_release(void)
{
    uint32_t received = SPI_LINK_FRAME_LEN - RX_STREAM->NDTR;

    s_cs_edges++;
    s_last_len = received;
    if (received >= 2U)
    {
        s_last_head[0] = s_rx_dma[0];
        s_last_head[1] = s_rx_dma[1];
    }

    dma_stop(RX_STREAM);
    dma_stop(TX_STREAM);
    spi2_reset_slave();

    if (received == SPI_LINK_FRAME_LEN)
    {
        uint8_t *t = s_rx_ready; s_rx_ready = s_rx_dma; s_rx_dma = t;
        s_rx_new = 1;
        s_full++;
    }
    else if (received > 0U)
    {
        s_short++;
    }

    /* 새 텔레메트리가 있으면 교체, 없으면 같은 프레임을 다시 보낸다 */
    if (s_tx_new)
    {
        uint8_t *t = s_tx_dma; s_tx_dma = s_tx_pending; s_tx_pending = t;
        s_tx_new = 0;
    }
    arm();
}

void PiSpi_Init(void)
{
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_SPI2_CLK_ENABLE();
    __HAL_RCC_DMA1_CLK_ENABLE();
    __HAL_RCC_SYSCFG_CLK_ENABLE();

    GPIO_InitTypeDef g = {0};
    g.Mode = GPIO_MODE_AF_PP;
    g.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    g.Alternate = GPIO_AF5_SPI2;

    /* CS(PB12): SPI 기능이 아니라 일반 입력 + EXTI. Pi 가 빠져 떠 있을 때 Low 로 읽히지 않게 풀업 */
    GPIO_InitTypeDef cs = {0};
    cs.Pin = GPIO_PIN_12;
    cs.Mode = GPIO_MODE_INPUT;
    cs.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOB, &cs);
    /* SCK: 모드 0 유휴 레벨이 low */
    g.Pin = GPIO_PIN_13;
    g.Pull = GPIO_PULLDOWN;
    HAL_GPIO_Init(GPIOB, &g);
    /* MISO / MOSI. MISO 는 이쪽이 내보내는 유일한 선이라 슬루를 낮춰 링잉을 줄인다
     * (LOW 도 Pi 1MHz 에는 충분, 수 MHz 로 올릴 땐 MEDIUM 이상으로) */
    g.Pin = GPIO_PIN_2 | GPIO_PIN_3;
    g.Pull = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &g);

    /* 첫 전송에서도 CRC 가 맞는 프레임이 나가도록 빈 텔레메트리로 채운다 */
    spi_link_telemetry_t empty;
    memset(&empty, 0, sizeof empty);
    spi_link_pack_telemetry(&empty, s_tx_dma);

    spi2_reset_slave();
    arm();

    /* EXTI12 <- PB12 상승 에지 */
    SYSCFG->EXTICR[3] = (SYSCFG->EXTICR[3] & ~SYSCFG_EXTICR4_EXTI12) | SYSCFG_EXTICR4_EXTI12_PB;
    EXTI->RTSR |= EXTI_RTSR_TR12;
    EXTI->FTSR &= ~EXTI_FTSR_TR12;
    EXTI->PR = EXTI_PR_PR12;
    EXTI->IMR |= EXTI_IMR_MR12;
    /* USART2 RX(5) 보다 높게: 다음 CS 가 내려가기 전에 반드시 재무장돼야 한다 */
    HAL_NVIC_SetPriority(EXTI15_10_IRQn, 4, 0);
    HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);
}

void EXTI15_10_IRQHandler(void)
{
    if (EXTI->PR & EXTI_PR_PR12)
    {
        EXTI->PR = EXTI_PR_PR12;
        if (cs_is_high())
        {
            on_cs_release();
        }
        else
        {
            s_glitches++;   /* 전송 중 CS 에 튄 잡음: 무시하고 계속 받는다 */
        }
    }
}

const uint8_t *PiSpi_TakeRxFrame(void)
{
    const uint8_t *frame = NULL;
    __disable_irq();
    if (s_rx_new)
    {
        uint8_t *t = s_rx_user; s_rx_user = s_rx_ready; s_rx_ready = t;
        s_rx_new = 0;
        frame = s_rx_user;
    }
    __enable_irq();
    return frame;   /* 다음 PiSpi_TakeRxFrame 호출 전까지 유효 */
}

uint8_t *PiSpi_TxBuffer(void)
{
    return s_tx_user;
}

void PiSpi_CommitTx(void)
{
    __disable_irq();
    uint8_t *t = s_tx_pending; s_tx_pending = s_tx_user; s_tx_user = t;
    s_tx_new = 1;
    __enable_irq();
}

void PiSpi_HoldIrq(int hold)
{
    /* NVIC 에서만 막는다. 그동안 온 CS 에지는 EXTI->PR 에 남아 있다가 풀자마자 처리된다.
     * Pi 는 20ms 마다 보내므로 수 ms 늦게 재무장해도 다음 프레임 전에 끝난다. */
    if (hold) HAL_NVIC_DisableIRQ(EXTI15_10_IRQn);
    else      HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);
}

uint32_t PiSpi_ShortTransfers(void)
{
    return s_short;
}

void PiSpi_GetStats(PiSpiStats_t *st)
{
    __disable_irq();
    st->cs_edges = s_cs_edges;
    st->full_frames = s_full;
    st->short_frames = s_short;
    st->glitches = s_glitches;
    st->last_len = s_last_len;
    st->last_head[0] = s_last_head[0];
    st->last_head[1] = s_last_head[1];
    __enable_irq();
    st->nss_level = (GPIOB->IDR & GPIO_PIN_12) ? 1U : 0U;
    st->sck_level = (GPIOB->IDR & GPIO_PIN_13) ? 1U : 0U;
}

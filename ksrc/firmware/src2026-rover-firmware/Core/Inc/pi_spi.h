/**
 * @file    pi_spi.h
 * @brief   Pi <-> Nucleo SPI2 슬레이브 링크 (DMA, 고정 길이 프레임)
 *
 * 핀 (rover_motor.ioc 와 동일):
 *   PB12 GPIO 입력 (Pi CE0)   -- 상승 에지 EXTI 로 프레임 경계 검출. SPI2 는 소프트웨어 NSS
 *                               (하드웨어 NSS 는 CS 잡음에 SCK 를 막아 비트가 밀렸다).
 *                               EXTI 진입 후 PB12 가 실제로 High 일 때만 프레임 끝으로 본다.
 *   PB13 SPI2_SCK  (Pi SCLK)
 *   PC2  SPI2_MISO (Pi MISO)
 *   PC3  SPI2_MOSI (Pi MOSI)
 * 모드 0 (CPOL=0, CPHA=0), 8비트, MSB first.
 *
 * SCK 상한: 슬레이브는 fPCLK1/2 까지. 지금 클럭이 HSI 16MHz (PLL 없음) 라서
 * **8MHz 가 절대 상한**이고, 여유를 두고 Pi 쪽은 1MHz 를 쓴다.
 *
 * 동작:
 *   CS 가 올라갈 때마다(EXTI12) DMA 를 멈추고, 받은 바이트 수가 정확히
 *   SPI_LINK_FRAME_LEN 이면 수신 프레임을 넘겨주고, 가장 최근 텔레메트리로
 *   송신 버퍼를 바꿔 다시 대기한다 (삼중 버퍼 포인터 교환만, ISR 수 us).
 *   짧게 끊긴 전송은 버리고 다음 CS 에서 자동으로 다시 맞춰진다 (DMA 가 한 번 밀려도 영원히 어긋나지 않는다).
 *
 * 지연: Pi 가 받는 텔레메트리는 **직전 CS 해제 시점까지 커밋된** 프레임이다.
 * 50Hz 제어주기와 Pi 교환 주기가 비동기라 나이는 0~40ms 이고, 같은 tick 을 두 번
 * 받거나 하나를 건너뛸 수 있다 (Pi 는 tick 으로 중복을 거른다).
 * CS 해제 후 재무장까지 수 us 가 걸리므로 Pi 는 전송 사이에 간격을 둔다 (50Hz 면 충분).
 *
 * CubeMX 에서 Generate Code 를 하면 MX_SPI2_Init() 이 생긴다. 이 모듈이 SPI2 를
 * 레지스터로 직접 잡으므로 **그 함수는 호출하지 말 것** (USART2/3 과 같은 방식).
 * 또 HAL_SPI_MODULE_ENABLED 와 stm32f4xx_hal_spi.c 가 Makefile 에 없어서
 * 재생성하면 링크가 깨진다 -- 재생성하지 않거나, 그 호출을 지운다.
 */

#ifndef PI_SPI_H
#define PI_SPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "spi_link.h"

/** SPI2 + DMA1(Stream3 RX / Stream4 TX, 채널 0) + EXTI12 초기화 후 첫 전송 대기 */
void PiSpi_Init(void);

/**
 * @brief  CS 가 올라간 뒤 완성된 수신 프레임을 가져온다 (복사 없음).
 * @return 새 프레임이 있으면 그 버퍼 (가장 최근 것만 남는다, 다음 호출 전까지 유효),
 *         없으면 NULL
 */
const uint8_t *PiSpi_TakeRxFrame(void);

/** 다음 텔레메트리를 채울 버퍼. 채운 뒤 PiSpi_CommitTx() 를 부른다. */
uint8_t *PiSpi_TxBuffer(void);

/** PiSpi_TxBuffer() 에 채운 프레임을 다음 SPI 전송에 쓰도록 넘긴다. */
void PiSpi_CommitTx(void);

/**
 * @brief  CS 인터럽트를 잠시 미룬다 (1) / 다시 받는다 (0).
 *         서보 응답(USART3 1Mbps 폴링, 바이트당 10us)을 읽는 동안 프레임 끝 ISR 이
 *         끼어들면 수신 오버런으로 읽기가 깨진다 (Pi 주기와 맞물려 몇 초마다 반복).
 */
void PiSpi_HoldIrq(int hold);

/** 길이가 맞지 않아 버린 전송 수 (CS 가 중간에 올라간 경우 등) */
uint32_t PiSpi_ShortTransfers(void);

/** 배선 진단용 누적 카운터 (부팅 후) */
typedef struct {
    uint32_t cs_edges;      /* 프레임 끝으로 인정한 CS 상승 에지 수 (0 이면 CE0->PB12 선 문제) */
    uint32_t full_frames;   /* 144바이트를 다 받은 전송 수 */
    uint32_t short_frames;  /* 일부만 받은 전송 수 */
    uint32_t glitches;      /* 무시한 CS 잡음 에지 수 (많으면 배선 개선 필요) */
    uint32_t last_len;      /* 마지막 전송에서 받은 바이트 수 (0 이면 SCLK->PB13 문제) */
    uint8_t  last_head[2];  /* 마지막 수신 첫 두 바이트 (A5 5A 가 아니면 MOSI->PC3 문제) */
    uint8_t  nss_level;     /* 지금 PB12 레벨 (Pi 유휴면 1) */
    uint8_t  sck_level;     /* 지금 PB13 레벨 (Pi 유휴면 0) */
} PiSpiStats_t;

void PiSpi_GetStats(PiSpiStats_t *st);

#ifdef __cplusplus
}
#endif

#endif /* PI_SPI_H */

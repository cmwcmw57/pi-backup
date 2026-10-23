/**
 * @file    imu_ism330.c
 * @brief   ISM330DHCX (SPI3) 드라이버 구현. 설명은 imu_ism330.h 참고.
 */

#include "imu_ism330.h"
#include "imu_mean_time.h"
#include "main.h"
#include <string.h>

/* 레지스터 (ISM330DHCX 데이터시트) */
#define REG_WHO_AM_I   0x0F
#define REG_CTRL1_XL   0x10
#define REG_CTRL2_G    0x11
#define REG_CTRL3_C    0x12
#define REG_STATUS     0x1E   /* bit0 XLDA, bit1 GDA: 새 가속도/자이로 샘플 있음 */
#define REG_OUTX_L_G   0x22   /* 0x22..0x27 자이로 XYZ, 0x28..0x2D 가속도 XYZ */

#define WHO_AM_I_VALUE 0x6B

#define CTRL3_SW_RESET 0x01
#define CTRL3_IF_INC   0x04
#define CTRL3_BDU      0x40
#define CTRL1_XL_CFG   0x58   /* ODR 208Hz (0101), FS ±4g (10) */
#define CTRL2_G_CFG    0x54   /* ODR 208Hz (0101), FS ±500dps (01) */

#define SPI_READ       0x80
#define SPI_TIMEOUT    2000U  /* 바이트당 폴링 한계 (1MHz 에서 1바이트 = 8us) */

static uint8_t s_valid = 0;

/* 제어주기 평균 (Imu_Poll 이 쌓고 Imu_TakeMean 이 꺼낸다) */
static int32_t s_sum[6];
static imu_time_accumulator_t s_read_times;
static imu_mean_time_t s_last_mean_time;
static uint8_t s_n = 0;

static inline void cs_low(void)  { GPIOC->BSRR = (uint32_t)GPIO_PIN_9 << 16; }
static inline void cs_high(void) { GPIOC->BSRR = GPIO_PIN_9; }

/* 1바이트 전이중 교환. 타임아웃이면 0 */
static int xfer(uint8_t tx, uint8_t *rx)
{
    uint32_t n = 0;
    while (!(SPI3->SR & SPI_SR_TXE)) { if (++n > SPI_TIMEOUT) return 0; }
    *(volatile uint8_t *)&SPI3->DR = tx;
    n = 0;
    while (!(SPI3->SR & SPI_SR_RXNE)) { if (++n > SPI_TIMEOUT) return 0; }
    *rx = *(volatile uint8_t *)&SPI3->DR;
    return 1;
}

static void wait_not_busy(void)
{
    for (uint32_t n = 0; (SPI3->SR & SPI_SR_BSY) && n < SPI_TIMEOUT; n++) { }
}

static int reg_write(uint8_t reg, uint8_t val)
{
    uint8_t dummy;
    cs_low();
    int ok = xfer(reg & 0x7F, &dummy) && xfer(val, &dummy);
    wait_not_busy();
    cs_high();
    return ok;
}

static int reg_read(uint8_t reg, uint8_t *buf, uint8_t len)
{
    uint8_t dummy;
    cs_low();
    int ok = xfer(reg | SPI_READ, &dummy);
    for (uint8_t i = 0; ok && i < len; i++)
    {
        ok = xfer(0x00, &buf[i]);
    }
    wait_not_busy();
    cs_high();
    return ok;
}

static int whoami_ok(void)
{
    uint8_t id = 0;
    return reg_read(REG_WHO_AM_I, &id, 1) && id == WHO_AM_I_VALUE;
}

/* 설정 레지스터가 우리가 쓴 값 그대로인지 (센서가 전원 흔들림으로 리셋되면
 * WHO_AM_I 는 그대로인데 설정만 기본값(정지)으로 돌아간다) */
static int config_ok(void)
{
    uint8_t v[3];
    if (!reg_read(REG_CTRL1_XL, v, 3)) return 0;
    return v[0] == CTRL1_XL_CFG && v[1] == CTRL2_G_CFG &&
           v[2] == (CTRL3_BDU | CTRL3_IF_INC);
}

static int configure(void)
{
    if (!whoami_ok()) return 0;

    reg_write(REG_CTRL3_C, CTRL3_SW_RESET);
    HAL_Delay(2);   /* 소프트 리셋 완료 대기 (데이터시트: 약 50us) */

    if (!reg_write(REG_CTRL3_C, CTRL3_BDU | CTRL3_IF_INC)) return 0;
    if (!reg_write(REG_CTRL1_XL, CTRL1_XL_CFG)) return 0;
    if (!reg_write(REG_CTRL2_G, CTRL2_G_CFG)) return 0;

    return config_ok();
}

int Imu_Init(void)
{
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_SPI3_CLK_ENABLE();

    GPIO_InitTypeDef g = {0};

    /* CS: 설정 전에 먼저 High 로 (센서가 I2C 모드로 오인하지 않게) */
    GPIOC->BSRR = GPIO_PIN_9;
    g.Pin = GPIO_PIN_9;
    g.Mode = GPIO_MODE_OUTPUT_PP;
    g.Pull = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &g);

    /* 출력 슬루를 가장 느리게: 1MHz 에는 충분하고, 점퍼선 링잉(가짜 클럭 -> 비트 오류,
     * 설정 레지스터 덮어쓰기)을 줄인다. 직렬 저항을 단 것과 비슷한 효과. */
    g.Pin = GPIO_PIN_10 | GPIO_PIN_11 | GPIO_PIN_12;
    g.Mode = GPIO_MODE_AF_PP;
    g.Pull = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_LOW;
    g.Alternate = GPIO_AF6_SPI3;
    HAL_GPIO_Init(GPIOC, &g);

    /* 마스터, 모드 0, 8비트, MSB first, BR=011 (fPCLK/16), 소프트웨어 NSS */
    SPI3->CR1 = 0;
    SPI3->CR2 = 0;
    SPI3->CR1 = SPI_CR1_MSTR | SPI_CR1_BR_0 | SPI_CR1_BR_1 | SPI_CR1_SSM | SPI_CR1_SSI;
    SPI3->CR1 |= SPI_CR1_SPE;

    HAL_Delay(20);  /* 전원 투입 후 부팅 시간 (데이터시트 10ms) */
    s_valid = (uint8_t)configure();
    return s_valid;
}

int Imu_Read(int16_t out[6])
{
    uint8_t b[12];
    if (!s_valid || !reg_read(REG_OUTX_L_G, b, sizeof b))
    {
        memset(out, 0, 6 * sizeof out[0]);
        return 0;
    }
    /* 12바이트가 전부 같으면 주소 자동증가가 꺼진 것 (SPI 잡음으로 설정 레지스터가
     * 덮어써진 경우 등 -- 같은 레지스터만 반복해 읽힌다). 실측 노이즈가 있어 정상
     * 데이터에선 사실상 나오지 않는다. 무효로 두고 다음 Imu_Check 에서 재설정. */
    int same = 1;
    for (int i = 1; i < (int)sizeof b; i++)
    {
        if (b[i] != b[0]) { same = 0; break; }
    }
    if (same)
    {
        s_valid = 0;
        memset(out, 0, 6 * sizeof out[0]);
        return 0;
    }
    for (int i = 0; i < 6; i++)
    {
        out[i] = (int16_t)(b[2 * i] | (b[2 * i + 1] << 8));
    }
    return 1;
}

int Imu_Check(void)
{
    /* WHO_AM_I 와 설정을 둘 다 확인하고, 어긋났으면 바로 재설정 */
    if (!s_valid || !whoami_ok() || !config_ok())
    {
        s_valid = (uint8_t)configure();
    }
    return s_valid;
}

int Imu_Poll(int16_t sample[6])
{
    uint8_t st;
    int16_t v[6];
    if (!s_valid || !reg_read(REG_STATUS, &st, 1) || (st & 0x03) != 0x03) return 0;
    if (!Imu_Read(v)) return 0;
    uint32_t read_ms = HAL_GetTick();
    if (s_n == 255) return 0;
    for (int i = 0; i < 6; i++) s_sum[i] += v[i];
    s_n++;
    imu_time_add(&s_read_times, read_ms);
    if (sample != NULL) memcpy(sample, v, sizeof v);
    return 1;
}

int Imu_TakeMean(int16_t out[6], uint8_t *n)
{
    *n = s_n;
    if (s_n == 0)
    {
        int ok = Imu_Read(out);       /* 순간값의 읽기 완료 시각; acquisition 시각이 아님 */
        imu_time_accumulator_t instant = {0};
        if (ok) imu_time_add(&instant, HAL_GetTick());
        s_last_mean_time = imu_time_take(&instant);
        return ok;
    }
    for (int i = 0; i < 6; i++)
    {
        int32_t q = s_sum[i];
        out[i] = (int16_t)((q >= 0 ? q + s_n / 2 : q - s_n / 2) / s_n);   /* 반올림 */
        s_sum[i] = 0;
    }
    s_last_mean_time = imu_time_take(&s_read_times);
    s_n = 0;
    return s_valid;
}

/* 마지막 평균의 대표 읽기 시각 - 제어 reference, microsecond 단위. */
int Imu_MeanReadOffsetUs(uint32_t reference_ms, int16_t *offset)
{
    return imu_mean_offset_us(&s_last_mean_time, reference_ms, offset);
}

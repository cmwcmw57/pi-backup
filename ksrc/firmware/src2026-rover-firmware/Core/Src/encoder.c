/**
 * @file    encoder.c
 * @brief   Quadrature encoder implementation (x4 mode, hall-sensor encoders)
 */

/* Includes ------------------------------------------------------------------*/
#include "encoder.h"
#include "main.h"   /* HAL types, Error_Handler, etc. */

/* Private types -------------------------------------------------------------*/

/** Per-encoder hardware descriptor */
typedef struct
{
    TIM_HandleTypeDef *htim;    /**< Encoder timer handle        */
    uint8_t            is32bit; /**< 1 if timer is 32-bit (TIM5) */
} EncoderHW_t;

/* External timer handles (defined in main.c by CubeMX) ---------------------*/
extern TIM_HandleTypeDef htim1;
extern TIM_HandleTypeDef htim4;
extern TIM_HandleTypeDef htim5;
extern TIM_HandleTypeDef htim8;

/* Private variables ---------------------------------------------------------*/

/** Lookup table: MotorID → encoder timer */
static const EncoderHW_t encoderHW[MOTOR_COUNT] =
{
    [MOTOR1] = { .htim = &htim1, .is32bit = 0 },   /* 16-bit, Period = 65535      */
    [MOTOR2] = { .htim = &htim4, .is32bit = 0 },   /* 16-bit, Period = 65535      */
    [MOTOR3] = { .htim = &htim5, .is32bit = 1 },   /* 32-bit, Period = 0xFFFFFFFF */
    [MOTOR4] = { .htim = &htim8, .is32bit = 0 },   /* 16-bit, Period = 65535      */
};

/* Exported functions --------------------------------------------------------*/

void Encoder_Init(void)
{
    for (uint8_t i = 0; i < MOTOR_COUNT; i++)
    {
        HAL_TIM_Encoder_Start((TIM_HandleTypeDef *)encoderHW[i].htim,
                              TIM_CHANNEL_ALL);
    }
}

int32_t Encoder_GetCount(MotorID id)
{
    if (id >= MOTOR_COUNT) return 0;

    const EncoderHW_t *hw = &encoderHW[id];
    uint32_t cnt = __HAL_TIM_GET_COUNTER(hw->htim);

    if (hw->is32bit)
    {
        /* TIM5: 32-bit counter – reinterpret as signed directly */
        return (int32_t)cnt;
    }
    else
    {
        /* 16-bit counter – cast to int16_t first for correct sign extension
         * e.g. 0xFFFF → -1, 0xFFFE → -2, etc. */
        return (int32_t)(int16_t)(uint16_t)cnt;
    }
}

uint8_t Encoder_IsWide(MotorID id)
{
    if (id >= MOTOR_COUNT) return 0U;
    return encoderHW[id].is32bit;
}

void Encoder_Reset(MotorID id)
{
    if (id >= MOTOR_COUNT) return;

    __HAL_TIM_SET_COUNTER(encoderHW[id].htim, 0U);
}

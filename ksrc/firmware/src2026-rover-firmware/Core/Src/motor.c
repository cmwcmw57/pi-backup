/**
 * @file    motor.c
 * @brief   DC Motor control implementation (Cytron MDD3A, 2-PWM mode)
 */

/* Includes ------------------------------------------------------------------*/
#include "motor.h"
#include "main.h"   /* HAL types, Error_Handler, etc. */

/* Private types -------------------------------------------------------------*/

/** Per-motor hardware descriptor */
typedef struct
{
    TIM_HandleTypeDef *htim;    /**< Timer handle            */
    uint32_t           chA;     /**< Forward PWM channel     */
    uint32_t           chB;     /**< Reverse PWM channel     */
} MotorHW_t;

/* External timer handles (defined in main.c by CubeMX) ---------------------*/
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim3;
extern TIM_HandleTypeDef htim12;

/* Private variables ---------------------------------------------------------*/

/** Lookup table: MotorID → (timer, channelA, channelB) */
static const MotorHW_t motorHW[MOTOR_COUNT] =
{
    [MOTOR1] = { .htim = &htim2,  .chA = TIM_CHANNEL_1, .chB = TIM_CHANNEL_2 },  /* PA5 / PB3   */
    [MOTOR2] = { .htim = &htim12, .chA = TIM_CHANNEL_1, .chB = TIM_CHANNEL_2 },  /* PB14 / PB15 */
    [MOTOR3] = { .htim = &htim3,  .chA = TIM_CHANNEL_1, .chB = TIM_CHANNEL_2 },  /* PA6 / PA7   */
    [MOTOR4] = { .htim = &htim3,  .chA = TIM_CHANNEL_3, .chB = TIM_CHANNEL_4 },  /* PB0 / PB1   */
};

/* Private helper ------------------------------------------------------------*/

/**
 * @brief  Clamp a value to the range [min, max].
 */
static inline int16_t clamp(int16_t val, int16_t min, int16_t max)
{
    if (val > max) return max;
    if (val < min) return min;
    return val;
}

/* Exported functions --------------------------------------------------------*/

void Motor_Init(void)
{
    for (uint8_t i = 0; i < MOTOR_COUNT; i++)
    {
        HAL_TIM_PWM_Start((TIM_HandleTypeDef *)motorHW[i].htim, motorHW[i].chA);
        HAL_TIM_PWM_Start((TIM_HandleTypeDef *)motorHW[i].htim, motorHW[i].chB);
    }
}

void Motor_SetSpeed(MotorID id, int16_t speed)
{
    if (id >= MOTOR_COUNT) return;

    speed = clamp(speed, -PWM_MAX, PWM_MAX);

    const MotorHW_t *hw = &motorHW[id];

    if (speed >= 0)
    {
        __HAL_TIM_SET_COMPARE(hw->htim, hw->chA, (uint32_t)speed);
        __HAL_TIM_SET_COMPARE(hw->htim, hw->chB, 0U);
    }
    else
    {
        __HAL_TIM_SET_COMPARE(hw->htim, hw->chA, 0U);
        __HAL_TIM_SET_COMPARE(hw->htim, hw->chB, (uint32_t)(-speed));
    }
}

void Motor_Stop(MotorID id)
{
    if (id >= MOTOR_COUNT) return;

    const MotorHW_t *hw = &motorHW[id];

    __HAL_TIM_SET_COMPARE(hw->htim, hw->chA, 0U);
    __HAL_TIM_SET_COMPARE(hw->htim, hw->chB, 0U);
}

void Motor_StopAll(void)
{
    for (uint8_t i = 0; i < MOTOR_COUNT; i++)
    {
        Motor_Stop((MotorID)i);
    }
}

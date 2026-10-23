/**
 * @file    motor.h
 * @brief   DC Motor control interface (Cytron MDD3A, 2-PWM mode)
 *
 * Four geared DC motors (SPG30E-GR56) driven by two MDD3A dual-channel
 * motor drivers.  Each motor uses two PWM channels: one for forward duty
 * and one for reverse duty.
 */

#ifndef MOTOR_H
#define MOTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>

/* Exported defines ----------------------------------------------------------*/

/** Maximum PWM duty value (= TIM ARR, 0–799 → ~20 kHz) */
#define PWM_MAX  799

/* Exported types ------------------------------------------------------------*/

/** Motor identifier (matches physical wiring order) */
typedef enum
{
    MOTOR1 = 0,
    MOTOR2,
    MOTOR3,
    MOTOR4,
    MOTOR_COUNT   /**< sentinel – also gives the number of motors */
} MotorID;

/* Exported function prototypes ----------------------------------------------*/

/**
 * @brief  Start all PWM channels used for motor drive.
 * @note   Call once after MX_TIMx_Init() have completed.
 */
void Motor_Init(void);

/**
 * @brief  Set motor speed and direction.
 * @param  id     Motor to control (MOTOR1 … MOTOR4).
 * @param  speed  Signed duty: -PWM_MAX … +PWM_MAX.
 *                Positive → channel-A carries duty, channel-B = 0 (forward).
 *                Negative → channel-B carries |duty|, channel-A = 0 (reverse).
 *                Values outside the range are clamped.
 */
void Motor_SetSpeed(MotorID id, int16_t speed);

/**
 * @brief  Stop a single motor (both PWM channels set to 0).
 * @param  id  Motor to stop.
 */
void Motor_Stop(MotorID id);

/**
 * @brief  Stop all four motors.
 */
void Motor_StopAll(void);

#ifdef __cplusplus
}
#endif

#endif /* MOTOR_H */

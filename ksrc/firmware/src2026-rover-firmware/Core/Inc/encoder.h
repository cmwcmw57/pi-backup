/**
 * @file    encoder.h
 * @brief   Quadrature encoder interface (x4 mode, hall-sensor encoders)
 *
 * Four encoders, each connected to a dedicated timer in Encoder Mode
 * (TI1 & TI2).  This module provides raw count reading only;
 * RPM / velocity calculations are handled elsewhere.
 */

#ifndef ENCODER_H
#define ENCODER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>
#include "motor.h"  /* for MotorID enum */

/* Exported function prototypes ----------------------------------------------*/

/**
 * @brief  Start all encoder timers in quadrature encoder mode.
 * @note   Call once after MX_TIMx_Init() have completed.
 */
void Encoder_Init(void);

/**
 * @brief  Read the current encoder count (signed).
 * @param  id  Motor / encoder to read (MOTOR1 … MOTOR4).
 * @return Current CNT value as a signed 32-bit integer.
 *
 * @note   TIM1 / TIM4 / TIM8 are 16-bit counters (Period = 65535):
 *         the raw 16-bit value is cast to int16_t first so that
 *         underflow (e.g. 0xFFFF → −1) is handled correctly, then
 *         widened to int32_t.
 * @note   TIM5 is a 32-bit counter (Period = 0xFFFFFFFF):
 *         the raw value is reinterpreted as int32_t directly.
 */
int32_t Encoder_GetCount(MotorID id);

/**
 * @brief  Is this encoder's timer a 32-bit counter?
 * @param  id  Motor / encoder to query.
 * @return 1 for the 32-bit timer (TIM5), 0 for the 16-bit ones.
 *
 * @note   Callers that difference successive Encoder_GetCount() values need
 *         this: a 16-bit counter wraps every 65536 ticks (~7.8 wheel revs,
 *         i.e. about 6 s at full speed), and the wrapped difference must be
 *         folded back with an (int16_t) cast or it reads as a 65535-tick jump.
 */
uint8_t Encoder_IsWide(MotorID id);

/**
 * @brief  Reset the encoder count to zero.
 * @param  id  Motor / encoder to reset.
 */
void Encoder_Reset(MotorID id);

#ifdef __cplusplus
}
#endif

#endif /* ENCODER_H */

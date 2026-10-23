/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stddef.h>
#include "motor.h"
#include "encoder.h"
#include "sts3215.h"
#include "joint.h"
#include "swerve_kinematics.h"
#include "mdd3a_driver.h"
#include "drive_align_gate.h"
#include "wheel_pi.h"
#include "spi_link.h"
#include "pi_spi.h"
#include "imu_ism330.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct {
    int32_t  last_raw_count;  /**< Previous raw encoder tick */
    int64_t  total_count;     /**< Cumulative tick count (no rollover) */
    float    rpm;             /**< Wheel RPM */
    float    rad_per_sec;     /**< Angular velocity [rad/s] */
    float    linear_vel;      /**< Linear velocity [m/s] = w * R */
} WheelState_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* Motor & Encoder Specifications */
#define GEAR_RATIO          131.0f             /**< 131:1 Gearbox reduction (SPG30E-GR131) */
/* 조향 가동범위는 바퀴마다 다르다 (구동모터가 몸체 안쪽을 향하게 고정).
 * joint.c 의 joint_config[].min_deg/max_deg 가 유일한 정본이고, Swerve_Control 이
 * 매 주기 읽어서 swerve_fold_to_range 에 넘긴다. */

/* 제어주기.  시뮬(sim/rl/env.py CTRL_HZ=50)과 같은 50Hz.
 * 명령(UART V / SPI)은 목표값만 바꾸고, IK·정렬 게이트·서보·PWM·텔레메트리는
 * 전부 이 주기에서만 나간다. */
#define CTRL_PERIOD_MS          20U

/* 구동 정렬 게이트 (sim/rl/env.py 의 drive_align_gate 와 같은 값).
 * 목표 조향각이 한 주기에 60도 넘게 바뀌면, 네 바퀴 실제각이 목표의 10도 안에
 * 들 때까지 구동을 보류한다.  0 으로 두면 게이트 OFF. */
#define DRIVE_GATE_TRIGGER_DEG  60.0f
#define DRIVE_GATE_RELEASE_DEG  10.0f
/* 이 시간 넘게 정렬이 안 되면 (모래에서 정지 조향이 안 먹힘, 기구 간섭 등) 보류를
 * 풀고 텔레메트리 GATE_TIMEOUT 을 세운다. 영원히 구동이 막히는 교착 방지. */
#define DRIVE_GATE_MAX_HOLD_MS  1500U

#define ENCODER_PPR         16.0f              /**< 16 pulses per motor rev. 2026-10-03 확인: MOT 1 799 2000 무부하 77.8 rpm (정격 76) */
#define ENCODER_CPR         (ENCODER_PPR * 4.0f) /**< 64 ticks in 4x mode (TIM_ENCODERMODE_TI12) */
#define TICKS_PER_REV       (ENCODER_CPR * GEAR_RATIO) /**< 8384.0 ticks per wheel rev */

/* 바퀴 반경.  URDF(ros/rover_description, rover_wheel_rev4) 충돌 실린더 반경이 정답이고
 * Isaac contract.WHEEL_RADIUS_M, sim/rl/config.py 의 wheel_r 과 같은 값이어야 한다.
 * 2026-10-12: 0.03995 (구 조립본 rev_2 STL) → 0.047503.  옛 값은 19% 작아서 텔레메트리
 * wheel_mps 와 duty 환산이 모두 19% 어긋나 있었다.  개루프 duty 에서는 "보고되는 속도가
 * 틀리다" 로 끝나지만, 바퀴 속도 PI 폐루프를 넣으면 PI 가 틀린 측정값을 목표에 맞추려고
 * duty 를 계속 왜곡하므로 치명적이다. */
#define WHEEL_RADIUS_M      0.047503f          /**< Radius R = 47.5mm, URDF rover_wheel_rev4 */
#define PI                  3.1415926535f

/* --- Swerve module geometry & actuator mapping (bench-test placeholder) ---
 * x=forward, y=left, matches firmware/common/tests/test_swerve_kinematics.c
 * and sim/mujoco/ksrc_rover.urdf. Real offsets pending CAD; module<->MOTOR/
 * servo-ID mapping is also unconfirmed against actual wiring -- both
 * placeholders until bench-verified. Servo IDs use joint.c's array index
 * (0-based; joint_config[] maps index->bus ID internally). */
/* CAD 기준 (ros/rover_description, 2026-10-13 사용자 결정 "CAD 기준이 맞음"):
 * 중심 -> 앞/뒤 조향축 102.431 mm (로커 피벗 85 + 조향 링크 17.431), 좌/우 132.5 mm.
 * sim/rl/config.py axle_x=0.102431, track=0.265 / Isaac contract.py MODULE_XY_M 과 같아야 한다.
 * (10-03 에 "실측"으로 적었던 105 mm 는 출처 기록이 없어 버림.  10-05 에 시뮬은 CAD 로 바뀌었는데
 *  펌웨어만 105 로 남아 있었다.)
 * 이전 STL 값 (±0.0918, ±0.11835).
 * 이전 (±0.12, ±0.12) 는 추정치였다.  순수 병진(vx, vy)에서는 모듈 위치가 결과에
 * 영향을 주지 않지만 omega != 0 이면 v_i = v_B + omega x r_i 로 오차가 남는다:
 *     제자리회전 w=0.8  -> 조향각 최대 7.20도
 *     선회 vx=0.2 w=0.5 -> 최대 5.15도
 *     실주행 중앙값     -> 0.90도
 * 시뮬과 반드시 같은 값이어야 한다 (같은 C 코드를 공유하는 의미가 없어진다). */
static const swerve_module_pos_t SWERVE_MODULES[SWERVE_NUM_MODULES] = {
    { 0.102431f,  0.1325f },  /* module 0: FL */
    { 0.102431f, -0.1325f },  /* module 1: FR */
    {-0.102431f,  0.1325f },  /* module 2: RL */
    {-0.102431f, -0.1325f },  /* module 3: RR */
};
/* 모듈 -> 구동모터.  2026-10-03 확정: 모터 번호 배치는 조향 서보와 같다.
 *     MOTOR1 LF   MOTOR3 RF
 *     MOTOR2 LB   MOTOR4 RB
 * 모듈 순서는 FL, FR, RL, RR 이므로 {MOTOR1, MOTOR3, MOTOR2, MOTOR4}. */
static const MotorID SWERVE_MODULE_MOTOR[SWERVE_NUM_MODULES] = { MOTOR1, MOTOR3, MOTOR2, MOTOR4 };

/* 모듈 -> 조향 서보 인덱스 (joint_config[] 인덱스 = 서보 ID - 1).
 *     서보1 LF   서보3 RF
 *     서보2 LB   서보4 RB
 * 모듈 순서는 FL, FR, RL, RR 이므로 {서보1, 서보3, 서보2, 서보4}. */
static const uint8_t SWERVE_MODULE_SERVO[SWERVE_NUM_MODULES] = { 0, 2, 1, 3 };

/* ★ 조립 후 확인 (TODO): 구동모터·엔코더 부호. 지금은 전부 +1 (미확인).
 * 좌우 모터가 거울 대칭으로 달리면 한쪽은 -1 이어야 한다. 틀리면 V 0.1 0 0 이
 * 전진이 아니라 제자리 회전이 된다.  확인 방법 (바퀴를 띄운 상태):
 *   1) MTEST 로 모터를 하나씩 돌려 MOTOR1..4 가 어느 바퀴인지 -> SWERVE_MODULE_MOTOR
 *   2) V 0.05 0 0 에서 전진 방향으로 안 도는 바퀴 -> 그 모터의 MOTOR_DRIVE_SIGN = -1
 *   3) 그 상태에서 LOG 의 m/s 가 음수로 나오는 바퀴 -> ENCODER_SIGN = -1
 * 인덱스는 MOTOR1..MOTOR4 (모듈 순서가 아님). 옵시디언 "실기 확인 목록" 에도 기록. */
/* 구동·엔코더 부호.  인덱스는 모터 번호: MOTOR1 LF, MOTOR2 LB, MOTOR3 RF, MOTOR4 RB.
 *
 * 모터는 넷 다 같은 방식으로 단다 (몸체 안에서 바깥을 보고, 샤프트에 바퀴).
 * 그래서 같은 duty + 면 넷 다 샤프트 기준 같은 방향으로 돌고, 샤프트가 좌우로
 * 반대를 보므로 **왼쪽과 오른쪽은 반대로 구른다** -> 부호는 L 끼리 같고 R 끼리 같다.
 *
 * 2026-10-03 로버에 달고 확인: LF·LB 는 duty + 에서 후진 -> L = -1, RB 는 전진 -> R = +1.
 * (RF 는 -1 로 관찰됐지만 대칭에 맞지 않아 +1 로 둔다. V 0.05 0 0 에서 RF 만 뒤로
 *  구르면 RF 모터 출력선(M+/M-)이 반대로 꽂힌 것 -> 부호가 아니라 **선을 바꿔** 고친다.
 *  부호로 덮으면 RF 의 엔코더 부호만 어긋난다.)
 *
 * 엔코더는 전원선과 무관하게 실제 회전을 센다. MOTOR1 이 duty + 에서 tick + 였으므로
 * (bench 실측) 전진할 때 + 가 되려면 ENCODER_SIGN = MOTOR_DRIVE_SIGN.
 * 확인: V 0.05 0 0 에서 넷 다 앞으로 구르고, LOG 의 M1~M4 m/s 가 넷 다 + . */
static const int8_t MOTOR_DRIVE_SIGN[MOTOR_COUNT] = { -1, -1, +1, +1 };
/* 2026-10-11: M1 엔코더는 A/B 선이 PA8/PA9 에 다른 모터와 반대로 물려 있다 (선은 못 바꿈).
 * TIM1 IC1 극성을 FALLING 으로 (MX_TIM1_Init + ioc) 해서 하드웨어에서 계수 방향을 뒤집었다.
 * 그래서 Raw·MOT 도 다른 모터와 같고, 부호표는 규칙대로 = MOTOR_DRIVE_SIGN. */
static const int8_t ENCODER_SIGN[MOTOR_COUNT]     = { -1, -1, +1, +1 };

/* Placeholder until bench-measured (free-spin m/s at 100% duty, see
 * mdd3a_driver.h's open item). Wrong magnitude only scales wheel speed
 * for a given vx/vy/omega, doesn't change direction/shape. */
/* 듀티 100% 에서의 바퀴 선속도.  시뮬은 mot_w_noload * wheel_r = 7.96 * 0.047503
 * = 0.378 m/s 로 계산한다.  이 값이 어긋나면 IK 의 desaturation 기준이 달라져
 * 같은 cmd_vel 에 대해 시뮬과 실기의 스로틀 스케일이 맞지 않는다.
 * 2026-10-12: 0.318 (구 반경 0.03995 기준) → 0.378.  실측 무부하 77.8 rpm = 8.15 rad/s
 * 이면 0.387 (+2%).  모터 모델링 D1 (firmware/MOTOR_MODELING_PLAN.md) 로 확정할 값이고,
 * 지금은 시뮬과 일치시킨다. */
#define SWERVE_MAX_WHEEL_MPS  0.378f

/* STS3215 레지스터 중 sts3215.h 에 없는 것 (공식 메모리표 기준) */
#define STS_REG_ID_ADDR    0x05  /**< 서보 버스 ID (EPROM, 0~253) */
#define STS_REG_LOCK_FLAG  0x37  /**< 잠금 플래그: 0 = EPROM 쓴 값이 전원 차단 후에도 유지 (기본값) */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim5;
TIM_HandleTypeDef htim8;
TIM_HandleTypeDef htim12;

/* USER CODE BEGIN PV */

/* === 터미널 명령 수신 버퍼 (USART2 RX 인터럽트) === */
#define CMD_BUF_SIZE  64
static char     cmd_buf[CMD_BUF_SIZE];  /**< ISR 이 문자를 모으는 버퍼 */
static char     cmd_line[CMD_BUF_SIZE]; /**< 완성된 줄 (메인 루프가 읽음) */
static uint8_t  cmd_idx = 0;            /**< 현재 버퍼 인덱스 */
static volatile uint8_t cmd_ready = 0;  /**< 1이면 완성된 명령이 있음 */

/** 주기적 엔코더/속도 로그 출력 여부. 조이스틱이나 SCAN/ZERO 같은 대화형
 *  작업 중에는 10Hz x 5줄이 터미널을 덮어버려서 끌 수 있어야 한다. */
static uint8_t  s_log_enabled = 1;

/** 부팅 모터 자가시험. 예전에는 기본으로 돌아서 전원을 넣으면 첫 주행 명령이 올
 *  때까지 바퀴가 2~3초씩 무한 반복으로 돌았다 (Pi 부팅 전 로버가 스스로 움직임).
 *  이제 기본 OFF 이고 터미널 MTEST 로만 켠다. */
static uint8_t  s_mtest_enabled = 0;

/** 텔레옵 중 5Hz 상태줄 ([CTRL] 받은 vx vy w / 조향각 / duty). 엔코더 로그(LOG)와
 *  따로 켠다 -- 조이스틱이 보낸 값이 STM 에 제대로 들어오는지 확인용.
 *  텔레옵이 시작돼도 꺼지지 않는다. 한 줄 ~90B 라 200ms 마다 ~8ms 송신 대기. */
static uint8_t  s_ctrl_log = 0;

/** 받은 문자를 되돌려 보낼지. 안 그러면 터미널에서 타이핑이 안 보인다.
 *  ISR 에서 송신 대기를 하므로 조이스틱이 연속 전송할 때는 수신 오버런을
 *  유발할 수 있어, 텔레옵이 시작되면 자동으로 꺼진다. */
static uint8_t  s_echo_enabled = 1;

/* === 텔레옵 상태 === */
/** 명령이 이 시간 이상 끊기면 목표 속도를 0 으로 (조이스틱 종료/USB·SPI 분리 대비) */
#define TELEOP_TIMEOUT_MS  500
static uint8_t  s_teleop_active = 0;      /**< 구동 명령을 한 번이라도 받으면 1 */
static uint32_t s_last_cmd_ms = 0;        /**< 마지막 구동 명령 시각 */
static uint8_t  s_teleop_timed_out = 0;   /**< 워치독이 이미 정지시켰으면 1 */
static float    s_cmd_target[3] = {0};    /**< 마지막으로 받은 (vx, vy, omega) */
static uint8_t  s_cmd_from_spi = 0;       /**< 마지막 명령 출처 (1=SPI, 0=UART) */

/* === 제어주기 결과 (텔레메트리로 나간다) === */
static float    s_cmd_applied[3] = {0};              /**< 워치독 적용 후 IK 입력 */
/* RL Res8 잔차 (SPI 타입 0x02). 모듈 순서. teleop_command 가 s_res_have 를 내리고 0x02 처리가 다시 세운다 */
static float    s_res_d[4] = {0};                    /**< 조향 잔차 [rad] (±SPI_RES_STEER_MAX 로 자름) */
static float    s_res_g[4] = {1.0f, 1.0f, 1.0f, 1.0f}; /**< 바퀴 속도 배율 */
static uint32_t s_res_ms = 0;                        /**< 마지막 잔차 수신 시각 */
static uint8_t  s_res_have = 0;
static uint8_t  s_res_active = 0;                    /**< 이번 주기에 잔차를 더했나 (텔레메트리) */
static float    s_steer_goal_rad[SWERVE_NUM_MODULES]; /**< 접은 뒤 목표 조향각 */
static int16_t  s_duty_applied[SWERVE_NUM_MODULES];   /**< 게이트 적용 후 duty */
static uint8_t  s_servo_read_err = 0;
static uint8_t  s_steer_clamped = 0;
static drive_align_gate_t s_gate;

/* === 바퀴 속도 PI (common/wheel_pi.h, MOTOR_MODELING_PLAN.md §8) ===
 * ON 이면 duty = FF + 기준 모델 PI (엔코더 속도를 목표에 맞춤), OFF 면 예전 개루프
 * duty = v / SWERVE_MAX_WHEEL_MPS × 799.  2026-10-14 실물 확인 뒤 기본 ON (시뮬도 PI 로 학습):
 *   벤치 계단 0->2 rad/s 90% 41–51 ms, 지면 0.15 m/s 직진 네 바퀴 정확히 0.150 (개루프 0.120–0.127),
 *   바퀴 간 duty 차이 7 %, 스톨 보호 동작.  개루프로 돌리려면 터미널 `WPI OFF`. */
#define WHEEL_PI_DEFAULT_ON  1
static uint8_t           s_wpi_on = WHEEL_PI_DEFAULT_ON;
static wheel_pi_cfg_t    s_wpi_cfg;
static wheel_pi_state_t  s_wpi[SWERVE_NUM_MODULES];     /**< 모듈 순서 */

/** 바퀴 PI 보호가 걸린 모듈이 있나 (0 = 스톨, 1 = 엔코더 방향 불일치 잠금). 텔레메트리 flags 용. */
static int wpi_any(int which)
{
  for (int m = 0; m < SWERVE_NUM_MODULES; m++)
    if (which ? s_wpi[m].enc_dir : s_wpi[m].stall) return 1;
  return 0;
}
static float             s_wheel_w[MOTOR_COUNT];         /**< 이번 주기 엔코더 속도 rad/s (+ = 전진, 20 ms 창), 모터 순서 */

/* === 센서 (매 제어주기 읽음, 모듈 순서 FL, FR, RL, RR) === */
/** 읽기 실패한 서보는 이 주기 수만큼 건너뛴다. 응답 없는 서보 하나가 매 주기
 *  수신 타임아웃(15ms)을 먹어 20ms 주기를 밀어내지 않게 한다. */
#define SERVO_RETRY_TICKS  25U   /* 0.5 s */
static JointFeedback_t s_servo_fb[SWERVE_NUM_MODULES];
static uint8_t  s_servo_valid = 0;                      /**< 비트 m = 모듈 m 이번 주기 유효 */
static uint8_t  s_servo_skip[SWERVE_NUM_MODULES] = {0}; /**< 남은 건너뛰기 주기 */
static uint8_t  s_servo_failed[SWERVE_NUM_MODULES] = {0}; /**< 직전 읽기 실패 */
static int16_t  s_imu_raw[6] = {0};                     /**< gyro xyz, acc xyz (제어주기 평균) */
static uint8_t  s_imu_n = 0;                            /**< 평균에 들어간 샘플 수 (0 = 순간값) */
static uint8_t  s_imu_valid = 0;

/* === Pi SPI 링크 상태 === */
static uint8_t  s_spi_seq_echo = 0;       /**< 마지막으로 받아들인 프레임 seq */
static uint16_t s_spi_rx_err = 0;         /**< 버린 프레임 수 (CRC/동기/길이/비유한값) */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM4_Init(void);
static void MX_TIM5_Init(void);
static void MX_TIM8_Init(void);
static void MX_TIM12_Init(void);
/* USER CODE BEGIN PFP */
static void MX_USART2_Init(void);
static void MX_USART3_Init(void);
static void servo_boot_sequence(void);
static void process_command(const char *cmd);
static void steer_step_response(int idx, float target, int raw, int ms_fixed);
static void print_help(void);
static void Swerve_Control(float vx, float vy, float omega, const float *d_steer, const float *wheel_gain);
static void teleop_command(float vx, float vy, float omega, uint8_t from_spi);
static void pi_spi_poll(void);
static void read_sensors(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static void MX_USART2_Init(void)
{
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_USART2_CLK_ENABLE();

  /* PA2: USART2_TX, PA3: USART2_RX (Connected to ST-LINK VCP) */
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Pin = GPIO_PIN_2 | GPIO_PIN_3;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* Configure Baud rate 115200 (Auto-calculated based on APB1 clock) */
  uint32_t pclk1 = HAL_RCC_GetPCLK1Freq();
  USART2->BRR = (pclk1 + (115200 / 2)) / 115200;
  USART2->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

int __io_putchar(int ch)
{
  while (!(USART2->SR & USART_SR_TXE));
  USART2->DR = (uint8_t)ch;
  return ch;
}

int _write(int file, char *ptr, int len)
{
  (void)file;
  for (int i = 0; i < len; i++)
  {
    if (*ptr == '\n')
    {
      __io_putchar('\r');
    }
    __io_putchar(*ptr++);
  }
  return len;
}

/* Newlib-nano syscall stubs (required when using printf with nano.specs) */
int _read(int file, char *ptr, int len)   { (void)file; (void)ptr; (void)len; return 0; }
int _close(int file)                      { (void)file; return -1; }
int _isatty(int file)                     { (void)file; return 1; }
int _lseek(int file, int ptr, int dir)    { (void)file; (void)ptr; (void)dir; return 0; }
#include <sys/stat.h>
int _fstat(int file, struct stat *st)     { (void)file; st->st_mode = S_IFCHR; return 0; }
int _getpid(void)                         { return 1; }
int _kill(int pid, int sig)               { (void)pid; (void)sig; return -1; }

/* ================================================================== */
/*        USART3 초기화 — STS3215 서보 통신 (PB10/PB11, 1Mbps)        */
/* ================================================================== */
static void MX_USART3_Init(void)
{
  /* GPIO 클럭 및 USART3 클럭 활성화 */
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_USART3_CLK_ENABLE();

  /* PB10: USART3_TX (Single-Wire Half-Duplex 모드, 이 핀 하나로 송수신 동시 처리) */
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Pin = GPIO_PIN_10;                 /* PB10 하나만 사용 */
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;            /* ★ 반드시 Open-Drain으로 설정 */
  GPIO_InitStruct.Pull = GPIO_PULLUP;                /* 내부 풀업 활성화 */
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART3;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* Configure Baud rate 1000000 (1Mbps, Auto-calculated based on APB1 clock) */
  uint32_t pclk1 = HAL_RCC_GetPCLK1Freq();
  USART3->BRR = (pclk1 + (1000000 / 2)) / 1000000;
  
  /* CR3: HDSEL (Half-Duplex Selection) 비트 세트 */
  USART3->CR3 |= USART_CR3_HDSEL;
  
  /* CR1: 송수신 활성화 및 USART 활성화 */
  USART3->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

/* ================================================================== */
/*        USART2 RX 인터럽트 초기화 및 바이트 처리 루틴               */
/* ================================================================== */
static void MX_USART2_RxInt_Init(void)
{
  /* USART2 RXNE 인터럽트 활성화 */
  USART2->CR1 |= USART_CR1_RXNEIE;
  HAL_NVIC_SetPriority(USART2_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(USART2_IRQn);
}

void USART2_Process_Rx_Byte(uint8_t byte)
{
  char ch = (char)byte;

  /* 엔터(\r 또는 \n) → 명령 완성
   *
   * 이전 구현은 cmd_ready 가 1인 동안 들어오는 문자를 전부 버렸다. 메인
   * 루프가 직전 명령을 처리하는 사이에 도착한 줄이 통째로 사라져서,
   * 조이스틱처럼 20Hz 로 계속 보내면 명령이 끊기고 조각만 남았다.
   * 완성된 줄은 별도 버퍼로 옮기고 수집은 계속한다 — 아직 안 읽힌 줄이
   * 있으면 새 줄이 덮어쓴다(최신 우선). 오래된 속도 명령을 실행하는 쪽이
   * 최신 명령을 놓치는 것보다 위험하므로 이 방향이 맞다. */
  if (ch == '\r' || ch == '\n')
  {
    /* 빈 줄(CRLF 의 두 번째 문자 등)은 무시 — 에코도, 명령 확정도 하지 않는다 */
    if (cmd_idx > 0)
    {
      if (s_echo_enabled) { __io_putchar('\r'); __io_putchar('\n'); }
      cmd_buf[cmd_idx] = '\0';
      memcpy(cmd_line, cmd_buf, (size_t)cmd_idx + 1);
      cmd_ready = 1;
      cmd_idx = 0;
    }
  }
  else if (ch == '\b' || ch == 0x7F)  /* Backspace / DEL — 오타 수정 */
  {
    if (cmd_idx > 0)
    {
      cmd_idx--;
      if (s_echo_enabled) { __io_putchar('\b'); __io_putchar(' '); __io_putchar('\b'); }
    }
  }
  else if (cmd_idx < CMD_BUF_SIZE - 1)
  {
    cmd_buf[cmd_idx++] = ch;
    if (s_echo_enabled) __io_putchar(ch);
  }
}

/* ================================================================== */
/*              서보 부팅 시퀀스 — PING → 토크 ON → 준비              */
/* ================================================================== */
static void servo_boot_sequence(void)
{
  printf("\r\n--------------------------------------------------\r\n");
  printf("       STS3215 Servo Bus Initialization            \r\n");
  printf("--------------------------------------------------\r\n");
  printf("USART3: PB10 Half-Duplex BUS, 1,000,000 bps\r\n");
  printf("Servo Count: %d\r\n\r\n", SERVO_COUNT);

  /* 1. 각 서보 PING */
  uint8_t servo_ids[] = {1, 2, 3, 4};  /* SERVO_COUNT=4 기준 */
  uint8_t all_ok = 1;

  for (int i = 0; i < SERVO_COUNT; i++)
  {
    STS_Status ret = sts_ping(servo_ids[i]);
    if (ret == STS_OK)
    {
      printf("  Servo ID=%d: OK (응답 정상)\r\n", servo_ids[i]);
    }
    else
    {
      printf("  Servo ID=%d: FAIL (err=%d)\r\n", servo_ids[i], ret);
      all_ok = 0;
    }
  }

  if (!all_ok)
  {
    printf("\r\n[경고] 일부 서보 응답 없음. 배선/전원/ID 확인 필요.\r\n");
    printf("  점검: TX/RX 교차, GND 공통, 1Mbps 보레이트, 어댑터 모드\r\n");
  }

  /* 2. 관절 모듈 초기화 (가속도 설정 + 토크 ON) */
  joint_init();

  printf("\r\n>>> Servo System Ready! <<<\r\n");
  printf("터미널 명령어: T / P / S / ON / OFF / PING / HELP\r\n");
  printf("--------------------------------------------------\r\n\r\n");
}

/* ================================================================== */
/*      스워브 기구학 연동 — vx,vy,omega -> 조향 서보 + 구동 모터      */
/* ================================================================== */

/* last_angle_rad per module persists across calls (needed for the
 * <=90deg shortest-path steering optimization in swerve_ik_compute). */
static swerve_module_state_t s_swerve_state[SWERVE_NUM_MODULES];

/* 한 제어주기: 차체 명령 -> IK -> 바퀴별 가동범위로 접기 -> (RL 잔차) -> 정렬 게이트
 *            -> 조향 서보(SYNC_WRITE) + 구동 PWM.
 * 텔레옵이 시작된 뒤 매 CTRL_PERIOD_MS 마다 호출된다.
 * d_steer / wheel_gain (모듈 순서, NULL = 없음): SPI 타입 0x02 의 RL Res8 잔차.  접은 IK 각에 더하고
 * 가동범위로 자르며, IK 바퀴 속도에 곱하고 최고속으로 자른다 (시뮬 Res8Action 과 같은 순서).
 * IK 의 접기 상태(s_swerve_state)는 잔차 전 IK 값으로 남는다 -- 시뮬 BatchedSwerveIK 와 같다. */
static void Swerve_Control(float vx, float vy, float omega, const float *d_steer, const float *wheel_gain)
{
  swerve_module_cmd_t cmd[SWERVE_NUM_MODULES];
  float lo[SWERVE_NUM_MODULES], hi[SWERVE_NUM_MODULES];
  float goal[SWERVE_NUM_MODULES], speed[SWERVE_NUM_MODULES];

  swerve_ik_compute(vx, vy, omega, SWERVE_MODULES, SWERVE_MAX_WHEEL_MPS,
                     s_swerve_state, cmd);

  /* 바퀴별 가동범위(joint.c) 안의 등가각 중 직전 명령에 가장 가까운 것을 고른다.
   * 시뮬과 **같은 C 함수** (swerve_fold_to_limit 은 이 함수의 대칭 버전).
   * unwind 는 끈다(0) -- 시뮬 기본값과 같다. */
  for (int m = 0; m < SWERVE_NUM_MODULES; m++)
  {
    float min_deg, max_deg;
    joint_get_limits_deg(SWERVE_MODULE_SERVO[m], &min_deg, &max_deg);
    lo[m] = min_deg * PI / 180.0f;
    hi[m] = max_deg * PI / 180.0f;
  }
  swerve_fold_to_range(lo, hi, 0.0f, 0.0f, s_swerve_state, cmd);

  for (int m = 0; m < SWERVE_NUM_MODULES; m++)
  {
    goal[m] = cmd[m].angle_rad;
    speed[m] = cmd[m].speed_mps;
    if (d_steer != NULL && wheel_gain != NULL)
    {
      goal[m] += d_steer[m];
      if (goal[m] < lo[m]) goal[m] = lo[m];
      if (goal[m] > hi[m]) goal[m] = hi[m];
      speed[m] *= wheel_gain[m];
      if (speed[m] > SWERVE_MAX_WHEEL_MPS) speed[m] = SWERVE_MAX_WHEEL_MPS;
      if (speed[m] < -SWERVE_MAX_WHEEL_MPS) speed[m] = -SWERVE_MAX_WHEEL_MPS;
    }
    s_steer_goal_rad[m] = goal[m];
  }

  /* 조향 목표 전송 (서보 인덱스 순서로 재배열) */
  float servo_deg[SERVO_COUNT];
  for (int m = 0; m < SWERVE_NUM_MODULES; m++)
  {
    servo_deg[SWERVE_MODULE_SERVO[m]] = goal[m] * 180.0f / PI;
  }
  s_steer_clamped = (on_target_angles_received(servo_deg, SERVO_COUNT) > 0);

  /* 구동 정렬 게이트: 큰 방향 전환이면 실제 조향각이 정렬될 때까지 보류.
   * 실제각은 이번 주기 시작에 읽은 서보 피드백(s_servo_fb)을 쓴다. */
  /* 2026-10-13: allow 를 매 주기 부른다.  예전에는 pending 일 때만 불러서 drive_align_gate.h (b)
   * "실측 트리거" (40도씩 세 번 돌린 120도 전환) 가 한 번도 안 걸렸다.  시뮬 DriveGate 도 이 순서. */
  drive_gate_update_goal(&s_gate, goal, speed);
  float actual[SWERVE_NUM_MODULES];
  for (int m = 0; m < SWERVE_NUM_MODULES; m++)
  {
    actual[m] = s_servo_fb[m].angle_deg * PI / 180.0f;
  }
  int allow = drive_gate_allow(&s_gate, goal, actual, s_servo_valid == 0x0F);

  for (int m = 0; m < SWERVE_NUM_MODULES; m++)
  {
    int16_t duty = 0;
    MotorID mot = SWERVE_MODULE_MOTOR[m];
    if (!allow)
    {
      wheel_pi_reset(&s_wpi[m]);   /* 보류 중 적분이 쌓이면 풀리는 순간 튄다 */
    }
    else if (s_wpi_on)
    {
      duty = wheel_pi_step(&s_wpi_cfg, &s_wpi[m], speed[m] / WHEEL_RADIUS_M, s_wheel_w[mot]);
    }
    else
    {
      float norm = mdd3a_normalize_speed(speed[m], SWERVE_MAX_WHEEL_MPS);
      duty = (int16_t)(norm * (float)PWM_MAX);
    }
    Motor_SetSpeed(mot, (int16_t)(duty * MOTOR_DRIVE_SIGN[mot]));
    s_duty_applied[m] = duty;
  }

  /* 디버그 출력은 LOG 또는 CTRL 이 켜져 있을 때만, 5Hz, 한 줄.  115200 baud 에서
   * 한 줄(~90B)도 8ms 를 막으므로 제어주기와 같은 빈도로 찍으면 안 된다.
   * v 는 워치독 적용 후 실제로 IK 에 들어간 차체 명령 (받은 값 그대로). */
  static uint32_t last_print_ms = 0;
  uint32_t print_now = HAL_GetTick();
  if ((s_log_enabled || s_ctrl_log) && print_now - last_print_ms >= 200)
  {
    last_print_ms = print_now;
    printf("[CTRL] v %+.3f %+.3f %+.3f | deg %.0f %.0f %.0f %.0f | duty %d %d %d %d%s%s\r\n",
           vx, vy, omega,
           goal[0] * 180.0f / PI, goal[1] * 180.0f / PI,
           goal[2] * 180.0f / PI, goal[3] * 180.0f / PI,
           s_duty_applied[0], s_duty_applied[1], s_duty_applied[2], s_duty_applied[3],
           s_gate.pending ? " GATE" : "",
           s_teleop_timed_out ? " WDOG" : "");
  }
}

/* 구동 명령 수신 (UART 아스키 V 또는 SPI).  목표값과 시각만 갱신하고,
 * 실제 출력은 다음 제어주기에서 나간다. */
static void teleop_command(float vx, float vy, float omega, uint8_t from_spi)
{
  if (!isfinite(vx) || !isfinite(vy) || !isfinite(omega))
  {
    return;
  }
  /* 첫 명령에서 텔레옵으로 전환: 모터 자동 테스트 시퀀스를 멈추지
   * 않으면 그쪽이 Motor_SetSpeed 로 명령을 계속 덮어쓴다. */
  if (!s_teleop_active)
  {
    s_teleop_active = 1;
    Motor_StopAll();
    s_mtest_enabled = 0;
    /* 연속 수신 중 ISR 이 송신을 기다리면 오버런이 나므로 에코를 끈다 */
    s_echo_enabled = 0;
    /* 로그는 USART2 송신을 busy-wait 해서 10Hz 로그 한 번이 루프를 ~30ms 막는다.
     * 50Hz 제어주기를 지키려면 끈다 (필요하면 LOG 로 다시 켤 수 있다). */
    s_log_enabled = 0;
    printf("[텔레옵] 시작 (%s) — 모터 자가시험 중단, 에코·로그 OFF (LOG 로 다시 켜기)\r\n",
           from_spi ? "SPI" : "UART");
  }
  if (s_teleop_timed_out)
  {
    printf("[텔레옵] 명령 재개\r\n");
  }
  s_cmd_target[0] = vx;
  s_cmd_target[1] = vy;
  s_cmd_target[2] = omega;
  s_cmd_from_spi = from_spi;
  s_res_have = 0;   /* 잔차 없는 명령(UART V, SPI 0x01)이 오면 바로 순수 IK.  0x02 는 아래에서 다시 세움 */
  s_last_cmd_ms = HAL_GetTick();
  s_teleop_timed_out = 0;
}

/* 매 제어주기 센서 읽기: 조향 서보 4개 (위치·속도·부하·전압·온도·상태) + IMU.
 * 서보 한 개 ~0.7ms (대부분 서보 반환 지연 500us), IMU ~0.12ms. */
static void read_sensors(void)
{
  s_servo_valid = 0;
  int retried = 0;
  /* 서보 응답 수신 중 Pi SPI 프레임 끝 ISR 이 끼면 USART3 오버런 -> 읽기 실패 */
  PiSpi_HoldIrq(1);
  for (int m = 0; m < SWERVE_NUM_MODULES; m++)
  {
    if (s_servo_skip[m] > 0)
    {
      s_servo_skip[m]--;
      continue;
    }
    /* 실패했던 서보 재시도는 한 주기에 하나만 -- 서보 전원이 꺼져 있으면 넷이
     * 동시에 타임아웃(각 ~3ms)을 먹어 주기가 밀린다. 나머지는 다음 주기로. */
    if (s_servo_failed[m])
    {
      if (retried) continue;
      retried = 1;
    }
    if (joint_read_feedback(SWERVE_MODULE_SERVO[m], &s_servo_fb[m]))
    {
      s_servo_valid |= (uint8_t)(1u << m);
      s_servo_failed[m] = 0;
    }
    else
    {
      s_servo_failed[m] = 1;
      s_servo_skip[m] = SERVO_RETRY_TICKS;
    }
  }
  PiSpi_HoldIrq(0);
  s_servo_read_err = (s_servo_valid != 0x0F);

  s_imu_valid = (uint8_t)Imu_TakeMean(s_imu_raw, &s_imu_n);
}

/* SPI 로 들어온 프레임 처리 (메인 루프에서 호출) */
static void pi_spi_poll(void)
{
  const uint8_t *frame = PiSpi_TakeRxFrame();
  if (frame == NULL) return;

  spi_link_cmd_t c;
  if (spi_link_parse_cmd(frame, &c) != SPI_LINK_OK)
  {
    s_spi_rx_err++;
    return;
  }
  s_spi_seq_echo = c.seq;
  if (c.type == SPI_LINK_TYPE_VEL_CMD || c.type == SPI_LINK_TYPE_VEL_RES)
  {
    if (!isfinite(c.vx) || !isfinite(c.vy) || !isfinite(c.omega))
    {
      s_spi_rx_err++;
      return;
    }
    teleop_command(c.vx, c.vy, c.omega, 1);
  }
  if (c.type == SPI_LINK_TYPE_VEL_RES)
  {
    /* 잔차가 하나라도 NaN 이면 잔차만 버린다 (속도 명령은 위에서 이미 받음 -> 순수 IK) */
    for (int m = 0; m < SWERVE_NUM_MODULES; m++)
    {
      if (!isfinite(c.d_steer[m]) || !isfinite(c.wheel_gain[m])) return;
    }
    for (int m = 0; m < SWERVE_NUM_MODULES; m++)
    {
      float d = c.d_steer[m], g = c.wheel_gain[m];
      s_res_d[m] = (d > SPI_RES_STEER_MAX) ? SPI_RES_STEER_MAX : ((d < -SPI_RES_STEER_MAX) ? -SPI_RES_STEER_MAX : d);
      s_res_g[m] = (g > SPI_RES_GAIN_MAX) ? SPI_RES_GAIN_MAX : ((g < SPI_RES_GAIN_MIN) ? SPI_RES_GAIN_MIN : g);
    }
    s_res_ms = HAL_GetTick();
    s_res_have = 1;
  }
}

/* ================================================================== */
/*                     터미널 명령 보조 함수                           */
/* ================================================================== */

/** cmd 가 key 로 시작하는지 (대소문자 무시) */
static int cmd_is(const char *cmd, const char *key)
{
  while (*key)
  {
    char c = *cmd;
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    if (c != *key) return 0;
    cmd++;
    key++;
  }
  return 1;
}

/**
 * @brief  현재 물리적 자세를 0도(위치값 2048)로 영점 기록
 *
 * 토크 스위치 레지스터(0x28)에 128을 쓰면 서보가 현재 위치를 2048로
 * 보정하고 EEPROM에 저장합니다 (공식 메모리표: "写128：任意当前位置较正为2048").
 * 쓰기만 하면 되므로 위치 읽기(sts_read)에 의존하지 않습니다.
 */
static void servo_zero_here(uint8_t id)
{
  uint8_t v = 128;
  STS_Status ret = sts_write(id, STS_REG_TORQUE_ENABLE, &v, 1);
  printf("  ID=%u 영점 기록 %s\r\n", id, (ret == STS_OK) ? "성공" : "실패/무응답");
}

/* ================================================================== */
/*         CAP — 구동모터 고속 캡처 (모터 모델링, 바퀴 띄우고)          */
/* ================================================================== */
/* duty 프로파일을 모터에 넣으면서 엔코더 누적 틱을 일정 간격으로 RAM 에 기록하고,
 * 끝난 뒤 텍스트로 출력한다. 출력이 115200 baud 라 실시간 스트리밍은 안 된다.
 *
 *   CAP <1-4|ALL|234..> <d0> <d1> <t_sw> <t_end> [RAMP] [DEC <n>]
 *     d0, d1 : duty (-799..799), **부호 보정 적용: + = 바퀴 전진** (MOT 와 다름)
 *     t_sw   : 이 시각(ms)에 d0 -> d1 계단.  RAMP 면 0..t_sw 동안 d0 -> d1 직선
 *     t_end  : 기록 길이 (ms).  샘플 수 = t_end / n <= CAP_MAX_SAMPLES
 *     DEC n  : 샘플 간격 n ms (기본 1)
 *   d0 != 0 이면 기록 전에 d0 로 CAP_SETTLE_MS 돌려 정상상태를 만든다.
 *   끝나면 모터 정지 (MDD3A 입력 둘 다 LOW = 브레이크).
 *
 * 출력 (pi/common/sysid_uart.py 가 파싱):
 *   [CAP] BEGIN motors=<비트> d0=.. d1=.. t_sw=.. t_end=.. ramp=.. dt_us=.. n=.. ticks_per_rev=..
 *   t_us,dp1,dp2,dp3,dp4   줄마다. dp = 직전 샘플 대비 틱 증가 (부호 보정 적용, 선택 안 한 모터는 0).
 *                          모터 번호 = MOTOR1..4 (MOT 와 같음, 모듈 순서 아님)
 *   [CAP] END late_max_us=.. vs_dv=..  샘플 시각이 예정보다 가장 늦은 정도, 끝 시점(d1 부하) 서보1 전압 0.1V 단위 (-1 = 읽기 실패)
 * 샘플 시각은 DWT 사이클 카운터 기준. 첫 샘플(t=0) 은 d0 상태이고, duty 는 각 샘플을
 * 찍은 직후 다음 구간 값으로 바뀐다. */
#define CAP_MAX_SAMPLES  2000
#define CAP_SETTLE_MS    800

static int32_t  s_cap_dp[CAP_MAX_SAMPLES][MOTOR_COUNT];
static uint32_t s_cap_t_us[CAP_MAX_SAMPLES];

static void cap_apply(uint8_t mask, int duty)
{
  for (int i = 0; i < MOTOR_COUNT; i++)
  {
    if (mask & (1u << i)) Motor_SetSpeed((MotorID)i, (int16_t)(duty * MOTOR_DRIVE_SIGN[i]));
  }
}

static void run_capture(const char *arg)
{
  while (*arg == ' ') arg++;
  uint8_t mask = 0;
  if (cmd_is(arg, "ALL"))
  {
    mask = (uint8_t)((1u << MOTOR_COUNT) - 1u);
    arg += 3;
  }
  else
  {
    /* 숫자를 붙여 쓰면 여러 개: CAP 234 = 모터 2·3·4 */
    while (*arg >= '1' && *arg <= '0' + MOTOR_COUNT)
    {
      mask |= (uint8_t)(1u << (*arg - '1'));
      arg++;
    }
    if (*arg != ' ') mask = 0;
  }
  int d0 = 0, d1 = 0, t_sw = 0, t_end = 0, n_read = 0;
  if (mask) n_read = sscanf(arg, "%d %d %d %d", &d0, &d1, &t_sw, &t_end);
  const int ramp = (strstr(arg, "RAMP") != NULL) || (strstr(arg, "ramp") != NULL);
  int dec = 1;
  const char *pd = strstr(arg, "DEC");
  if (pd == NULL) pd = strstr(arg, "dec");
  if (pd != NULL && sscanf(pd + 3, "%d", &dec) != 1) dec = 0;

  const int ok = (mask != 0) && (n_read == 4) && dec >= 1 && dec <= 50 &&
                 d0 >= -PWM_MAX && d0 <= PWM_MAX && d1 >= -PWM_MAX && d1 <= PWM_MAX &&
                 t_sw >= 0 && t_sw <= t_end && t_end >= dec && (t_end / dec) <= CAP_MAX_SAMPLES;
  if (!ok)
  {
    printf("[CAP] 사용법: CAP <1-4|ALL|234..> <d0> <d1> <t_sw_ms> <t_end_ms> [RAMP] [DEC <n_ms>]\r\n");
    printf("      duty -%d~+%d (+ = 바퀴 전진), t_end/n <= %d.  ** 바퀴를 띄운 상태에서 **\r\n",
           PWM_MAX, PWM_MAX, CAP_MAX_SAMPLES);
    printf("      예) CAP ALL 0 400 100 600        (0->400 계단 응답)\r\n");
    printf("          CAP ALL 799 0 100 1500       (최고속에서 정지 = 제동 감속)\r\n");
    printf("          CAP 1 0 150 2000 2000 RAMP   (0->150 천천히 = 데드밴드)\r\n");
    printf("          CAP 234 0 400 100 600        (모터 2·3·4 만 동시에)\r\n");
    return;
  }

  s_mtest_enabled = 0;
  Motor_StopAll();
  if (d0 != 0)
  {
    cap_apply(mask, d0);
    HAL_Delay(CAP_SETTLE_MS);
  }

  const int n = t_end / dec;
  const uint32_t cyc_per_us = SystemCoreClock / 1000000u;
  const uint32_t step_cyc = (uint32_t)dec * 1000u * cyc_per_us;
  int32_t last[MOTOR_COUNT];
  for (int i = 0; i < MOTOR_COUNT; i++) last[i] = Encoder_GetCount((MotorID)i);
  uint32_t late_max = 0;
  int cur_duty = d0;
  const uint32_t c0 = DWT->CYCCNT;

  for (int k = 0; k < n; k++)
  {
    const uint32_t due = (uint32_t)k * step_cyc;
    while ((DWT->CYCCNT - c0) < due) { }
    const uint32_t now = DWT->CYCCNT - c0;
    if (now - due > late_max) late_max = now - due;
    s_cap_t_us[k] = now / cyc_per_us;
    for (int i = 0; i < MOTOR_COUNT; i++)
    {
      const int32_t c = Encoder_GetCount((MotorID)i);
      int32_t d = c - last[i];
      if (!Encoder_IsWide((MotorID)i)) d = (int32_t)(int16_t)d;   /* 16비트 타이머 감김 */
      last[i] = c;
      s_cap_dp[k][i] = (mask & (1u << i)) ? d * ENCODER_SIGN[i] : 0;
    }
    const int t_next = (k + 1) * dec;
    int want = d1;
    if (t_next < t_sw)
    {
      want = ramp ? d0 + (int)((long)(d1 - d0) * t_next / t_sw) : d0;
    }
    if (want != cur_duty)
    {
      cap_apply(mask, want);
      cur_duty = want;
    }
  }
  /* 아직 d1 으로 도는 중(부하 상태)에 서보1 전원 전압을 읽는다. 반복 측정 사이 배터리 처짐 확인용
   * (서보와 구동이 같은 배터리일 때만 배터리 지표).  읽기 ~0.7 ms, 샘플링이 끝난 뒤라 기록엔 영향 없음. */
  int vs_dv = -1;
  {
    JointFeedback_t fb;
    PiSpi_HoldIrq(1);
    if (joint_read_feedback(0, &fb)) vs_dv = fb.voltage;
    PiSpi_HoldIrq(0);
  }
  Motor_StopAll();

  printf("[CAP] BEGIN motors=%u d0=%d d1=%d t_sw=%d t_end=%d ramp=%d dt_us=%d n=%d ticks_per_rev=%d\r\n",
         (unsigned)mask, d0, d1, t_sw, t_end, ramp, dec * 1000, n, (int)TICKS_PER_REV);
  for (int k = 0; k < n; k++)
  {
    printf("%lu,%ld,%ld,%ld,%ld\r\n", (unsigned long)s_cap_t_us[k], (long)s_cap_dp[k][0],
           (long)s_cap_dp[k][1], (long)s_cap_dp[k][2], (long)s_cap_dp[k][3]);
  }
  printf("[CAP] END late_max_us=%lu vs_dv=%d\r\n", (unsigned long)(late_max / cyc_per_us), vs_dv);
}

/* ================================================================== */
/*   ICAP — IMU 208 Hz 원시 샘플 기록 (진동·필터 측정, 주행 중에도)       */
/* ================================================================== */
/* CAP 과 달리 제어 루프를 막지 않는다: 메인 루프의 Imu_Poll 이 새 샘플을 읽을 때마다 RAM 에 쌓고,
 * 끝난 뒤 ICAP DUMP 로 꺼낸다 (덤프는 ~7 s 동안 루프를 막으므로 정지 상태에서만).
 *   ICAP [ms]   기록 시작 (기본 9000, 최대 CAP_MAX_SAMPLES 샘플 ≈ 9.6 s).  ICAP 만 치면 상태
 *   ICAP DUMP   [ICAP] BEGIN n=.. odr_hz=208 gyro_dps_lsb=0.0175 acc_g_lsb=0.000122
 *               t_us,gx,gy,gz,ax,ay,az   칩 좌표계 원시값 (Pi nucleo_link 의 IMU_CHIP_TO_BODY 로 몸체 좌표)
 *               [ICAP] END gaps=..       샘플 간격이 7 ms 넘은 횟수 (서보 읽기 등으로 놓친 샘플)
 * 버퍼는 CAP 것을 같이 쓴다 (한 줄 int32 x4 = 16 B 에 int16 x6). */
static uint8_t  s_icap_rec = 0;
static int      s_icap_n = 0;
static uint32_t s_icap_c0 = 0, s_icap_end_us = 0;

static void icap_store(const int16_t smp[6])
{
  const uint32_t t_us = (DWT->CYCCNT - s_icap_c0) / (SystemCoreClock / 1000000u);
  if (s_icap_n >= CAP_MAX_SAMPLES || t_us > s_icap_end_us)
  {
    s_icap_rec = 0;
    printf("[ICAP] 기록 끝 (%d 샘플). ICAP DUMP 로 꺼내기\r\n", s_icap_n);
    return;
  }
  s_cap_t_us[s_icap_n] = t_us;
  memcpy(s_cap_dp[s_icap_n], smp, 6 * sizeof smp[0]);
  s_icap_n++;
}

static void run_icap(const char *arg)
{
  while (*arg == ' ') arg++;
  if (cmd_is(arg, "DUMP"))
  {
    if (s_icap_rec)
    {
      printf("[ICAP] 아직 기록 중 (%d 샘플)\r\n", s_icap_n);
      return;
    }
    if (s_teleop_active && !s_teleop_timed_out &&
        (s_cmd_applied[0] != 0.0f || s_cmd_applied[1] != 0.0f || s_cmd_applied[2] != 0.0f))
    {
      printf("[ICAP] 주행 명령 중에는 덤프 안 함 (~7 s 루프 정지). 멈춘 뒤 다시\r\n");
      return;
    }
    int gaps = 0;
    printf("[ICAP] BEGIN n=%d odr_hz=208 gyro_dps_lsb=0.0175 acc_g_lsb=0.000122\r\n", s_icap_n);
    for (int k = 0; k < s_icap_n; k++)
    {
      const int16_t *v = (const int16_t *)s_cap_dp[k];
      if (k > 0 && s_cap_t_us[k] - s_cap_t_us[k - 1] > 7000u) gaps++;
      printf("%lu,%d,%d,%d,%d,%d,%d\r\n", (unsigned long)s_cap_t_us[k], v[0], v[1], v[2], v[3], v[4], v[5]);
    }
    printf("[ICAP] END gaps=%d\r\n", gaps);
    return;
  }
  if (*arg == '\0' && s_icap_rec)
  {
    printf("[ICAP] 기록 중 %d 샘플\r\n", s_icap_n);
    return;
  }
  int ms = 9000;
  if (*arg != '\0' && (sscanf(arg, "%d", &ms) != 1 || ms < 100 || ms > 60000))
  {
    printf("[ICAP] 사용법: ICAP [ms 100-60000] (기본 9000) / ICAP DUMP\r\n");
    return;
  }
  s_icap_n = 0;
  s_icap_c0 = DWT->CYCCNT;
  s_icap_end_us = (uint32_t)ms * 1000u;
  s_icap_rec = 1;
  printf("[ICAP] 기록 시작 %d ms (최대 %d 샘플 ≈ %.1f s)\r\n", ms, CAP_MAX_SAMPLES, CAP_MAX_SAMPLES / 208.0f);
}

/* ================================================================== */
/*       WCAP — 바퀴 속도 PI 계단 캡처 (PI 벤치 검증, 바퀴 띄우고)       */
/* ================================================================== */
/* CAP 과 같지만 duty 대신 바퀴 속도 목표 [rad/s] 를 주고, 제어주기(20 ms)마다 wheel_pi_step 으로
 * duty 를 정한다 (텔레옵과 같은 s_wpi_cfg, WPI ON/OFF 와 무관하게 PI).  엔코더 기록은 CAP 과 같다.
 *
 *   WCAP <1-4|ALL|234..> <w0> <w1> <t_sw> <t_end> [RAMP] [DEC <n>]
 *     w0, w1 : 바퀴 속도 rad/s (소수 가능, + = 전진, |w| <= 8)
 *   w0 != 0 이면 기록 전에 w0 로 CAP_SETTLE_MS 돌린다 (PI 정착).
 *
 * 출력 (pi/common/sysid_uart.py 가 파싱):
 *   [WCAP] BEGIN motors=.. w0_mrad=.. w1_mrad=.. t_sw=.. t_end=.. ramp=.. dt_us=.. n=.. ticks_per_rev=.. pi_ms=20
 *   t_us,dp1,dp2,dp3,dp4      CAP 과 같음 (MOTOR1..4 순서)
 *   D,t_ms,d1,d2,d3,d4        PI 주기마다 이번에 건 duty (+ = 전진, 모터 순서)
 *   [WCAP] END late_max_us=.. vs_dv=..                                                        */
#define WCAP_MAX_PI  (CAP_MAX_SAMPLES / 20 + 1)
static int16_t  s_wcap_duty[WCAP_MAX_PI][MOTOR_COUNT];
static uint16_t s_wcap_t_ms[WCAP_MAX_PI];

/* 한 PI 주기: 직전 주기 이후 엔코더 차분으로 속도를 재고 duty 를 건다. last 는 모터별 카운터. */
static void wcap_pi_tick(uint8_t mask, float w_ref, int32_t last[MOTOR_COUNT], wheel_pi_state_t st[MOTOR_COUNT],
                         int16_t out[MOTOR_COUNT])
{
  for (int i = 0; i < MOTOR_COUNT; i++)
  {
    out[i] = 0;
    if (!(mask & (1u << i))) continue;
    const int32_t c = Encoder_GetCount((MotorID)i);
    int32_t d = c - last[i];
    if (!Encoder_IsWide((MotorID)i)) d = (int32_t)(int16_t)d;
    last[i] = c;
    const float w = (float)(d * ENCODER_SIGN[i]) / TICKS_PER_REV * 2.0f * PI / s_wpi_cfg.dt;
    out[i] = wheel_pi_step(&s_wpi_cfg, &st[i], w_ref, w);
    Motor_SetSpeed((MotorID)i, (int16_t)(out[i] * MOTOR_DRIVE_SIGN[i]));
  }
}

static uint8_t parse_motor_mask(const char **parg)
{
  const char *arg = *parg;
  while (*arg == ' ') arg++;
  uint8_t mask = 0;
  if (cmd_is(arg, "ALL"))
  {
    mask = (uint8_t)((1u << MOTOR_COUNT) - 1u);
    arg += 3;
  }
  else
  {
    while (*arg >= '1' && *arg <= '0' + MOTOR_COUNT)
    {
      mask |= (uint8_t)(1u << (*arg - '1'));
      arg++;
    }
    if (*arg != ' ') mask = 0;
  }
  *parg = arg;
  return mask;
}

static void run_wcap(const char *arg)
{
  uint8_t mask = parse_motor_mask(&arg);
  float v[4] = {0};
  int n_read = 0;
  if (mask)
  {
    const char *p = arg;
    for (; n_read < 4; n_read++)          /* nano.specs 라 sscanf %f 가 안 된다 -> strtof */
    {
      char *endp;
      v[n_read] = strtof(p, &endp);
      if (endp == p) break;
      p = endp;
    }
  }
  const float w0 = v[0], w1 = v[1];
  const int t_sw = (int)v[2], t_end = (int)v[3];
  const int ramp = (strstr(arg, "RAMP") != NULL) || (strstr(arg, "ramp") != NULL);
  int dec = 1;
  const char *pd = strstr(arg, "DEC");
  if (pd == NULL) pd = strstr(arg, "dec");
  if (pd != NULL && sscanf(pd + 3, "%d", &dec) != 1) dec = 0;
  const int pi_ms = CTRL_PERIOD_MS;

  const int ok = (mask != 0) && (n_read == 4) && dec >= 1 && dec <= 20 && (pi_ms % dec) == 0 &&
                 fabsf(w0) <= 8.0f && fabsf(w1) <= 8.0f &&
                 t_sw >= 0 && t_sw <= t_end && t_end >= dec && (t_end / dec) <= CAP_MAX_SAMPLES;
  if (!ok)
  {
    printf("[WCAP] 사용법: WCAP <1-4|ALL|234..> <w0> <w1> <t_sw_ms> <t_end_ms> [RAMP] [DEC <n_ms>]\r\n");
    printf("       바퀴 속도 rad/s (|w|<=8, + = 전진), PI 20 ms (n 은 20 의 약수), t_end/n <= %d. ** 바퀴 띄우고 **\r\n",
           CAP_MAX_SAMPLES);
    printf("       예) WCAP ALL 0 2 100 600       (0->2 rad/s 계단)\r\n");
    printf("           WCAP ALL 0 0.3 100 1500    (저속 기동: 개루프면 정지마찰에 막히는 속도)\r\n");
    return;
  }

  s_mtest_enabled = 0;
  Motor_StopAll();
  wheel_pi_state_t st[MOTOR_COUNT];
  int32_t last_pi[MOTOR_COUNT], last[MOTOR_COUNT];
  int16_t duty[MOTOR_COUNT];
  for (int i = 0; i < MOTOR_COUNT; i++)
  {
    wheel_pi_clear_fault(&st[i]);   /* 지역 변수: 방향 불일치 상태까지 0 으로 */
    last_pi[i] = Encoder_GetCount((MotorID)i);
  }
  const uint32_t cyc_per_us = SystemCoreClock / 1000000u;
  if (w0 != 0.0f)
  {
    const uint32_t c_s = DWT->CYCCNT;
    for (int k = 0; k < CAP_SETTLE_MS / pi_ms; k++)
    {
      while ((DWT->CYCCNT - c_s) < (uint32_t)k * (uint32_t)pi_ms * 1000u * cyc_per_us) { }
      wcap_pi_tick(mask, w0, last_pi, st, duty);
    }
  }

  const int n = t_end / dec;
  const uint32_t step_cyc = (uint32_t)dec * 1000u * cyc_per_us;
  for (int i = 0; i < MOTOR_COUNT; i++) last[i] = Encoder_GetCount((MotorID)i);
  uint32_t late_max = 0;
  int n_pi = 0;
  const uint32_t c0 = DWT->CYCCNT;
  for (int k = 0; k < n; k++)
  {
    const uint32_t due = (uint32_t)k * step_cyc;
    while ((DWT->CYCCNT - c0) < due) { }
    const uint32_t now = DWT->CYCCNT - c0;
    if (now - due > late_max) late_max = now - due;
    s_cap_t_us[k] = now / cyc_per_us;
    for (int i = 0; i < MOTOR_COUNT; i++)
    {
      const int32_t c = Encoder_GetCount((MotorID)i);
      int32_t d = c - last[i];
      if (!Encoder_IsWide((MotorID)i)) d = (int32_t)(int16_t)d;
      last[i] = c;
      s_cap_dp[k][i] = (mask & (1u << i)) ? d * ENCODER_SIGN[i] : 0;
    }
    const int t_ms = k * dec;
    if ((t_ms % pi_ms) == 0 && n_pi < WCAP_MAX_PI)
    {
      float w = w1;
      if (t_ms < t_sw) w = ramp ? w0 + (w1 - w0) * (float)t_ms / (float)t_sw : w0;
      wcap_pi_tick(mask, w, last_pi, st, duty);
      s_wcap_t_ms[n_pi] = (uint16_t)t_ms;
      for (int i = 0; i < MOTOR_COUNT; i++) s_wcap_duty[n_pi][i] = duty[i];
      n_pi++;
    }
  }
  int vs_dv = -1;
  {
    JointFeedback_t fb;
    PiSpi_HoldIrq(1);
    if (joint_read_feedback(0, &fb)) vs_dv = fb.voltage;
    PiSpi_HoldIrq(0);
  }
  Motor_StopAll();

  printf("[WCAP] BEGIN motors=%u w0_mrad=%d w1_mrad=%d t_sw=%d t_end=%d ramp=%d dt_us=%d n=%d ticks_per_rev=%d pi_ms=%d\r\n",
         (unsigned)mask, (int)lroundf(w0 * 1000.0f), (int)lroundf(w1 * 1000.0f), t_sw, t_end, ramp, dec * 1000, n,
         (int)TICKS_PER_REV, pi_ms);
  for (int k = 0; k < n; k++)
  {
    printf("%lu,%ld,%ld,%ld,%ld\r\n", (unsigned long)s_cap_t_us[k], (long)s_cap_dp[k][0],
           (long)s_cap_dp[k][1], (long)s_cap_dp[k][2], (long)s_cap_dp[k][3]);
  }
  for (int j = 0; j < n_pi; j++)
  {
    printf("D,%u,%d,%d,%d,%d\r\n", (unsigned)s_wcap_t_ms[j], s_wcap_duty[j][0], s_wcap_duty[j][1],
           s_wcap_duty[j][2], s_wcap_duty[j][3]);
  }
  printf("[WCAP] END late_max_us=%lu vs_dv=%d\r\n", (unsigned long)(late_max / cyc_per_us), vs_dv);
}

/* ================================================================== */
/*              터미널 명령 파서 — 줄 단위 텍스트 처리                 */
/* ================================================================== */
/* 조향 서보 계단응답 측정 (STEP 명령). 한 서보를 목표각으로 보내고 피드백을 최대한 빨리
 * (~0.7ms 간격) 읽어 시작 지연, 10->90% 시간, 최고 속도, 도달 시간, 오버슈트를 찍는다.
 * 나머지 서보는 현재 각도로 다시 보내 제자리를 지킨다. 텔레옵 중에는 쓰지 않는다.
 *
 * raw=1 (STEP <n> <deg> RAW [MS <ms>]) : 궤적도 덤프한다 (모델링 #13·#15, pi/common/sysid_uart.py).
 *   1 ms 격자로 최대 CAP_MAX_SAMPLES 개를 CAP 버퍼에 기록 (CAP 과 동시에 안 쓰므로 공유).
 *   변화가 2도 미만인 작은 계단(데드밴드·백래시)은 RAW 에서만 허용하고, ±1도 도달 판정 대신
 *   ms(기본 800) 동안 고정 기록한다.  큰 계단도 MS 를 주면 고정 기록 (멈춘 위치 측정용).
 *   출력 (요약 줄들 뒤, 마지막 "읽기" 줄 앞):
 *     [STEPRAW] BEGIN servo=n start_cdeg=.. target_cdeg=.. n=.. acc=.. dz_cw=.. dz_ccw=..
 *     t_us,angle_cdeg,speed_dps,load     (각도 0.01도, 속도 도/s, 부하 0.1%; 관절 기준 부호)
 *     [STEPRAW] END
 *   acc / dz_* 는 서보 레지스터 0x29 / 0x1A·0x1B 를 그 자리에서 읽은 값 (-1 = 실패). */
#define STEP_SMALL_MIN_DEG  0.08f       /* 1 스텝 = 0.088도 */

static void steer_step_response(int idx, float target, int raw, int ms_fixed)
{
  float lo, hi;
  joint_get_limits_deg(idx, &lo, &hi);
  if (target < lo) target = lo;
  if (target > hi) target = hi;

  float ang[SERVO_COUNT];
  JointFeedback_t fb;
  for (int i = 0; i < SERVO_COUNT; i++)
  {
    PiSpi_HoldIrq(1);
    int ok = joint_read_feedback(i, &fb);
    PiSpi_HoldIrq(0);
    if (!ok)
    {
      printf("[STEP] 서보%d 읽기 실패 -- 서보 전원·배선 확인\r\n", i + 1);
      return;
    }
    ang[i] = fb.angle_deg;
  }
  const float start = ang[idx];
  const float delta = target - start;
  const int small = fabsf(delta) < 2.0f;
  if ((small && !raw) || fabsf(delta) < STEP_SMALL_MIN_DEG)
  {
    printf("[STEP] 이미 목표 근처 (현재 %+.1f도). 다른 각도로 시험%s\r\n", start,
           raw ? "" : " (2도 미만은 RAW 로)");
    return;
  }
  int reg_acc = -1, dz_cw = -1, dz_ccw = -1;
  if (raw)
  {
    uint8_t b[2];
    PiSpi_HoldIrq(1);
    if (sts_read((uint8_t)(idx + 1), STS_REG_ACCELERATION, 1, b) == STS_OK) reg_acc = b[0];
    if (sts_read((uint8_t)(idx + 1), 0x1A, 2, b) == STS_OK) { dz_cw = b[0]; dz_ccw = b[1]; }
    PiSpi_HoldIrq(0);
  }
  /* MS 지정(RAW) 이면 큰 계단도 고정 시간 기록 (멈춘 위치를 재려고).  아니면 작은 계단 800 ms, 큰 계단은 ±1도 도달 시 끝 */
  const int fixed = raw && (small || ms_fixed > 0);
  const float t_limit = fixed ? (ms_fixed > 0 ? (float)ms_fixed : 800.0f) : 3000.0f;

  const uint16_t speed_set = joint_get_speed();
  const float cyc_per_ms = (float)SystemCoreClock / 1000.0f;
  ang[idx] = target;
  const uint32_t c0 = DWT->CYCCNT;
  on_target_angles_received(ang, SERVO_COUNT);

  float t10 = -1.0f, t90 = -1.0f, t_settle = -1.0f, band_since = -1.0f, t_move = -1.0f;
  float peak_reg = 0.0f, peak_pos = 0.0f, overshoot = 0.0f, t = 0.0f;
  float d_t = 0.0f, d_a = start, last = start;
  const float move_th = fminf(0.5f, 0.5f * fabsf(delta));
  int n = 0, fail = 0, nraw = 0;
  while (1)
  {
    PiSpi_HoldIrq(1);
    int ok = joint_read_feedback(idx, &fb);
    PiSpi_HoldIrq(0);
    t = (float)(DWT->CYCCNT - c0) / cyc_per_ms;
    if (t > t_limit) break;
    if (!ok) { fail++; continue; }
    n++;
    const float a = fb.angle_deg;
    last = a;
    if (raw && nraw < CAP_MAX_SAMPLES && t >= (float)nraw)      /* 1 ms 격자 */
    {
      s_cap_t_us[nraw] = (uint32_t)(t * 1000.0f);
      s_cap_dp[nraw][0] = (int32_t)lroundf(a * 100.0f);
      s_cap_dp[nraw][1] = (int32_t)lroundf((float)fb.speed * 360.0f / 4096.0f);
      s_cap_dp[nraw][2] = fb.load;
      nraw++;
    }
    const float prog = (a - start) / delta;          /* 0 -> 1 */
    if (t_move < 0.0f && fabsf(a - start) >= move_th) t_move = t;
    if (t10 < 0.0f && prog >= 0.1f) t10 = t;
    if (t90 < 0.0f && prog >= 0.9f) t90 = t;
    const float reg = fabsf((float)fb.speed) * 360.0f / 4096.0f;
    if (reg > peak_reg) peak_reg = reg;
    if (t - d_t >= 10.0f)                              /* 위치 미분은 10ms 창 */
    {
      const float v = fabsf(a - d_a) / ((t - d_t) / 1000.0f);
      if (v > peak_pos) peak_pos = v;
      d_t = t;
      d_a = a;
    }
    const float over = (prog - 1.0f) * fabsf(delta);
    if (over > overshoot) overshoot = over;
    if (fixed) continue;                               /* 고정 시간 기록 */
    if (fabsf(a - target) <= 1.0f)
    {
      if (band_since < 0.0f) band_since = t;
      else if (t - band_since >= 100.0f) { t_settle = band_since; break; }
    }
    else
    {
      band_since = -1.0f;
    }
  }

  printf("[STEP] 서보%d %+.1f -> %+.1f도 (변화 %+.1f), 속도설정 S %u%s\r\n",
         idx + 1, start, target, delta, speed_set, speed_set == 0 ? " (최대)" : "");
  if (t_move >= 0.0f)
    printf("       시작 지연(%.2f도 움직일 때까지): %.1fms\r\n", move_th, t_move);
  else
    printf("       안 움직임 (%.0fms 동안 %.2f도 미만). 최종 %+.2f도\r\n", t_limit, move_th, last);
  if (fixed)
  {
    printf("       고정 기록 %.0fms: 최종 %+.2f도 (목표 대비 %+.2f)\r\n", t_limit, last, last - target);
  }
  if (!small)
  {
    if (t10 >= 0.0f && t90 > t10)
    {
      printf("       10->90%%: %.0fms -> 평균 %.0f도/s\r\n",
             t90 - t10, 0.8f * fabsf(delta) / ((t90 - t10) / 1000.0f));
    }
    else
    {
      printf("       90%% 에 못 감 (3초 안). 서보가 막혔거나 토크 OFF?\r\n");
    }
    printf("       최고속도: 위치미분 %.0f도/s, 서보 보고값 %.0f도/s\r\n", peak_pos, peak_reg);
    if (t_settle >= 0.0f)
      printf("       도달(±1도 100ms 유지): %.0fms, 오버슈트 %.1f도, 최종 %+.1f도\r\n", t_settle, overshoot, last);
    else if (!fixed)
      printf("       ±1도 안에 안정 못 함 (3초). 최종 %+.1f도, 오버슈트 %.1f도\r\n", last, overshoot);
  }
  if (raw)
  {
    printf("[STEPRAW] BEGIN servo=%d start_cdeg=%ld target_cdeg=%ld n=%d acc=%d dz_cw=%d dz_ccw=%d\r\n",
           idx + 1, (long)lroundf(start * 100.0f), (long)lroundf(target * 100.0f), nraw, reg_acc, dz_cw, dz_ccw);
    for (int k = 0; k < nraw; k++)
      printf("%lu,%ld,%ld,%ld\r\n", (unsigned long)s_cap_t_us[k], (long)s_cap_dp[k][0],
             (long)s_cap_dp[k][1], (long)s_cap_dp[k][2]);
    printf("[STEPRAW] END\r\n");
  }
  printf("       읽기 %d회 (평균 %.2fms 간격), 실패 %d\r\n", n, n > 0 ? t / (float)n : 0.0f, fail);
}

static void process_command(const char *cmd)
{
  /* --- "V <vx> <vy> <omega>" : 스워브 기구학 명령 (body-frame m/s, rad/s) --- */
  if (cmd[0] == 'V' || cmd[0] == 'v')
  {
    /* nano.specs 에 -u _scanf_float 가 없어서 sscanf("%f") 는 동작하지 않음.
     * strtof 는 플래그 없이 쓸 수 있음 (T 명령과 동일한 방식). */
    float v[3] = {0.0f, 0.0f, 0.0f};
    int parsed = 0;
    const char *p = cmd + 1;
    for (int i = 0; i < 3; i++)
    {
      char *endp;
      float val = strtof(p, &endp);
      if (endp == p) break;
      v[i] = val;
      parsed++;
      p = endp;
    }

    if (parsed == 3)
    {
      teleop_command(v[0], v[1], v[2], 0);
    }
    else
    {
      printf("[CMD] 사용법: V <vx> <vy> <omega>  (예: V 0.1 0 0)\r\n");
    }
  }
  /* --- "T 10.5 -20" : 목표 각도 설정 --- */
  else if (cmd[0] == 'T' || cmd[0] == 't')
  {
    float angles[SERVO_COUNT] = {0};
    int parsed = 0;

    /* "T" 이후의 숫자들을 파싱 */
    const char *p = cmd + 1;
    for (int i = 0; i < SERVO_COUNT; i++)
    {
      char *endp;
      float val = strtof(p, &endp);
      if (endp == p) break;  /* 더 이상 숫자 없음 */
      angles[i] = val;
      parsed++;
      p = endp;
    }

    if (parsed > 0)
    {
      printf("[CMD] 목표 각도:");
      for (int i = 0; i < parsed; i++)
      {
        printf(" %.1f", angles[i]);
      }
      printf(" (도)\r\n");

      int clamped = on_target_angles_received(angles, parsed);
      if (clamped > 0)
      {
        printf("[CMD] 경고: %d개 각도가 가동범위(joint.c min/max_deg)로 잘림\r\n", clamped);
      }
      if (s_teleop_active)
      {
        printf("[CMD] 텔레옵 중에는 다음 제어주기가 조향을 덮어씁니다\r\n");
      }
    }
    else
    {
      printf("[CMD] 사용법: T <각1> <각2> <각3> <각4>  (예: T 45 -45 45 -45)\r\n");
    }
  }
  /* --- "PING" : 연결 확인 (ID 1~SERVO_COUNT) ---
   * 주의: 반드시 단일 문자 "P" 분기보다 먼저 와야 함. 뒤에 두면 'P'가
   * "PING"을 먼저 잡아채서 이 분기에 영영 도달하지 못함. */
  else if (cmd_is(cmd, "PING"))
  {
    for (int i = 0; i < SERVO_COUNT; i++)
    {
      STS_Status ret = sts_ping(i + 1);
      printf("  Servo ID=%d: %s\r\n", i + 1, (ret == STS_OK) ? "OK" : "FAIL");
    }
  }
  /* --- "P" : 현재 각도 출력 --- */
  else if (cmd[0] == 'P' || cmd[0] == 'p')
  {
    printf("[현재 각도]\r\n");
    for (int i = 0; i < SERVO_COUNT; i++)
    {
      float angle = joint_get_angle(i);
      int16_t pos = sts_read_position(i + 1);  /* ID = i+1 */
      printf("  Servo %d: %.1f deg (raw pos=%d)\r\n", i + 1, angle, pos);
    }
  }
  /* --- "SCAN" : 버스 전체 스캔 (ID 1~253) ---
   * PING 과 마찬가지로 단일 문자 "S"(속도) 분기보다 먼저 와야 함. */
  else if (cmd_is(cmd, "SCAN"))
  {
    /* 응답 없는 ID마다 수신 타임아웃(15ms)을 기다리므로 최대 ~4초간
     * 이 루프가 메인 루프를 막음. 그동안 모터가 계속 돌지 않도록 정지. */
    Motor_StopAll();

    printf("[SCAN] 버스 스캔 (ID 1~253), 약 1초 소요...\r\n");
    int found = 0;
    for (int id = 1; id <= 253; id++)
    {
      if (sts_ping((uint8_t)id) == STS_OK)
      {
        printf("  ID=%3d : 응답 OK\r\n", id);
        found++;
      }
      if ((id % 64) == 0) printf("  ... %d 까지 검사\r\n", id);
    }
    printf("[SCAN] 완료 — 서보 %d개 발견\r\n", found);
    if (found == 0)
    {
      printf("  배선(PB10/GND), 서보 전원(12V), 보레이트 1Mbps 확인 필요\r\n");
    }
  }
  /* --- "SPI" : Pi 링크 배선 진단 (단일 문자 "S" 분기보다 먼저) --- */
  else if (cmd_is(cmd, "SPI"))
  {
    PiSpiStats_t st;
    PiSpi_GetStats(&st);
    printf("[SPI] CS에지 %lu 잡음 %lu | 완전 %lu 짧음 %lu | 마지막 %lu바이트 첫 %02X %02X | "
           "파싱오류 %lu | 지금 PB12(CS)=%u PB13(SCK)=%u\r\n",
           (unsigned long)st.cs_edges, (unsigned long)st.glitches, (unsigned long)st.full_frames,
           (unsigned long)st.short_frames, (unsigned long)st.last_len,
           st.last_head[0], st.last_head[1], (unsigned long)s_spi_rx_err,
           st.nss_level, st.sck_level);
  }
  /* --- "STEP <1-4> <deg>" : 조향 서보 속도·도달시간 측정 (단일 문자 "S" 분기보다 먼저) --- */
  else if (cmd_is(cmd, "STEP"))
  {
    /* sscanf("%f") 는 nano.specs 에서 안 된다 (위 T 명령 주석 참고) -> strtol/strtof */
    char *p1 = NULL, *p2 = NULL;
    long n = strtol(cmd + 4, &p1, 10);
    float target = (p1 != NULL) ? strtof(p1, &p2) : 0.0f;
    int parsed = (p1 != cmd + 4) && (p2 != NULL) && (p2 != p1);
    const int raw = parsed && (strstr(p2, "RAW") != NULL || strstr(p2, "raw") != NULL);
    int ms = 0;
    const char *pm = parsed ? strstr(p2, "MS") : NULL;
    if (pm != NULL) ms = (int)strtol(pm + 2, NULL, 10);
    if (ms < 0 || ms > 3000) ms = 0;                  /* 0 = 지정 안 함 */
    if (s_teleop_active)
    {
      printf("[STEP] 텔레옵 중에는 못 씁니다 (리셋 후 사용)\r\n");
    }
    else if (!parsed || n < 1 || n > SERVO_COUNT)
    {
      printf("[STEP] 사용법: STEP <서보 1-4> <목표각 도> [RAW [MS <ms>]]  예) STEP 1 90 / STEP 1 0.3 RAW MS 800\r\n");
    }
    else
    {
      steer_step_response((int)n - 1, target, raw, ms);
    }
  }
  /* --- "S 200" : 이동 속도 설정 --- */
  else if (cmd[0] == 'S' || cmd[0] == 's')
  {
    int speed = 0;
    if (sscanf(cmd + 1, "%d", &speed) == 1 && speed >= 0)
    {
      joint_set_speed((uint16_t)speed);
    }
    else
    {
      printf("[CMD] 사용법: S <속도>  (예: S 200, S 0=최대속도)\r\n");
    }
  }
  /* --- "ON" : 토크 ON --- */
  else if ((cmd[0] == 'O' || cmd[0] == 'o') && (cmd[1] == 'N' || cmd[1] == 'n'))
  {
    joint_set_torque_all(1);
  }
  /* --- "OFF" : 토크 OFF --- */
  else if ((cmd[0] == 'O' || cmd[0] == 'o') && (cmd[1] == 'F' || cmd[1] == 'f'))
  {
    joint_set_torque_all(0);
  }
  /* --- "LOG" : 주기적 엔코더 로그 on/off (대화형 작업 시 필수) --- */
  else if (cmd_is(cmd, "LOG"))
  {
    s_log_enabled = !s_log_enabled;
    printf("[LOG] 주기 로그 %s\r\n", s_log_enabled ? "ON" : "OFF");
  }
  /* --- "ECHO" : 입력 문자 되돌려보내기 on/off --- */
  else if (cmd_is(cmd, "ECHO"))
  {
    s_echo_enabled = !s_echo_enabled;
    printf("[ECHO] 입력 에코 %s\r\n", s_echo_enabled ? "ON" : "OFF");
  }
  /* --- "ZERO <id>" / "ZERO ALL" : 현재 자세를 0도로 영점 기록 --- */
  else if (cmd_is(cmd, "ZERO"))
  {
    const char *arg = cmd + 4;
    while (*arg == ' ') arg++;

    if (cmd_is(arg, "ALL"))
    {
      printf("[ZERO] 서보 %d개 전체 영점 기록...\r\n", SERVO_COUNT);
      for (int i = 0; i < SERVO_COUNT; i++)
      {
        servo_zero_here((uint8_t)(i + 1));
      }
    }
    else
    {
      int id = 0;
      if (sscanf(arg, "%d", &id) == 1 && id >= 1 && id <= 253)
      {
        printf("[ZERO] ID=%d 영점 기록...\r\n", id);
        servo_zero_here((uint8_t)id);
      }
      else
      {
        printf("[ZERO] 사용법: ZERO <id>  또는  ZERO ALL\r\n");
        printf("       순서: OFF -> 손으로 바퀴 정면 정렬 -> ZERO ALL -> ON -> T 0 0 0 0\r\n");
        return;
      }
    }
    printf("[ZERO] 완료. 'ON' 으로 토크 켜고 'T 0 0 0 0' 으로 확인하세요.\r\n");
  }
  /* --- "ID <현재> <새>" : 서보 버스 ID 변경 --- */
  else if (cmd_is(cmd, "ID"))
  {
    int old_id = -1, new_id = -1;
    if (sscanf(cmd + 2, "%d %d", &old_id, &new_id) == 2 &&
        old_id >= 0 && old_id <= 254 && new_id >= 0 && new_id <= 253)
    {
      printf("[ID] %d -> %d 변경 시도\r\n", old_id, new_id);

      /* 잠금 플래그 0 = EPROM 쓴 값이 전원 차단 후에도 유지됨.
       * 기본값이 0이지만 누가 1로 바꿔놨을 수 있으므로 명시적으로 씀. */
      uint8_t unlock = 0;
      sts_write((uint8_t)old_id, STS_REG_LOCK_FLAG, &unlock, 1);

      uint8_t v = (uint8_t)new_id;
      STS_Status ret = sts_write((uint8_t)old_id, STS_REG_ID_ADDR, &v, 1);

      if (old_id == STS_BROADCAST_ID)
      {
        printf("[ID] 브로드캐스트 전송 완료 (응답 없는 것이 정상)\r\n");
      }
      else if (ret != STS_OK)
      {
        printf("[ID] 경고: 쓰기 응답 이상 (err=%d)\r\n", ret);
      }

      printf("[ID] 새 ID=%d 확인: %s\r\n", new_id,
             (sts_ping((uint8_t)new_id) == STS_OK) ? "OK" : "응답 없음");
    }
    else
    {
      printf("[ID] 사용법: ID <현재ID> <새ID>\r\n");
      printf("     ** 반드시 서보를 1개만 연결한 상태에서 실행 **\r\n");
      printf("     (여러 개 물린 채로 하면 전부 같은 ID가 됨)\r\n");
      printf("     254 = 브로드캐스트, 현재 ID를 몰라도 됨.  예) ID 254 3\r\n");
    }
  }
  /* --- "GATE" : 구동 정렬 게이트 on/off ---
   * 서보를 떼고 구동모터만 벤치 시험할 때 끈다. 켜 둔 채로 서보 각을 못 읽으면
   * 구동이 계속 보류된다 (어느 바퀴가 어디를 보는지 모르면 구동하지 않는다). */
  else if (cmd_is(cmd, "GATE"))
  {
    int on = (s_gate.trigger_rad <= 0.0f);
    drive_gate_init(&s_gate, on ? DRIVE_GATE_TRIGGER_DEG * PI / 180.0f : 0.0f,
                    DRIVE_GATE_RELEASE_DEG * PI / 180.0f,
                    DRIVE_GATE_MAX_HOLD_MS / CTRL_PERIOD_MS);
    printf("[GATE] 구동 정렬 게이트 %s\r\n", on ? "ON" : "OFF (서보 없이 벤치 시험용)");
  }
  /* --- "CAP <1-4|ALL|234..> <d0> <d1> <t_sw> <t_end> [RAMP] [DEC <n>]" : 구동모터 고속 캡처 ---
   * 모터 모델링용 (firmware/MOTOR_MODELING_PLAN.md §0). 50Hz 로그로는 시정수(~6ms)가
   * 안 보이므로 엔코더를 1kHz(DEC n 이면 n ms 간격)로 RAM 에 기록한 뒤 한꺼번에 출력한다. */
  else if (cmd_is(cmd, "CAP"))
  {
    if (s_teleop_active)
    {
      printf("[CAP] 텔레옵 중에는 못 씁니다 (리셋 후 사용)\r\n");
    }
    else
    {
      run_capture(cmd + 3);
    }
  }
  /* --- "WPI [ON|OFF|SET <이름> <값>]" : 바퀴 속도 PI (common/wheel_pi.h) ---
   * 인자 없으면 상태와 이득 출력.  SET 은 RAM 값 (재부팅하면 설계값).  이름: kp ki imax kick fkin kduty taum */
  else if (cmd_is(cmd, "WPI"))
  {
    const char *a = cmd + 3;
    while (*a == ' ') a++;
    if (cmd_is(a, "ON") || cmd_is(a, "OFF"))
    {
      s_wpi_on = cmd_is(a, "ON");
      /* ON/OFF 는 엔코더 고장(보호 3) 래치도 푼다: 배선 확인 후 다시 켜는 것 */
      for (int m = 0; m < SWERVE_NUM_MODULES; m++) wheel_pi_clear_fault(&s_wpi[m]);
    }
    else if (cmd_is(a, "SET"))
    {
      a += 3;
      while (*a == ' ') a++;
      static const struct { const char *name; size_t off; } F[] = {
        {"KP", offsetof(wheel_pi_cfg_t, kp)}, {"KI", offsetof(wheel_pi_cfg_t, ki)},
        {"IMAX", offsetof(wheel_pi_cfg_t, i_max)}, {"KICK", offsetof(wheel_pi_cfg_t, kick)},
        {"FKIN", offsetof(wheel_pi_cfg_t, f_kin)}, {"KDUTY", offsetof(wheel_pi_cfg_t, k_duty)},
        {"TAUM", offsetof(wheel_pi_cfg_t, tau_m)}, {"CORR", offsetof(wheel_pi_cfg_t, corr_max)},
        {"STALLT", offsetof(wheel_pi_cfg_t, stall_t)},
      };
      int done = 0;
      for (size_t f = 0; f < sizeof F / sizeof F[0] && !done; f++)
      {
        size_t len = strlen(F[f].name);
        if (cmd_is(a, F[f].name) && a[len] == ' ')
        {
          char *endp;
          float val = strtof(a + len, &endp);
          if (endp != a + len && isfinite(val) && val >= 0.0f && !(f == 5 && val <= 0.0f))
          {
            *(float *)((char *)&s_wpi_cfg + F[f].off) = val;
            for (int m = 0; m < SWERVE_NUM_MODULES; m++) wheel_pi_reset(&s_wpi[m]);
            done = 1;
          }
        }
      }
      if (!done) printf("[WPI] 사용법: WPI SET <kp|ki|imax|kick|fkin|kduty|taum|corr|stallt> <값 >= 0>\r\n");
    }
    else if (*a != '\0')
    {
      printf("[WPI] 사용법: WPI [ON|OFF|SET <이름> <값>]\r\n");
    }
    printf("[WPI] %s  kp %.1f ki %.0f imax %.0f kick %.0f fkin %.1f kduty %.4f taum %.3f dt %.3f\r\n",
           s_wpi_on ? "ON (PI)" : "OFF (개루프 duty)", s_wpi_cfg.kp, s_wpi_cfg.ki, s_wpi_cfg.i_max,
           s_wpi_cfg.kick, s_wpi_cfg.f_kin, s_wpi_cfg.k_duty, s_wpi_cfg.tau_m, s_wpi_cfg.dt);
    printf("[WPI] 보호: corr %.0f  stall %.2f rad/s %.2f s  방향 %.0f duty %.2f rad/s %.2f s, 재시도 %.1f s, %.0f s 안 %.0f회면 잠금\r\n",
           s_wpi_cfg.corr_max, s_wpi_cfg.stall_w, s_wpi_cfg.stall_t,
           s_wpi_cfg.enc_duty, s_wpi_cfg.enc_w, s_wpi_cfg.enc_t,
           s_wpi_cfg.enc_retry_t, s_wpi_cfg.enc_window_t, s_wpi_cfg.enc_trips_max);
    for (int m = 0; m < SWERVE_NUM_MODULES; m++)
      printf("[WPI] 모듈 %d: %s%s방향불일치 %u회%s\r\n", m, s_wpi[m].stall ? "스톨(FF만) " : "",
             s_wpi[m].enc_hold ? "재시도 대기(FF만) " : "", (unsigned)s_wpi[m].enc_trips,
             s_wpi[m].enc_dir ? " -> 잠김(FF만, WPI ON 으로 해제)" : "");
  }
  /* --- "ICAP [ms|DUMP]" : IMU 원시 샘플 기록 (위 run_icap) --- */
  else if (cmd_is(cmd, "ICAP"))
  {
    run_icap(cmd + 4);
  }
  /* --- "WCAP ..." : 바퀴 속도 PI 계단 캡처 (위 run_wcap) --- */
  else if (cmd_is(cmd, "WCAP"))
  {
    if (s_teleop_active)
    {
      printf("[WCAP] 텔레옵 중에는 못 씁니다 (리셋 후 사용)\r\n");
    }
    else
    {
      run_wcap(cmd + 4);
    }
  }
  /* --- "ACC <0-254>" : 조향 서보 4개 가속도 레지스터 (joint_init 기본 50, 0 = 제한 없음) ---
   * 모델링 S1: 같은 STEP 을 ACC 50 / ACC 0 으로 재서 가감속 제한이 응답을 얼마나 늦추는지 본다.
   * RAM 값이라 서보 전원을 끄면 다음 부팅에 joint_init 이 다시 50 으로 쓴다. */
  else if (cmd_is(cmd, "ACC"))
  {
    int acc = -1;
    if (sscanf(cmd + 3, "%d", &acc) == 1 && acc >= 0 && acc <= 254)
    {
      for (int i = 0; i < SERVO_COUNT; i++)
      {
        STS_Status ret = sts_set_acceleration((uint8_t)(i + 1), (uint8_t)acc);
        printf("[ACC] ID=%d 가속도 %d %s\r\n", i + 1, acc, (ret == STS_OK) ? "OK" : "실패/무응답");
      }
    }
    else
    {
      printf("[ACC] 사용법: ACC <0-254>  (기본 50, 0 = 가감속 제한 없음)\r\n");
    }
  }
  /* --- "MOT <1-4|ALL> <duty> <ms>" : 구동모터 부호 확인 (조립 후 1회) ---
   * MTEST 와 달리 **한 모터를 정확히 한 번** 돌리고 엔코더 변화를 찍는다.
   * MOTOR_DRIVE_SIGN / ENCODER_SIGN 을 **일부러 적용하지 않는다** -- 지금
   * 재려는 것이 그 부호 자체이므로, 보정 전의 날 것 관계를 봐야 한다.
   * 바퀴를 **반드시 띄운 상태**에서 쓴다. */
  else if (cmd_is(cmd, "MOT"))
  {
    if (s_teleop_active)
    {
      printf("[MOT] 텔레옵 중에는 못 씁니다 (리셋 후 사용)\r\n");
    }
    else
    {
      const char *arg = cmd + 3;
      while (*arg == ' ') arg++;

      int first = 0, last = 0, duty = 0, ms = 0, ok = 0;
      if (cmd_is(arg, "ALL"))
      {
        first = 0; last = MOTOR_COUNT - 1;
        ok = (sscanf(arg + 3, "%d %d", &duty, &ms) == 2);
      }
      else
      {
        int idx = 0;
        ok = (sscanf(arg, "%d %d %d", &idx, &duty, &ms) == 3);
        if (ok && (idx < 1 || idx > MOTOR_COUNT)) ok = 0;
        first = last = idx - 1;
      }
      if (ok && (duty < -PWM_MAX || duty > PWM_MAX)) ok = 0;
      if (ok && (ms < 50 || ms > 3000)) ok = 0;

      if (!ok)
      {
        printf("[MOT] 사용법: MOT <1-4|ALL> <duty> <ms>\r\n");
        printf("       duty -%d~+%d, ms 50~3000.  예) MOT 1 300 1000\r\n", PWM_MAX, PWM_MAX);
        printf("       ** 바퀴를 띄운 상태에서 쓸 것 **\r\n");
      }
      else
      {
        /* 자가시험 시퀀스가 끼어들어 명령을 덮어쓰지 않게 끈다. */
        s_mtest_enabled = 0;
        Motor_StopAll();

        for (int i = first; i <= last; i++)
        {
          Encoder_Reset((MotorID)i);
          int32_t c0 = Encoder_GetCount((MotorID)i);

          Motor_SetSpeed((MotorID)i, (int16_t)duty);   /* 부호 보정 없이 그대로 */
          HAL_Delay((uint32_t)ms);
          Motor_SetSpeed((MotorID)i, 0);

          int32_t c1 = Encoder_GetCount((MotorID)i);
          int32_t d  = c1 - c0;
          /* 16비트 타이머는 65536 틱(=7.8회전)마다 감싸므로 접어준다. */
          if (!Encoder_IsWide((MotorID)i)) d = (int32_t)(int16_t)(c1 - c0);

          float rev = (float)d / TICKS_PER_REV;
          printf("[MOT] MOTOR%d duty=%+d %dms -> 엔코더 %+ld tick (%+.3f rev, %+.1f rpm)\r\n",
                 i + 1, duty, ms, (long)d, rev, rev * 60000.0f / (float)ms);
          if (d == 0)
          {
            printf("      엔코더 0 -- 배선·전원·듀티 확인 (듀티가 너무 작아 기동 못 했을 수 있음)\r\n");
          }
          else
          {
            printf("      => duty %c 일 때 이 엔코더는 %c 로 센다\r\n",
                   duty >= 0 ? '+' : '-', d > 0 ? '+' : '-');
            /* 지금 main.c 부호표를 적용하면: 바퀴가 실제로 간 방향(구동 부호)과
             * 텔레메트리·LOG m/s 부호(엔코더 부호)가 같아야 한다. 손으로 못 돌릴 때 확인용. */
            int fwd = (duty * MOTOR_DRIVE_SIGN[i]) > 0;          /* 이 duty 가 전진이었나 */
            int tlm_pos = (d * ENCODER_SIGN[i]) > 0;              /* 텔레메트리는 + 로 보나 */
            printf("      [보정 후] 바퀴 %s, 텔레메트리 %c  -> %s\r\n",
                   fwd ? "전진" : "후진", tlm_pos ? '+' : '-',
                   (fwd == tlm_pos) ? "OK" : "불일치 (ENCODER_SIGN 반대)");
          }
          HAL_Delay(200);   /* 관성이 멎을 시간 */
        }
        Motor_StopAll();
        printf("[MOT] 완료. 바퀴가 **전진 방향**으로 돈 duty 부호를 기록할 것.\r\n");
        printf("      전진이 duty - 였으면 그 모터의 MOTOR_DRIVE_SIGN = -1\r\n");
        printf("      전진인데 tick 이 - 였으면 그 모터의 ENCODER_SIGN = -1\r\n");
      }
    }
  }
  /* --- "CTRL [ON|OFF]" : 텔레옵 상태줄 (받은 vx vy w) on/off ---
   * 인자 없으면 토글. 조이스틱 --monitor 가 시작할 때 "CTRL ON" 을 보낸다. */
  else if (cmd_is(cmd, "CTRL"))
  {
    const char *arg = cmd + 4;
    while (*arg == ' ') arg++;
    if (cmd_is(arg, "ON"))       s_ctrl_log = 1;
    else if (cmd_is(arg, "OFF")) s_ctrl_log = 0;
    else                         s_ctrl_log = !s_ctrl_log;
    printf("[CTRL] 상태줄 %s (텔레옵 중 5Hz: 받은 v / 조향각 / duty)\r\n",
           s_ctrl_log ? "ON" : "OFF");
  }
  /* --- "MTEST" : 모터 자가시험 on/off (텔레옵 전만) ---
   * 모터를 하나씩 정/역 2초씩 돌린다. 바퀴를 띄운 상태에서 모터 번호·방향
   * 확인용. 첫 구동 명령이 오면 자동으로 멈춘다. */
  else if (cmd_is(cmd, "MTEST"))
  {
    if (s_teleop_active)
    {
      printf("[MTEST] 텔레옵 중에는 못 켭니다 (리셋 후 사용)\r\n");
    }
    else
    {
      s_mtest_enabled = !s_mtest_enabled;
      Motor_StopAll();
      printf("[MTEST] 모터 자가시험 %s\r\n",
             s_mtest_enabled ? "ON — 바퀴를 띄우세요" : "OFF");
    }
  }
  /* --- "HELP" : 사용법 --- */
  else if (strncmp(cmd, "HELP", 4) == 0 || strncmp(cmd, "help", 4) == 0)
  {
    print_help();
  }
  /* --- 알 수 없는 명령 ---
   * 조이스틱이 연속 전송 중일 때 명령 처리에 밀려 바이트가 잘리면 여기로
   * 조각이 들어온다. 그때마다 도움말 전체를 찍으면 UART 가 마비되므로
   * 한 줄만 출력한다. */
  else
  {
    printf("[CMD] 알 수 없는 명령: '%s'  (HELP 입력 시 도움말)\r\n", cmd);
  }
}

static void print_help(void)
{
  printf("========== 서보 명령어 도움말 ==========\r\n");
  printf("  T <a1> <a2>  : 목표 각도 설정 (도)\r\n");
  printf("                 예) T 10.5 -20\r\n");
  printf("  V <vx> <vy> <omega> : 스워브 기구학 (m/s, m/s, rad/s)\r\n");
  printf("                 예) V 0.1 0 0  (직진)\r\n");
  printf("  P            : 현재 각도 출력\r\n");
  printf("  S <speed>    : 이동 속도 설정 (0=최대)\r\n");
  printf("                 예) S 200\r\n");
  printf("  ON           : 토크 ON\r\n");
  printf("  OFF          : 토크 OFF\r\n");
  printf("  PING         : 서보 연결 확인 (ID 1~4)\r\n");
  printf("  SCAN         : 버스 전체 스캔 (ID 1~253, 약 1초)\r\n");
  printf("  ID <현재> <새> : 서보 ID 변경 [1개만 연결한 상태에서!]\r\n");
  printf("                 예) ID 254 3   (254=브로드캐스트)\r\n");
  printf("  ZERO <id|ALL>: 현재 자세를 0도로 영점 기록 [EEPROM 저장]\r\n");
  printf("                 순서) OFF -> 정면 정렬 -> ZERO ALL -> ON\r\n");
  printf("  MOT <n|ALL> <duty> <ms> : 구동모터 1개를 날 부호로 돌리고\r\n");
  printf("                 엔코더 변화 출력 [바퀴 띄우고!]  예) MOT 1 300 1000\r\n");
  printf("  CAP <n|234|ALL> <d0> <d1> <t_sw> <t_end> [RAMP] [DEC n] : 구동모터 1kHz 캡처\r\n");
  printf("                 (모터 모델링, + = 전진) [바퀴 띄우고!]  예) CAP ALL 0 400 100 600\r\n");
  printf("  ICAP [ms] / ICAP DUMP : IMU 208 Hz 원시 기록 (주행 중 가능), 멈춘 뒤 덤프\r\n");
  printf("  WPI [ON|OFF|SET <이름> <값>] : 바퀴 속도 PI (기본 OFF = 개루프 duty). 인자 없으면 이득 출력\r\n");
  printf("  WCAP <n|234|ALL> <w0> <w1> <t_sw> <t_end> [RAMP] [DEC n] : PI 계단 캡처, 바퀴 rad/s\r\n");
  printf("                 [바퀴 띄우고!]  예) WCAP ALL 0 2 100 600\r\n");
  printf("  ACC <0-254>  : 조향 서보 가속도 (기본 50, 0 = 제한 없음)\r\n");
  printf("  MTEST        : 모터 자가시험 시퀀스 on/off\r\n");
  printf("  LOG          : 주기 엔코더 로그 on/off (기본 ON)\r\n");
  printf("  ECHO         : 입력 문자 에코 on/off (기본 ON)\r\n");
  printf("  GATE         : 구동 정렬 게이트 on/off (기본 ON)\r\n");
  printf("  CTRL [ON|OFF]: 텔레옵 상태줄 (받은 vx vy w, 조향각, duty) 5Hz\r\n");
  printf("  SPI          : Pi SPI 링크 진단 (CS 에지·수신 바이트·핀 레벨)\r\n");
  printf("  STEP <n> <도>: 조향 서보 n 을 목표각으로 보내며 속도·도달시간 측정\r\n");
  printf("  STEP <n> <도> RAW [MS <ms>]: 위 + 1 ms 궤적 덤프. 2도 미만 작은 계단(데드밴드)은 RAW 로만, ms 동안 고정 기록\r\n");
  printf("  MTEST        : 모터 자가시험 on/off (기본 OFF, 바퀴 띄우고)\r\n");
  printf("  HELP         : 이 도움말\r\n");
  printf("========================================\r\n");
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM1_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_TIM5_Init();
  MX_TIM8_Init();
  MX_TIM12_Init();
  /* USER CODE BEGIN 2 */
  MX_USART2_Init();
  Motor_Init();
  Encoder_Init();

  printf("\r\n==================================================\r\n");
  printf("     STM32F446RE Rover Motor & Encoder Test       \r\n");
  printf("==================================================\r\n");
  printf("Baud rate: 115200 bps\r\n");
  printf("Motor Speed: +/-400 (PWM Duty ~50%%)\r\n\r\n");

  /* === 서보 모터 초기화 === */
  MX_USART3_Init();           /* USART3 (PB10/PB11, 1Mbps) 레지스터 초기화 */
  sts_init();                 /* STS3215 드라이버 초기화 */
  MX_USART2_RxInt_Init();     /* USART2 RX 인터럽트 활성화 */
  servo_boot_sequence();      /* PING → 토크 ON → 준비 완료 메시지 */
  drive_gate_init(&s_gate, DRIVE_GATE_TRIGGER_DEG * PI / 180.0f,
                  DRIVE_GATE_RELEASE_DEG * PI / 180.0f,
                  DRIVE_GATE_MAX_HOLD_MS / CTRL_PERIOD_MS);
  wheel_pi_default_cfg(&s_wpi_cfg);
  s_wpi_cfg.dt = (float)CTRL_PERIOD_MS / 1000.0f;
  for (int m = 0; m < SWERVE_NUM_MODULES; m++) wheel_pi_clear_fault(&s_wpi[m]);
  PiSpi_Init();               /* SPI2 슬레이브 (Pi 링크, PB12/PB13/PC2/PC3) */
  printf("IMU (ISM330DHCX, SPI3): %s\r\n",
         Imu_Init() ? "OK" : "응답 없음 (배선/CS=PC9 확인, 1초마다 재시도)");

  HAL_Delay(1000);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  WheelState_t wheels[MOTOR_COUNT] = {0};
  uint32_t last_log_time = HAL_GetTick();
  /* 엔코더 속도 추정을 **로깅과 분리**한다.  예전엔 last_log_time 하나로 둘을
   * 같이 돌려서, 로그 주기를 바꾸면 속도 추정 주기까지 바뀌었다 (그리고 10Hz 는
   * 바퀴 속도 폐루프에 너무 느리다).
   * 8384 tick/rev 이므로 20ms 창에서도 분해능이 충분하다:
   *     0.20 m/s -> 119 tick / 20ms,   0.02 m/s -> 12 tick / 20ms */
  uint32_t last_enc_time = HAL_GetTick();
  /* 속도 계산용 dt 는 CPU 사이클 카운터로 잰다. HAL_GetTick(1ms) 로 20ms 창을
   * 재면 19/20/21ms 가 섞여 속도에 샘플마다 ±5% 잡음이 생긴다.
   * 16MHz 에서 32비트 카운터는 268초마다 랩하지만 차분만 쓰므로 문제없다. */
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  uint32_t last_enc_cyc = DWT->CYCCNT;
  const uint32_t ENC_PERIOD_MS = CTRL_PERIOD_MS;  /* 50 Hz 제어주기와 같이 돈다 */
  uint8_t ctrl_tick = 0;
  uint32_t step_start_time = HAL_GetTick();
  uint8_t step = 0;

  /* Initialize last raw counts */
  for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
    wheels[i].last_raw_count = Encoder_GetCount((MotorID)i);
  }

  /* Test parameters */
  const int16_t TEST_SPEED = 400; /* ~50% PWM Duty (PWM_MAX is 799) */

  uint32_t last_imu_poll_ms = HAL_GetTick();
  while (1)
  {
    uint32_t now = HAL_GetTick();

    /* 0. IMU 샘플 쌓기 (1 ms 마다 상태 확인, 새 샘플이면 더함). 제어주기에 평균을 보낸다.
     *    ICAP 기록 중이면 그 샘플을 RAM 에도 남긴다 (주행하면서 진동 측정). */
    if (now != last_imu_poll_ms)
    {
      int16_t smp[6];
      last_imu_poll_ms = now;
      if (Imu_Poll(smp) && s_icap_rec) icap_store(smp);
    }

    /* 1a. 엔코더 속도 추정 (50 Hz, 로깅과 독립) */
    if (now - last_enc_time >= ENC_PERIOD_MS)
    {
      uint32_t cyc = DWT->CYCCNT;
      float dt = (float)(cyc - last_enc_cyc) / (float)SystemCoreClock;
      last_enc_cyc = cyc;
      last_enc_time = now;

      for (uint8_t i = 0; i < MOTOR_COUNT; i++)
      {
        int32_t current_raw = Encoder_GetCount((MotorID)i);
        int32_t delta_tick = current_raw - wheels[i].last_raw_count;
        /* TIM1/4/8 are 16-bit: they roll over every 65536 ticks (~7.8 wheel
         * revolutions, ~6 s at full speed).  Without this fold the rollover
         * reads as a 65535-tick jump -> a ~4700 RPM spike on one wheel. */
        if (!Encoder_IsWide((MotorID)i)) delta_tick = (int16_t)delta_tick;
        delta_tick *= ENCODER_SIGN[i];   /* 조립 후 확인할 부호 (위 TODO 참고) */
        wheels[i].last_raw_count = current_raw;
        wheels[i].total_count += delta_tick;

        /* Wheel RPM = (delta_tick / TICKS_PER_REV) * (60.0f / dt) */
        wheels[i].rpm = ((float)delta_tick / TICKS_PER_REV) * (60.0f / dt);

        /* Angular velocity w [rad/s] = RPM * (2*pi / 60) */
        wheels[i].rad_per_sec = wheels[i].rpm * (2.0f * PI / 60.0f);

        /* Linear velocity v [m/s] = w * R */
        wheels[i].linear_vel = wheels[i].rad_per_sec * WHEEL_RADIUS_M;
        s_wheel_w[i] = wheels[i].rad_per_sec;
      }

      /* 1a'. 센서 (항상. 텔레옵 전에도 Pi 가 볼 수 있다) */
      read_sensors();
      if ((ctrl_tick % 50U) == 0U || !s_imu_valid)
      {
        Imu_Check();   /* 1초마다 WHO_AM_I·설정 재확인. 무효가 되면 다음 주기에 바로 재설정 */
      }

      /* 1b. 제어주기 (텔레옵 중에만. 그 전에는 모터 자가시험·T 명령이 액추에이터를 쓴다) */
      if (s_teleop_active)
      {
        /* 워치독: 명령이 끊기면 목표를 0 으로.  모터를 직접 끄는 대신 목표를
         * 0 으로 두어야 다음 주기가 옛 명령을 다시 내보내지 않는다.
         * 조향은 IK 데드밴드가 마지막 각을 유지한다. */
        if (now - s_last_cmd_ms > TELEOP_TIMEOUT_MS)
        {
          if (!s_teleop_timed_out)
          {
            s_teleop_timed_out = 1;
            printf("[텔레옵] 명령 %dms 이상 끊김 — 정지\r\n", TELEOP_TIMEOUT_MS);
          }
          s_cmd_applied[0] = s_cmd_applied[1] = s_cmd_applied[2] = 0.0f;
        }
        else
        {
          s_cmd_applied[0] = s_cmd_target[0];
          s_cmd_applied[1] = s_cmd_target[1];
          s_cmd_applied[2] = s_cmd_target[2];
        }
        /* RL 잔차: 명령이 살아 있고 잔차가 SPI_RES_TIMEOUT_MS 안에 왔을 때만. 아니면 순수 IK */
        s_res_active = (uint8_t)(!s_teleop_timed_out && s_res_have &&
                                 (now - s_res_ms) <= SPI_RES_TIMEOUT_MS);
        Swerve_Control(s_cmd_applied[0], s_cmd_applied[1], s_cmd_applied[2],
                       s_res_active ? s_res_d : NULL, s_res_active ? s_res_g : NULL);
      }

      /* 1c. 텔레메트리 (항상. 텔레옵 전에도 Pi 가 엔코더를 볼 수 있다).
       *     t_ms 는 **Nucleo 시각** -- Pi 수신시각을 쓰면 전송 지터가 오차가 된다. */
      {
        spi_link_telemetry_t t;
        memset(&t, 0, sizeof t);
        int imu_read_time_valid = s_imu_valid &&
            Imu_MeanReadOffsetUs(now, &t.imu_mean_read_offset_us);
        t.seq_echo = s_spi_seq_echo;
        t.tick = ctrl_tick++;
        t.flags = (uint16_t)(
            (s_teleop_active    ? SPI_TLM_FLAG_TELEOP_ACTIVE  : 0u) |
            (s_teleop_timed_out ? SPI_TLM_FLAG_WATCHDOG       : 0u) |
            (s_gate.pending     ? SPI_TLM_FLAG_GATE_PENDING   : 0u) |
            (s_cmd_from_spi     ? SPI_TLM_FLAG_SRC_SPI        : 0u) |
            (s_servo_read_err   ? SPI_TLM_FLAG_SERVO_READ_ERR : 0u) |
            (s_steer_clamped    ? SPI_TLM_FLAG_STEER_CLAMPED  : 0u) |
            (s_imu_valid        ? SPI_TLM_FLAG_IMU_VALID      : 0u) |
            (s_gate.timed_out   ? SPI_TLM_FLAG_GATE_TIMEOUT   : 0u) |
            (s_wpi_on           ? SPI_TLM_FLAG_WHEEL_PI       : 0u) |
            (wpi_any(0)         ? SPI_TLM_FLAG_WHEEL_STALL    : 0u) |
            (wpi_any(1)         ? SPI_TLM_FLAG_ENC_DIR        : 0u) |
            (s_res_active       ? SPI_TLM_FLAG_RES_ACTIVE     : 0u) |
            (imu_read_time_valid ? SPI_TLM_FLAG_IMU_READ_TIME_VALID : 0u));
        t.t_ms = now;
        t.cmd_vx = s_cmd_applied[0];
        t.cmd_vy = s_cmd_applied[1];
        t.cmd_omega = s_cmd_applied[2];
        for (int m = 0; m < SWERVE_NUM_MODULES; m++)
        {
          const WheelState_t *w = &wheels[SWERVE_MODULE_MOTOR[m]];
          t.enc_ticks[m] = (int32_t)w->total_count;
          t.wheel_mps[m] = w->linear_vel;
          t.steer_rad[m] = s_steer_goal_rad[m];
          t.duty[m] = s_duty_applied[m];
          t.steer_meas_rad[m] = s_servo_fb[m].angle_deg * PI / 180.0f;
          t.steer_speed[m] = s_servo_fb[m].speed;
          t.steer_load[m] = s_servo_fb[m].load;
          t.steer_volt[m] = s_servo_fb[m].voltage;
          t.steer_temp[m] = s_servo_fb[m].temp;
          t.steer_status[m] = s_servo_fb[m].status;
        }
        t.servo_valid = s_servo_valid;
        t.imu_n = s_imu_n;
        memcpy(t.imu, s_imu_raw, sizeof t.imu);
        t.rx_err = (uint16_t)(s_spi_rx_err + PiSpi_ShortTransfers());
        spi_link_pack_telemetry(&t, PiSpi_TxBuffer());
        PiSpi_CommitTx();
      }
    }

    /* 1d. 로깅 (10 Hz).  printf 는 제어 루프와 절대 같은 주기에 두지 않는다. */
    if (now - last_log_time >= 100)
    {
      float dt = (float)(now - last_log_time) / 1000.0f;
      last_log_time = now;
      if (s_log_enabled)
      {
      printf("[TIME:%6lums | dt:%3dms | R=%dmm]\r\n", now, (int)(dt * 1000.0f),
             (int)(WHEEL_RADIUS_M * 1000.0f + 0.5f));
      for (uint8_t i = 0; i < MOTOR_COUNT; i++)
      {
        /* 부호는 따로 찍는다. (int)-0.05f == 0 이라 "정수.소수" 로 나누면 |값|<1 에서
         * 마이너스가 사라진다 (바퀴 m/s 는 최대 ~0.38 이라 항상 + 로 보였다).
         * Raw 는 하드웨어 카운터 그대로 (ENCODER_SIGN 미적용), 나머지 셋은 적용. */
        float rpm = wheels[i].rpm, rad = wheels[i].rad_per_sec, vel = wheels[i].linear_vel;
        int rpm_x10  = (int)(fabsf(rpm) * 10.0f + 0.5f);
        int rad_x100 = (int)(fabsf(rad) * 100.0f + 0.5f);
        int vel_x1k  = (int)(fabsf(vel) * 1000.0f + 0.5f);

        printf(" M%d | Raw:%7ld | %c%3d.%1d RPM | %c%2d.%02d rad/s | %c%d.%03d m/s\r\n",
               i + 1,
               wheels[i].last_raw_count,
               rpm < 0.0f ? '-' : '+', rpm_x10 / 10, rpm_x10 % 10,
               rad < 0.0f ? '-' : '+', rad_x100 / 100, rad_x100 % 100,
               vel < 0.0f ? '-' : '+', vel_x1k / 1000, vel_x1k % 1000);
      }
      printf("-----------------------------------------------------------------\r\n");
      }  /* if (s_log_enabled) */
    }

    /* 2. Step-by-Step Motor Test Sequence (MTEST 로 켰을 때만)
     *    텔레옵(V 명령)이 시작되면 중단한다. 안 그러면 이 시퀀스가
     *    Motor_SetSpeed 로 조이스틱 명령을 몇 초마다 덮어쓴다.
     *    모터 번호·방향을 보려는 시험이라 MOTOR_DRIVE_SIGN 을 적용하지 않는다. */
    if (!s_teleop_active && s_mtest_enabled)
    {
    switch (step)
    {
      /* --- Motor 1 Test --- */
      case 0:
        if (s_log_enabled) printf("\r\n>>> [Step 0] Motor 1 Forward (+%d) for 2s <<<\r\n", TEST_SPEED);
        Motor_SetSpeed(MOTOR1, TEST_SPEED);
        step++;
        step_start_time = now;
        break;
      case 1:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 1] Motor 1 Reverse (-%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR1, -TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 2:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 2] Motor 1 Stop (1s pause) <<<\r\n");
          Motor_Stop(MOTOR1);
          step++;
          step_start_time = now;
        }
        break;

      /* --- Motor 2 Test --- */
      case 3:
        if (now - step_start_time >= 1000) {
          if (s_log_enabled) printf("\r\n>>> [Step 3] Motor 2 Forward (+%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR2, TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 4:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 4] Motor 2 Reverse (-%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR2, -TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 5:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 5] Motor 2 Stop (1s pause) <<<\r\n");
          Motor_Stop(MOTOR2);
          step++;
          step_start_time = now;
        }
        break;

      /* --- Motor 3 Test --- */
      case 6:
        if (now - step_start_time >= 1000) {
          if (s_log_enabled) printf("\r\n>>> [Step 6] Motor 3 Forward (+%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR3, TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 7:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 7] Motor 3 Reverse (-%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR3, -TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 8:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 8] Motor 3 Stop (1s pause) <<<\r\n");
          Motor_Stop(MOTOR3);
          step++;
          step_start_time = now;
        }
        break;

      /* --- Motor 4 Test --- */
      case 9:
        if (now - step_start_time >= 1000) {
          if (s_log_enabled) printf("\r\n>>> [Step 9] Motor 4 Forward (+%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR4, TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 10:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 10] Motor 4 Reverse (-%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR4, -TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 11:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 11] Motor 4 Stop (1s pause) <<<\r\n");
          Motor_Stop(MOTOR4);
          step++;
          step_start_time = now;
        }
        break;

      /* --- All Motors Forward Test --- */
      case 12:
        if (now - step_start_time >= 1000) {
          if (s_log_enabled) printf("\r\n>>> [Step 12] All Motors Forward (+%d) for 3s <<<\r\n", TEST_SPEED);
          for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
            Motor_SetSpeed((MotorID)i, TEST_SPEED);
          }
          step++;
          step_start_time = now;
        }
        break;
      case 13:
        if (now - step_start_time >= 3000) {
          if (s_log_enabled) printf(">>> [Step 13] All Motors Stop & Reset Encoders (2s pause) <<<\r\n");
          Motor_StopAll();
          for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
            Encoder_Reset((MotorID)i);
            wheels[i].last_raw_count = 0;
            wheels[i].total_count = 0;
          }
          step = 0; /* Loop test sequence */
          step_start_time = now + 1000; /* 2s pause */
        }
        break;

      default:
        step = 0;
        break;
    }
    }  /* if (!s_teleop_active) */

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    /* 3. 서보 터미널 명령 처리 (비블로킹)
     *    파싱하는 동안 ISR 이 cmd_line 을 덮어쓸 수 있으므로 지역 버퍼로
     *    복사한 뒤 처리한다. */
    pi_spi_poll();

    if (cmd_ready)
    {
      char line[CMD_BUF_SIZE];
      __disable_irq();
      memcpy(line, cmd_line, sizeof line);
      cmd_ready = 0;
      __enable_irq();
      line[sizeof line - 1] = '\0';

      process_command(line);
    }

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 0;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 65535;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_FALLING;  /* M1 엔코더 A/B 가 PA8/PA9 에 다른 모터와 반대로 물려 있어 계수 방향 반전 (ioc 와 동일) */
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 799;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 0;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 799;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 0;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 65535;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim4, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */

}

/**
  * @brief TIM5 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM5_Init(void)
{

  /* USER CODE BEGIN TIM5_Init 0 */

  /* USER CODE END TIM5_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM5_Init 1 */

  /* USER CODE END TIM5_Init 1 */
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = 0;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 4294967295;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim5, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim5, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM5_Init 2 */

  /* USER CODE END TIM5_Init 2 */

}

/**
  * @brief TIM8 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM8_Init(void)
{

  /* USER CODE BEGIN TIM8_Init 0 */

  /* USER CODE END TIM8_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM8_Init 1 */

  /* USER CODE END TIM8_Init 1 */
  htim8.Instance = TIM8;
  htim8.Init.Prescaler = 0;
  htim8.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim8.Init.Period = 65535;
  htim8.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim8.Init.RepetitionCounter = 0;
  htim8.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim8, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim8, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM8_Init 2 */

  /* USER CODE END TIM8_Init 2 */

}

/**
  * @brief TIM12 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM12_Init(void)
{

  /* USER CODE BEGIN TIM12_Init 0 */

  /* USER CODE END TIM12_Init 0 */

  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM12_Init 1 */

  /* USER CODE END TIM12_Init 1 */
  htim12.Instance = TIM12;
  htim12.Init.Prescaler = 0;
  htim12.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim12.Init.Period = 799;
  htim12.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim12.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim12) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim12, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim12, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM12_Init 2 */

  /* USER CODE END TIM12_Init 2 */
  HAL_TIM_MspPostInit(&htim12);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

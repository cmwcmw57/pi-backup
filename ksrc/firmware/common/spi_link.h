/*
 * spi_link.h
 *
 * Pi(SPI 마스터) <-> Nucleo(SPI2 슬레이브) 프레임. 순수 C (HAL 없음).
 * 파이썬 쪽 pi/common/nucleo_link.py 와 **바이트 단위로 동일**해야 한다.
 *
 * SPI 는 전이중이라 한 번의 전송(CS low -> 144 바이트 -> CS high)에서 동시에:
 *   MOSI (Pi -> Nucleo) : 속도 명령 프레임 (뒤쪽은 0 패딩)
 *   MISO (Nucleo -> Pi) : 텔레메트리 프레임 (직전 50Hz 제어주기에 만든 것)
 * 양방향 모두 고정 길이 SPI_LINK_FRAME_LEN 바이트다. Pi 는 항상 이 길이로 보낸다.
 *
 * 멀티바이트 값은 전부 리틀엔디안, float 는 IEEE-754 binary32.
 * 모든 값은 **같은 제어주기(t_ms)** 에 샘플한 것이다.
 *
 * 명령 프레임 (MOSI)
 *   [0]  0xA5  [1] 0x5A   동기 바이트
 *   [2]  버전 (SPI_LINK_VERSION)
 *   [3]  타입: 0x01 = 속도 명령, 0x02 = 속도 명령 + 바퀴별 잔차 (RL Res8),
 *        0x00 = 폴링만 (명령 없음, 워치독 갱신 안 함)
 *   [4]  seq  (Pi 가 매 전송 +1. 텔레메트리에 되돌려 받아 왕복 확인)
 *   [5]  예약 (0)
 *   [6]  vx  float  m/s   (x=전진)
 *   [10] vy  float  m/s   (y=좌측)
 *   [14] w   float  rad/s (+=반시계)
 *   [18] CRC16 ([2]..[17])                      -- 타입 0x00·0x01
 *   [20..143] 0
 *
 *   타입 0x02 (2026-10-13, RL Res8 = 펌웨어 IK 결과에 바퀴별 보정):
 *   [6..17] vx, vy, w   (위와 같음. 펌웨어가 지금처럼 IK·접기를 한다)
 *   [18] d_steer   float x4 rad   IK 조향각(접은 뒤)에 더할 값, 모듈 순서 FL FR RL RR
 *   [34] wheel_gain float x4      IK 바퀴 속도에 곱할 값 (1 = 그대로)
 *   [50] CRC16 ([2]..[49])
 *   [52..143] 0
 *   펌웨어는 |d_steer| <= SPI_RES_STEER_MAX, wheel_gain 은 SPI_RES_GAIN_MIN..MAX 로 자르고,
 *   잔차가 SPI_RES_TIMEOUT_MS 넘게 안 오면 잔차 없이(순수 IK) 계속 간다.
 *
 * 텔레메트리 프레임 (MISO). x4 배열은 전부 **모듈 순서 FL, FR, RL, RR**.
 *   [0]   0xA5  [1] 0x5A
 *   [2]   버전
 *   [3]   타입 0x81
 *   [4]   seq_echo   마지막으로 받아들인 명령의 seq
 *   [5]   tick       제어주기 카운터 (u8, 랩)
 *   [6]   flags u16  (SPI_TLM_FLAG_*)
 *   [8]   t_ms u32   Nucleo HAL_GetTick() -- 이 프레임을 만든 제어주기 시각
 *   [12]  cmd vx, vy, w  float x3  워치독 적용 후 IK 에 들어간 차체 명령
 *   -- 구동 모터 (Cytron 엔코더) --
 *   [24]  enc_ticks  i32 x4   누적 엔코더 틱 (8384 틱/바퀴 1회전, 부호 = 바퀴 정방향)
 *   [40]  wheel_mps  float x4 엔코더 추정 바퀴 선속도 (20ms 차분)
 *   [56]  steer_cmd  float x4 목표 조향각 rad (접은 뒤, 서보에 보낸 값)
 *   [72]  duty       i16 x4   실제 PWM duty (-799..799, 게이트 적용 후)
 *   -- IMU (ISM330DHCX, 센서 좌표계 원시값, 제어주기 평균 (imu_n). IMU_VALID 일 때만 유효) --
 *   [80]  gyro x,y,z  i16 x3  17.5 mdps/LSB (±500 dps)
 *   [86]  acc  x,y,z  i16 x3  0.122 mg/LSB  (±4 g)
 *   [92]  rx_err u16  CRC/동기/길이 오류로 버린 명령 프레임 수 (랩)
 *   -- 조향 서보 (STS3215 실측. servo_valid 비트가 1 인 모듈만 유효) --
 *   [94]  steer_meas  float x4 현재 조향각 rad (offset/dir 적용, steer_cmd 와 같은 기준)
 *   [110] steer_speed i16 x4   현재 속도 step/s (4096 step = 360도, dir 적용)
 *   [118] steer_load  i16 x4   부하 0.1% 단위 (-1000..1000, dir 적용)
 *   [126] steer_volt  u8 x4    전압 0.1V 단위
 *   [130] steer_temp  u8 x4    온도 섭씨
 *   [134] steer_status u8 x4   서보 상태 비트 (레지스터 65: 전압/센서/온도/전류/각도/과부하)
 *   [138] servo_valid u8       비트 m = 모듈 m 의 서보 값이 이번 주기에 읽혔다
 *   [139] imu_n u8             imu 값이 이번 주기(20 ms) 동안 읽은 샘플 몇 개의 평균인지 (208 Hz → 보통 3–4).
 *                              0 = 평균 없음, 순간값 하나 (2026-10-13 이전 펌웨어도 0)
 *   [140..141] imu_mean_read_offset_us i16 (IMU_READ_TIME_VALID flag일 때만 유효)
 *   [142] CRC16 ([2]..[141])
 *
 * CRC16 = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, 반사/xorout 없음).
 * 동기 바이트는 CRC 에 넣지 않는다 (teleop_protocol 과 같은 방식).
 */

#ifndef SPI_LINK_H
#define SPI_LINK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPI_LINK_FRAME_LEN   144
#define SPI_LINK_SYNC0       0xA5
#define SPI_LINK_SYNC1       0x5A
#define SPI_LINK_VERSION     2

#define SPI_LINK_TYPE_POLL       0x00
#define SPI_LINK_TYPE_VEL_CMD    0x01
#define SPI_LINK_TYPE_VEL_RES    0x02  /* 속도 명령 + 바퀴별 잔차 (RL Res8) */
#define SPI_LINK_TYPE_TELEMETRY  0x81

/* 텔레메트리 flags 비트 */
#define SPI_TLM_FLAG_TELEOP_ACTIVE  (1u << 0)  /* 구동 명령을 한 번이라도 받음 */
#define SPI_TLM_FLAG_WATCHDOG       (1u << 1)  /* 명령 끊김으로 정지 중 */
#define SPI_TLM_FLAG_GATE_PENDING   (1u << 2)  /* 조향 정렬 대기로 구동 보류 중 */
#define SPI_TLM_FLAG_SRC_SPI        (1u << 3)  /* 마지막 명령 출처: 1=SPI, 0=UART 아스키 */
#define SPI_TLM_FLAG_IMU_VALID      (1u << 4)  /* imu 필드 유효 */
#define SPI_TLM_FLAG_SERVO_READ_ERR (1u << 5)  /* 서보 하나 이상 읽기 실패 (servo_valid 참고) */
#define SPI_TLM_FLAG_STEER_CLAMPED  (1u << 6)  /* 조향각이 가동범위 경계로 잘림 */
#define SPI_TLM_FLAG_GATE_TIMEOUT   (1u << 7)  /* 정렬 대기가 1.5초 넘어 강제로 풀림 (조향 막힘 의심) */
#define SPI_TLM_FLAG_WHEEL_PI      (1u << 8)  /* 구동이 바퀴 속도 PI (0 = 개루프 duty). wheel_pi.h */
#define SPI_TLM_FLAG_WHEEL_STALL   (1u << 9)  /* 바퀴 PI 스톨 보호 중인 모듈 있음 (FF 만) */
#define SPI_TLM_FLAG_ENC_DIR       (1u << 10) /* 엔코더 방향 불일치로 잠긴 모듈 있음 (FF 만, WPI ON 으로 해제). wheel_pi.h 보호 3 */
#define SPI_TLM_FLAG_RES_ACTIVE    (1u << 11) /* 이번 주기 IK 에 바퀴별 잔차(타입 0x02)를 더함 */

#define SPI_TLM_FLAG_IMU_READ_TIME_VALID (1u << 12) /* MCU 평균 표본 읽기 시각 유효 */

typedef enum {
    SPI_LINK_OK = 0,
    SPI_LINK_ERR_SYNC,      /* 동기 바이트 불일치 */
    SPI_LINK_ERR_VERSION,   /* 버전 불일치 */
    SPI_LINK_ERR_CRC,       /* CRC 불일치 */
    SPI_LINK_ERR_TYPE       /* 알 수 없는 타입 */
} spi_link_status_t;

/* 타입 0x02 잔차의 펌웨어 쪽 한계 (시뮬 contract.py RES8_STEER_SCALE_RAD 30°, RES8_WHEEL_SCALE 0.6 과 같게) */
#define SPI_RES_STEER_MAX    0.52359878f  /* |d_steer| 상한 [rad] = 30° */
#define SPI_RES_GAIN_MIN     0.4f         /* wheel_gain 하한 = 1 - 0.6 */
#define SPI_RES_GAIN_MAX     1.6f         /* wheel_gain 상한 = 1 + 0.6 */
#define SPI_RES_TIMEOUT_MS   100u         /* 잔차가 이만큼 안 오면 순수 IK (속도 명령 워치독은 따로 500 ms) */

typedef struct {
    uint8_t type;   /* SPI_LINK_TYPE_POLL / VEL_CMD / VEL_RES */
    uint8_t seq;
    float   vx, vy, omega;
    float   d_steer[4];     /* VEL_RES 만. 그 외 타입은 parse 가 0 으로 */
    float   wheel_gain[4];  /* VEL_RES 만. 그 외 타입은 parse 가 1 로 */
} spi_link_cmd_t;

typedef struct {
    uint8_t  seq_echo;
    uint8_t  tick;
    uint16_t flags;
    uint32_t t_ms;
    float    cmd_vx, cmd_vy, cmd_omega;
    int32_t  enc_ticks[4];
    float    wheel_mps[4];
    float    steer_rad[4];      /* 목표 조향각 (steer_cmd) */
    int16_t  duty[4];
    int16_t  imu[6];          /* gyro x,y,z, acc x,y,z */
    uint16_t rx_err;
    float    steer_meas_rad[4];
    int16_t  steer_speed[4];
    int16_t  steer_load[4];
    uint8_t  steer_volt[4];
    uint8_t  steer_temp[4];
    uint8_t  steer_status[4];
    uint8_t  servo_valid;
    uint8_t  imu_n;           /* imu 평균에 들어간 샘플 수 (0 = 순간값) */
    int16_t imu_mean_read_offset_us; /* t_ms 기준 signed us, 실제 tick 정밀도는 1ms */
} spi_link_telemetry_t;

uint16_t spi_link_crc16(const uint8_t *data, uint32_t len);

/* 명령 프레임 해석. OK 가 아니면 out 은 건드리지 않는다. */
spi_link_status_t spi_link_parse_cmd(const uint8_t frame[SPI_LINK_FRAME_LEN],
                                     spi_link_cmd_t *out);

/* 명령 프레임 생성 (Pi 쪽 기준 구현 / 테스트용). */
void spi_link_pack_cmd(const spi_link_cmd_t *cmd, uint8_t frame[SPI_LINK_FRAME_LEN]);

/* 텔레메트리 프레임 생성. */
void spi_link_pack_telemetry(const spi_link_telemetry_t *t,
                             uint8_t frame[SPI_LINK_FRAME_LEN]);

/* 텔레메트리 프레임 해석 (테스트용 / 루프백 확인용). */
spi_link_status_t spi_link_parse_telemetry(const uint8_t frame[SPI_LINK_FRAME_LEN],
                                           spi_link_telemetry_t *out);

#ifdef __cplusplus
}
#endif

#endif /* SPI_LINK_H */

/**
 * @file    joint.c
 * @brief   관절(Joint) 각도 관리 모듈 구현
 *
 * 물리적 각도(도) ↔ 서보 위치값(0~4095) 변환을 처리하며,
 * on_target_angles_received() 콜백 함수를 통해 입력 소스(터미널/SPI)와
 * 모터 제어 로직을 완전히 분리합니다.
 *
 * 변환 공식:
 *   position = offset + dir × (theta × 4096.0 / 360.0)
 *   theta    = (position - offset) × 360.0 / 4096.0 × dir
 */

#include "joint.h"
#include "sts3215.h"
#include <stdio.h>
#include <stdlib.h>

/* ========================== 관절 설정 배열 ========================== */
/*
 * ★ 캘리브레이션 후 아래 값을 실제 로봇에 맞게 수정하세요.
 *
 * 캘리브레이션 방법:
 *   1. 터미널에서 "OFF" 입력 (토크 해제)
 *   2. 서보를 손으로 원하는 0도 위치(직진 방향)로 돌림
 *   3. "P" 입력하여 현재 위치값 읽기 (예: "Servo 1: pos=2100")
 *   4. 아래 offset 값을 해당 값으로 수정 (예: .offset = 2100)
 *   5. 재빌드 후 "ON" → "T 0 0" 입력하여 0도 위치 확인
 */
static JointConfig_t joint_config[SERVO_COUNT] = {
    /*
     * 서보 ID <-> 바퀴 (구동모터가 몸체 안쪽을 향하도록 고정한 배치):
     *
     *     서보1 (LF)      서보3 (RF)
     *     서보2 (LB)      서보4 (RB)
     *
     * 기구학 모듈 순서(FL, FR, RL, RR) 와 다르다. 변환은 main.c 의
     * SWERVE_MODULE_SERVO[] 한 곳에서만 한다.
     *
     * min_deg / max_deg 는 **바퀴마다 다를 수 있는 기구 가동범위**다 (배선·구동모터
     * 간섭). 기구학(swerve_fold_to_range)이 이 값을 그대로 읽어서 범위 안의
     * 등가각만 고르므로, 여기만 고치면 된다.
     *   - 폭(max - min) 이 180도 이상이어야 모든 방향을 낼 수 있다.
     *   - ±90도 (2026-10-11, URDF 조향 관절 한계 ±1.5708 과 같게. 예전 ±100도).
     *     폭이 정확히 180도라 겹치는 구간이 없다: 바퀴 방향이 ±90도를 넘나들면
     *     (예: 순수 옆걸음 근처) 등가각이 반대쪽 끝으로 바뀌어 180도 반전이 난다.
     *     그 동안은 drive_align_gate 가 구동을 보류한다. 히스테리시스가 필요하면
     *     기구 간섭이 허락하는 만큼 다시 넓힐 것 (시뮬 config 와 같이).
     *     바퀴별로 간섭이 다르면 ZERO -> T 로 확인해서 따로 바꿀 것.
     */
    /* idx 0: 서보1, LF */
    {
        .id      = 1,
        .offset  = STS_POS_CENTER,  /* 0도 = 2048 (ZERO 로 영점 기록) */
        .dir     = +1,              /* 양수 각도 = 위치값 증가 방향 (T 로 확인) */
        .min_deg = -90.0f,
        .max_deg = +90.0f,
    },
    /* idx 1: 서보2, LB */
    {
        .id      = 2,
        .offset  = STS_POS_CENTER,
        .dir     = +1,              /* 양수 각도 = 위치값 증가 방향 (T 로 확인) */
        .min_deg = -90.0f,
        .max_deg = +90.0f,
    },
    /* idx 2: 서보3, RF */
    {
        .id      = 3,
        .offset  = STS_POS_CENTER,
        .dir     = +1,              /* 양수 각도 = 위치값 증가 방향 (T 로 확인) */
        .min_deg = -90.0f,
        .max_deg = +90.0f,
    },
    /* idx 3: 서보4, RB */
    {
        .id      = 4,
        .offset  = STS_POS_CENTER,
        .dir     = +1,              /* 양수 각도 = 위치값 증가 방향 (T 로 확인) */
        .min_deg = -90.0f,
        .max_deg = +90.0f,
    },
};

/** 이동 속도 (step/s, 0=최대속도). "S 200" 명령으로 변경 가능 */
static uint16_t joint_goal_speed = 0;

/* ========================== 내부 헬퍼 ========================== */

/**
 * @brief  각도(도) → 서보 위치값(0~4095) 변환
 */
static uint16_t angle_to_position(const JointConfig_t *cfg, float deg)
{
    float pos_f = (float)cfg->offset + (float)cfg->dir * (deg * 4096.0f / 360.0f);

    /* 위치값 클램프 (0~4095) */
    if (pos_f < (float)STS_POS_MIN) pos_f = (float)STS_POS_MIN;
    if (pos_f > (float)STS_POS_MAX) pos_f = (float)STS_POS_MAX;

    return (uint16_t)(pos_f + 0.5f);  /* 반올림 */
}

/**
 * @brief  서보 위치값(0~4095) → 각도(도) 역변환
 */
static float position_to_angle(const JointConfig_t *cfg, uint16_t pos)
{
    return (float)cfg->dir * ((float)pos - (float)cfg->offset) * 360.0f / 4096.0f;
}

/* ========================== 공개 API 구현 ========================== */

void joint_init(void)
{
    printf("[Joint] 관절 모듈 초기화 (서보 %d개)...\r\n", SERVO_COUNT);

    for (int i = 0; i < SERVO_COUNT; i++)
    {
        uint8_t id = joint_config[i].id;

        /* 가속도 설정 (50 = 부드러운 가감속) */
        STS_Status ret = sts_set_acceleration(id, 50);
        if (ret != STS_OK)
        {
            printf("[Joint] 경고: 서보 ID=%d 가속도 설정 실패 (err=%d)\r\n", id, ret);
        }

        /* 토크 ON */
        ret = sts_set_torque(id, 1);
        if (ret != STS_OK)
        {
            printf("[Joint] 경고: 서보 ID=%d 토크 ON 실패 (err=%d)\r\n", id, ret);
        }
        else
        {
            printf("[Joint] 서보 ID=%d 토크 ON 완료\r\n", id);
        }
    }

    /* 목표 속도를 펌웨어 기본값(joint_goal_speed = 0 = 최대)으로 맞춘다. 토크 ON **뒤에** 쓴다.
     * 위치 SYNC_WRITE 는 속도를 보내지 않아서, 안 하면 서보(12V 따로 켜져 있음)가
     * STM 리셋 전에 S 로 넣은 값을 계속 기억한다. 들어갔는지 되읽어 찍는다. */
    joint_set_speed(joint_goal_speed);
    for (int i = 0; i < SERVO_COUNT; i++)
    {
        uint8_t b[2] = { 0, 0 };
        if (sts_read(joint_config[i].id, STS_REG_GOAL_SPEED, 2, b) == STS_OK)
        {
            printf("[Joint] 서보 ID=%d 목표속도 레지스터 = %u\r\n",
                   joint_config[i].id, (unsigned)(b[0] | (b[1] << 8)));
        }
        else
        {
            printf("[Joint] 서보 ID=%d 목표속도 되읽기 실패\r\n", joint_config[i].id);
        }
    }

    printf("[Joint] 관절 모듈 초기화 완료!\r\n");
}

int on_target_angles_received(float *theta, int count)
{
    /*
     * ★ 핵심 콜백 함수 ★
     *
     * 호출처: 터미널 "T" 명령, 기구학 제어주기(main.c Swerve_Control)
     *
     * 동작:
     *   1. 각도 → 위치값 변환 (범위 클램프 포함)
     *   2. SYNC_WRITE로 모든 서보를 한 사이클에 동시 이동
     *
     * 클램프는 조용히 한다 -- 50Hz 제어주기에서 매번 printf 하면 UART 가
     * 포화된다. 필요한 쪽(T 명령)이 반환값을 보고 알린다.
     */

    if (count > SERVO_COUNT) count = SERVO_COUNT;

    uint8_t  ids[SERVO_COUNT];
    uint16_t positions[SERVO_COUNT];
    uint16_t speeds[SERVO_COUNT];
    int clamped = 0;

    for (int i = 0; i < count; i++)
    {
        float deg = theta[i];
        const JointConfig_t *cfg = &joint_config[i];

        if (deg < cfg->min_deg) { deg = cfg->min_deg; clamped++; }
        if (deg > cfg->max_deg) { deg = cfg->max_deg; clamped++; }

        ids[i]       = cfg->id;
        positions[i] = angle_to_position(cfg, deg);
        speeds[i]    = joint_goal_speed;
    }

    /* SYNC_WRITE로 한 사이클에 모든 서보 동시 이동 */
    sts_sync_write_position(ids, positions, speeds, (uint8_t)count);
    return clamped;
}

void joint_get_limits_deg(int idx, float *min_deg, float *max_deg)
{
    if (idx < 0 || idx >= SERVO_COUNT) { *min_deg = 0.0f; *max_deg = 0.0f; return; }
    *min_deg = joint_config[idx].min_deg;
    *max_deg = joint_config[idx].max_deg;
}

int joint_read_angle(int idx, float *deg)
{
    if (idx < 0 || idx >= SERVO_COUNT) return 0;
    int16_t pos = sts_read_position(joint_config[idx].id);
    if (pos < 0) return 0;
    *deg = position_to_angle(&joint_config[idx], (uint16_t)pos);
    return 1;
}

float joint_get_angle(int idx)
{
    if (idx < 0 || idx >= SERVO_COUNT) return -999.0f;

    float deg;
    if (!joint_read_angle(idx, &deg))
    {
        printf("[Joint] 에러: 서보 ID=%d 위치 읽기 실패\r\n", joint_config[idx].id);
        return -999.0f;
    }
    return deg;
}

/* STS 계열의 부호 비트 표현: 크기 + 방향 비트 (2의 보수가 아님) */
static int16_t sts_signed(uint16_t raw, int sign_bit)
{
    uint16_t mag = raw & (uint16_t)((1u << sign_bit) - 1u);
    return (raw & (1u << sign_bit)) ? (int16_t)-(int16_t)mag : (int16_t)mag;
}

int joint_read_feedback(int idx, JointFeedback_t *fb)
{
    if (idx < 0 || idx >= SERVO_COUNT) return 0;
    const JointConfig_t *cfg = &joint_config[idx];

    /* 56-57 위치, 58-59 속도, 60-61 부하, 62 전압, 63 온도, 64 비동기플래그, 65 상태 */
    uint8_t b[10];
    if (sts_read(cfg->id, STS_REG_PRESENT_POSITION, sizeof b, b) != STS_OK) return 0;

    /* 위치도 부호 비트 15 표현이다 (다회전 모드에서 음수가 나올 수 있다) */
    int16_t pos = sts_signed((uint16_t)(b[0] | (b[1] << 8)), 15);
    fb->angle_deg = (float)cfg->dir * ((float)pos - (float)cfg->offset) * 360.0f / 4096.0f;
    fb->speed   = (int16_t)(cfg->dir * sts_signed((uint16_t)(b[2] | (b[3] << 8)), 15));
    fb->load    = (int16_t)(cfg->dir * sts_signed((uint16_t)(b[4] | (b[5] << 8)), 10));
    fb->voltage = b[6];
    fb->temp    = b[7];
    fb->status  = b[9];
    return 1;
}

uint16_t joint_get_speed(void)
{
    return joint_goal_speed;
}

void joint_set_speed(uint16_t speed)
{
    joint_goal_speed = speed;

    /* 각 서보에 개별적으로 Goal Speed 설정 */
    for (int i = 0; i < SERVO_COUNT; i++)
    {
        uint8_t data[2];
        data[0] = (uint8_t)(speed & 0xFF);
        data[1] = (uint8_t)((speed >> 8) & 0xFF);
        sts_write(joint_config[i].id, STS_REG_GOAL_SPEED, data, 2);
    }

    printf("[Joint] 이동 속도 설정: %u step/s\r\n", speed);
}

void joint_set_torque_all(uint8_t on)
{
    for (int i = 0; i < SERVO_COUNT; i++)
    {
        sts_set_torque(joint_config[i].id, on);
    }
    printf("[Joint] 전체 서보 토크 %s\r\n", on ? "ON" : "OFF");
}

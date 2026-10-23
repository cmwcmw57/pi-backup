/* Cytron MDD3A 구동모터 듀티 계산 구현. 자세한 근거는 mdd3a_driver.h 참고. */
#include "mdd3a_driver.h"

static float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

mdd3a_channel_cmd_t mdd3a_speed_to_duty(float speed_norm)
{
    mdd3a_channel_cmd_t cmd;
    float s = clampf(speed_norm, -1.0f, 1.0f);

    if (s >= 0.0f) {
        cmd.duty_a = s;
        cmd.duty_b = 0.0f;
    } else {
        cmd.duty_a = 0.0f;
        cmd.duty_b = -s;
    }
    return cmd;
}

float mdd3a_normalize_speed(float speed_mps, float max_speed_mps)
{
    if (max_speed_mps <= 0.0f) {
        return 0.0f;
    }
    return clampf(speed_mps / max_speed_mps, -1.0f, 1.0f);
}

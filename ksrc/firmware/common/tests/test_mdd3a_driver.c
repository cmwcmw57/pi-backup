/*
 * MDD3A 듀티 계산 호스트 테스트. STM32/HAL 불필요.
 *   gcc -std=c99 -Wall -Wextra -o /tmp/mdd3a_test test_mdd3a_driver.c mdd3a_driver.c -lm && /tmp/mdd3a_test
 */
#include "mdd3a_driver.h"
#include <math.h>
#include <stdio.h>

static int g_fail = 0;

static void expect_near(const char *label, float got, float want, float tol)
{
    if (fabsf(got - want) > tol) {
        printf("FAIL %s: got %.4f want %.4f\n", label, got, want);
        g_fail = 1;
    } else {
        printf("ok   %s: %.4f\n", label, got);
    }
}

int main(void)
{
    mdd3a_channel_cmd_t c;

    c = mdd3a_speed_to_duty(0.6f);
    expect_near("fwd.duty_a", c.duty_a, 0.6f, 1e-6f);
    expect_near("fwd.duty_b", c.duty_b, 0.0f, 1e-6f);

    c = mdd3a_speed_to_duty(-0.6f);
    expect_near("rev.duty_a", c.duty_a, 0.0f, 1e-6f);
    expect_near("rev.duty_b", c.duty_b, 0.6f, 1e-6f);

    c = mdd3a_speed_to_duty(0.0f);
    expect_near("zero.duty_a", c.duty_a, 0.0f, 1e-6f);
    expect_near("zero.duty_b", c.duty_b, 0.0f, 1e-6f);

    c = mdd3a_speed_to_duty(1.5f); /* clamp */
    expect_near("clamp_hi.duty_a", c.duty_a, 1.0f, 1e-6f);
    expect_near("clamp_hi.duty_b", c.duty_b, 0.0f, 1e-6f);

    c = mdd3a_speed_to_duty(-2.0f); /* clamp */
    expect_near("clamp_lo.duty_a", c.duty_a, 0.0f, 1e-6f);
    expect_near("clamp_lo.duty_b", c.duty_b, 1.0f, 1e-6f);

    /* mutual exclusion invariant across a sweep */
    {
        float s;
        int violated = 0;
        for (s = -1.5f; s <= 1.5f; s += 0.05f) {
            mdd3a_channel_cmd_t cc = mdd3a_speed_to_duty(s);
            if (cc.duty_a > 0.0f && cc.duty_b > 0.0f) violated = 1;
        }
        if (violated) {
            printf("FAIL mutual_exclusion: some speed produced both duty_a and duty_b nonzero\n");
            g_fail = 1;
        } else {
            printf("ok   mutual_exclusion: never both nonzero across sweep\n");
        }
    }

    expect_near("normalize.half", mdd3a_normalize_speed(0.3f, 0.6f), 0.5f, 1e-6f);
    expect_near("normalize.clamp_neg", mdd3a_normalize_speed(-0.9f, 0.6f), -1.0f, 1e-6f);
    expect_near("normalize.uncalibrated", mdd3a_normalize_speed(5.0f, 0.0f), 0.0f, 1e-6f);
    expect_near("normalize.uncalibrated_neg_max", mdd3a_normalize_speed(5.0f, -1.0f), 0.0f, 1e-6f);

    if (g_fail) {
        printf("\nSOME TESTS FAILED\n");
        return 1;
    }
    printf("\nALL TESTS PASSED\n");
    return 0;
}

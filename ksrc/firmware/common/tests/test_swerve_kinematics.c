/*
 * 스워브 역기구학 호스트 테스트. STM32/HAL 불필요:
 *   gcc -std=c99 -lm -o /tmp/swerve_test test_swerve_kinematics.c swerve_kinematics.c && /tmp/swerve_test
 */
#include "swerve_kinematics.h"
#include <math.h>
#include <stdio.h>

#define PI 3.14159265358979323846

static int g_fail = 0;

static void expect_near(const char *label, float got, float want, float tol)
{
    if (fabsf(got - want) > tol) {
        printf("FAIL %s: got %.5f want %.5f (tol %.5f)\n", label, got, want, tol);
        g_fail = 1;
    } else {
        printf("ok   %s: %.5f\n", label, got);
    }
}

/* Symmetric 4-module layout, e.g. 200mm track x 200mm wheelbase.
 * Order: 0=FL 1=FR 2=RL 3=RR (x+ forward, y+ left) -- placeholder,
 * replace with real module offsets from CAD. */
static const swerve_module_pos_t MODULES[SWERVE_NUM_MODULES] = {
    { 0.12f,  0.12f },  /* FL */
    { 0.12f, -0.12f },  /* FR */
    {-0.12f,  0.12f },  /* RL */
    {-0.12f, -0.12f },  /* RR */
};

int main(void)
{
    swerve_module_state_t state[SWERVE_NUM_MODULES] = {0};
    swerve_module_cmd_t out[SWERVE_NUM_MODULES];
    int i;

    /* 1) pure forward: all modules point 0 rad, speed 1 */
    swerve_ik_compute(1.0f, 0.0f, 0.0f, MODULES, -1.0f, state, out);
    for (i = 0; i < SWERVE_NUM_MODULES; i++) {
        char lbl[32];
        snprintf(lbl, sizeof lbl, "fwd[%d].angle", i);
        expect_near(lbl, out[i].angle_rad, 0.0f, 1e-4f);
        snprintf(lbl, sizeof lbl, "fwd[%d].speed", i);
        expect_near(lbl, out[i].speed_mps, 1.0f, 1e-4f);
    }

    /* 2) pure strafe left: all modules point +90 deg, speed 1 */
    swerve_ik_compute(0.0f, 1.0f, 0.0f, MODULES, -1.0f, state, out);
    for (i = 0; i < SWERVE_NUM_MODULES; i++) {
        char lbl[32];
        snprintf(lbl, sizeof lbl, "strafe[%d].angle", i);
        expect_near(lbl, out[i].angle_rad, (float)PI / 2.0f, 1e-4f);
        snprintf(lbl, sizeof lbl, "strafe[%d].speed", i);
        expect_near(lbl, out[i].speed_mps, 1.0f, 1e-4f);
    }

    /* 3) pure CCW rotation in place: FL tangential dir is 135 deg, but
     * from the default last_angle=0 that's a >90 deg turn, so the
     * shortest-path optimizer flips it to -45 deg + reversed speed
     * (same ground-contact vector, <=90 deg of steering motion). */
    {
        swerve_module_state_t rot_state[SWERVE_NUM_MODULES] = {0};
        swerve_ik_compute(0.0f, 0.0f, 1.0f, MODULES, -1.0f, rot_state, out);
        expect_near("rotate[FL].angle_deg", out[0].angle_rad * 180.0f / (float)PI, -45.0f, 1e-2f);
        expect_near("rotate[FL].speed", out[0].speed_mps, -0.12f * sqrtf(2.0f), 1e-4f);
    }

    /* 4) continuity through the +-pi seam: no flip, no 2*pi jump */
    {
        swerve_module_state_t s[SWERVE_NUM_MODULES] = {0};
        s[0].last_angle_rad = 3.0f; /* ~171.9 deg */
        s[0].initialized = 1;
        /* body command whose module-0 raw atan2 lands near -171.9 deg,
         * i.e. just across the wrap boundary from 3.0 rad */
        float target_raw = -3.0f;
        float vx = cosf(target_raw), vy = sinf(target_raw);
        swerve_ik_compute(vx, vy, 0.0f, MODULES, -1.0f, s, out);
        /* expect angle to continue smoothly to ~3.283 rad (3.0 + wrap(-3-3)),
         * not snap back near -3.0, and speed to stay positive */
        expect_near("wrap[0].angle", out[0].angle_rad, 3.28319f, 1e-3f);
        expect_near("wrap[0].speed_sign", out[0].speed_mps > 0 ? 1.0f : -1.0f, 1.0f, 1e-6f);
    }

    /* 5) angle optimization: 170 deg target from 0 becomes -10 deg + flipped speed */
    {
        swerve_module_state_t s[SWERVE_NUM_MODULES] = {0};
        float target_raw = 170.0f * (float)PI / 180.0f;
        float vx = cosf(target_raw), vy = sinf(target_raw);
        swerve_ik_compute(vx, vy, 0.0f, MODULES, -1.0f, s, out);
        expect_near("opt[0].angle_deg", out[0].angle_rad * 180.0f / (float)PI, -10.0f, 1e-2f);
        if (out[0].speed_mps >= 0) {
            printf("FAIL opt[0].speed_sign: expected negative, got %.5f\n", out[0].speed_mps);
            g_fail = 1;
        } else {
            printf("ok   opt[0].speed_sign: %.5f\n", out[0].speed_mps);
        }
    }

    /* 6) desaturation: cap wheel speed, verify proportional scaling */
    {
        swerve_module_state_t s[SWERVE_NUM_MODULES] = {0};
        swerve_ik_compute(1.0f, 0.0f, 5.0f, MODULES, 0.5f, s, out);
        float max_abs = 0.0f;
        for (i = 0; i < SWERVE_NUM_MODULES; i++) {
            if (fabsf(out[i].speed_mps) > max_abs) max_abs = fabsf(out[i].speed_mps);
        }
        expect_near("desat.max_speed", max_abs, 0.5f, 1e-4f);
    }

    /* 7) deadband: near-zero command holds last angle, zero speed */
    {
        swerve_module_state_t s[SWERVE_NUM_MODULES] = {0};
        s[0].last_angle_rad = 1.2345f;
        s[0].initialized = 1;
        swerve_ik_compute(0.0f, 0.0f, 0.0f, MODULES, -1.0f, s, out);
        expect_near("deadband[0].angle_held", out[0].angle_rad, 1.2345f, 1e-6f);
        expect_near("deadband[0].speed_zero", out[0].speed_mps, 0.0f, 1e-6f);
    }

    /* 8) 비대칭 범위: 모듈 0 만 [0, 180도]. IK 는 -45도(역회전)를 내지만 범위 밖이라
     *    등가각 135도(정회전)로 옮겨야 한다. 지면 속도벡터 방향은 그대로여야 한다. */
    {
        swerve_module_state_t s[SWERVE_NUM_MODULES] = {0};
        const float lo[SWERVE_NUM_MODULES] = { 0.0f, (float)(-PI), (float)(-PI), (float)(-PI) };
        const float hi[SWERVE_NUM_MODULES] = { (float)PI, (float)PI, (float)PI, (float)PI };
        swerve_ik_compute(-0.1f, 0.1f, 0.0f, MODULES, -1.0f, s, out);
        expect_near("range.ik_angle", out[0].angle_rad, (float)(-PI / 4.0), 1e-5f);
        swerve_fold_to_range(lo, hi, 0.0f, 0.0f, s, out);
        expect_near("range[0].angle", out[0].angle_rad, (float)(3.0 * PI / 4.0), 1e-5f);
        expect_near("range[0].vx", out[0].speed_mps * cosf(out[0].angle_rad), -0.1f, 1e-5f);
        expect_near("range[0].vy", out[0].speed_mps * sinf(out[0].angle_rad), 0.1f, 1e-5f);
        expect_near("range[1].angle_unchanged", out[1].angle_rad, (float)(-PI / 4.0), 1e-5f);
    }

    /* 9) 폭 < 180도: 표현 불가능한 방향은 경계로 클램프 (범위를 절대 넘지 않음) */
    {
        swerve_module_state_t s[SWERVE_NUM_MODULES] = {0};
        const float lo[SWERVE_NUM_MODULES] = { 0.0f, 0.0f, 0.0f, 0.0f };
        const float hi[SWERVE_NUM_MODULES] = { (float)(PI / 4.0), (float)(PI / 4.0),
                                               (float)(PI / 4.0), (float)(PI / 4.0) };
        swerve_ik_compute(-0.1f, 0.1f, 0.0f, MODULES, -1.0f, s, out);
        swerve_fold_to_range(lo, hi, 0.0f, 0.0f, s, out);
        expect_near("narrow[0].clamped_hi", out[0].angle_rad, (float)(PI / 4.0), 1e-6f);
    }

    /* 2026-10-14 회귀: 순수 옆방향(vx=0)을 계속 주면 IK 누적각이 90° 를 1 ulp 위아래로 오가는데,
     * 가동범위 ±90° 접기가 그걸 범위 밖으로 보고 +90/-90 을 매 주기 번갈아 냈다 (게이트가 계속 걸림). */
    {
        swerve_module_state_t s[SWERVE_NUM_MODULES] = {0};
        const float lo[SWERVE_NUM_MODULES] = { (float)(-PI / 2), (float)(-PI / 2), (float)(-PI / 2), (float)(-PI / 2) };
        const float hi[SWERVE_NUM_MODULES] = { (float)(PI / 2), (float)(PI / 2), (float)(PI / 2), (float)(PI / 2) };
        int k, flips = 0;
        float prev = 0.0f;
        for (k = 0; k < 5; k++) {
            swerve_ik_compute(0.05f, 0.0f, 0.0f, MODULES, 0.378f, s, out);
            swerve_fold_to_range(lo, hi, 0.0f, 0.0f, s, out);
        }
        for (k = 0; k < 50; k++) {
            swerve_ik_compute(0.0f, 0.05f, 0.0f, MODULES, 0.378f, s, out);
            swerve_fold_to_range(lo, hi, 0.0f, 0.0f, s, out);
            if (k > 0 && fabsf(out[0].angle_rad - prev) > 1.0f) flips++;
            prev = out[0].angle_rad;
        }
        expect_near("sideways_no_toggle.flips", (float)flips, 0.0f, 0.5f);
        expect_near("sideways_no_toggle.angle", out[0].angle_rad, (float)(PI / 2), 2e-4f);
        expect_near("sideways_no_toggle.speed", out[0].speed_mps, 0.05f, 1e-6f);
    }

    if (g_fail) {
        printf("\nSOME TESTS FAILED\n");
        return 1;
    }
    printf("\nALL TESTS PASSED\n");
    return 0;
}

/*
 * 구동 정렬 게이트 호스트 테스트. STM32/HAL 불필요:
 *   gcc -std=c99 -Wall -Wextra -I.. -o /tmp/gate_test test_drive_align_gate.c ../drive_align_gate.c -lm && /tmp/gate_test
 */
#include "drive_align_gate.h"
#include <math.h>
#include <stdio.h>

#define DEG (3.14159265358979323846f / 180.0f)

static int g_fail = 0;

static void expect_int(const char *label, int got, int want)
{
    if (got != want) {
        printf("FAIL %s: got %d want %d\n", label, got, want);
        g_fail = 1;
    } else {
        printf("ok   %s: %d\n", label, got);
    }
}

int main(void)
{
    drive_align_gate_t g;
    const float zero[4] = {0, 0, 0, 0};
    const float fwd[4]  = {0.1f, 0.1f, 0.1f, 0.1f};

    /* 1) 첫 구동 명령은 대기로 시작, 실제각이 맞으면 즉시 해제 */
    drive_gate_init(&g, 60.0f * DEG, 10.0f * DEG, 0);
    expect_int("first.pending", drive_gate_update_goal(&g, zero, fwd), 1);
    expect_int("first.allow_aligned", drive_gate_allow(&g, zero, zero, 1), 1);

    /* 2) 작은 변화(30도)는 대기를 만들지 않는다 */
    {
        const float g30[4] = {30 * DEG, 30 * DEG, 30 * DEG, 30 * DEG};
        expect_int("small.pending", drive_gate_update_goal(&g, g30, fwd), 0);
        expect_int("small.allow", drive_gate_allow(&g, g30, zero, 1), 1);
    }

    /* 3) 한 바퀴만 90도 바뀌어도 대기, 10도 밖이면 보류, 안에 들면 해제 */
    {
        const float goal[4] = {120 * DEG, 30 * DEG, 30 * DEG, 30 * DEG};
        const float far[4]  = { 60 * DEG, 30 * DEG, 30 * DEG, 30 * DEG};
        const float near[4] = {112 * DEG, 31 * DEG, 29 * DEG, 30 * DEG};
        expect_int("big.pending", drive_gate_update_goal(&g, goal, fwd), 1);
        expect_int("big.hold_far", drive_gate_allow(&g, goal, far, 1), 0);
        /* 목표가 그대로면 update 를 다시 불러도 대기 유지 */
        expect_int("big.still_pending", drive_gate_update_goal(&g, goal, fwd), 1);
        expect_int("big.hold_read_fail", drive_gate_allow(&g, goal, near, 0), 0);
        expect_int("big.release_near", drive_gate_allow(&g, goal, near, 1), 1);
    }

    /* 4) 정지 명령(속도 0)으로 큰 조향 변화는 대기를 만들지 않는다 */
    {
        const float goal[4] = {-60 * DEG, -60 * DEG, 30 * DEG, 30 * DEG};
        expect_int("stopped.no_pending", drive_gate_update_goal(&g, goal, zero), 0);
    }

    /* 5) +179 -> -179 는 wrap 하지 않는다 (358도 변화로 본다) */
    {
        const float a[4] = {179 * DEG, 0, 0, 0};
        const float b[4] = {-179 * DEG, 0, 0, 0};
        drive_gate_init(&g, 60.0f * DEG, 10.0f * DEG, 0);
        drive_gate_update_goal(&g, a, fwd);
        drive_gate_allow(&g, a, a, 1);
        expect_int("seam.pending", drive_gate_update_goal(&g, b, fwd), 1);
        expect_int("seam.hold_at_old", drive_gate_allow(&g, b, a, 1), 0);
    }

    /* 6) trigger <= 0 이면 게이트 OFF */
    {
        const float b[4] = {-179 * DEG, 0, 0, 0};
        drive_gate_init(&g, 0.0f, 10.0f * DEG, 0);
        expect_int("off.pending", drive_gate_update_goal(&g, b, fwd), 0);
        expect_int("off.allow", drive_gate_allow(&g, b, zero, 0), 1);
    }

    /* 7) 실측 트리거: 40도씩 세 번(목표 급변 없음)이어도 실제각이 60도 넘게 뒤처지면 대기 */
    {
        const float a0[4] = {0, 0, 0, 0};
        const float a1[4] = {40 * DEG, 40 * DEG, 40 * DEG, 40 * DEG};
        const float a2[4] = {80 * DEG, 80 * DEG, 80 * DEG, 80 * DEG};
        const float a3[4] = {120 * DEG, 120 * DEG, 120 * DEG, 120 * DEG};
        drive_gate_init(&g, 60.0f * DEG, 10.0f * DEG, 0);
        drive_gate_update_goal(&g, a0, fwd);
        expect_int("ramp.first_release", drive_gate_allow(&g, a0, a0, 1), 1);
        drive_gate_update_goal(&g, a1, fwd);
        expect_int("ramp.40_allow", drive_gate_allow(&g, a1, a0, 1), 1);
        drive_gate_update_goal(&g, a2, fwd);
        expect_int("ramp.80_hold", drive_gate_allow(&g, a2, a0, 1), 0);
        drive_gate_update_goal(&g, a3, fwd);
        expect_int("ramp.120_hold", drive_gate_allow(&g, a3, a1, 1), 0);
        expect_int("ramp.release", drive_gate_allow(&g, a3, a3, 1), 1);
    }

    /* 8) 타임아웃: 5주기 안에 정렬 안 되면 풀고 timed_out, 그 뒤 실측 트리거는 억제 */
    {
        const float a0[4] = {0, 0, 0, 0};
        const float a9[4] = {90 * DEG, 90 * DEG, 90 * DEG, 90 * DEG};
        int k, allowed = 0;
        drive_gate_init(&g, 60.0f * DEG, 10.0f * DEG, 5);
        drive_gate_update_goal(&g, a0, fwd);
        drive_gate_allow(&g, a0, a0, 1);
        for (k = 0; k < 5; k++) {
            drive_gate_update_goal(&g, a9, fwd);
            allowed = drive_gate_allow(&g, a9, a0, 1);   /* 계속 0도에 막혀 있음 */
        }
        expect_int("timeout.released", allowed, 1);
        expect_int("timeout.flag", g.timed_out, 1);
        drive_gate_update_goal(&g, a9, fwd);
        expect_int("timeout.no_retrigger", drive_gate_allow(&g, a9, a0, 1), 1);
        /* 목표가 다시 크게 바뀌면 새 대기, 플래그 초기화 */
        expect_int("timeout.new_jump_pending", drive_gate_update_goal(&g, a0, fwd), 1);
        expect_int("timeout.flag_cleared", g.timed_out, 0);
    }

    if (g_fail) {
        printf("\nSOME TESTS FAILED\n");
        return 1;
    }
    printf("\nALL TESTS PASSED\n");
    return 0;
}

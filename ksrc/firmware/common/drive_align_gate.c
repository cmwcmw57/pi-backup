/* 구동 정렬 게이트 구현. 규칙은 drive_align_gate.h 참고. */
#include "drive_align_gate.h"
#include <math.h>

/* 시뮬(env.py)의 1e-6 과 같은 "구동 명령 있음" 판정 */
#define DRIVE_GATE_SPEED_EPS 1e-6f

void drive_gate_init(drive_align_gate_t *g, float trigger_rad, float release_rad,
                     unsigned max_hold_ticks)
{
    int i;
    g->trigger_rad = trigger_rad;
    g->release_rad = release_rad;
    for (i = 0; i < DRIVE_GATE_NUM_MODULES; i++) {
        g->prev_goal[i] = 0.0f;
    }
    g->has_prev = 0;
    g->pending = 0;
    g->driving = 0;
    g->max_hold_ticks = max_hold_ticks;
    g->hold_ticks = 0;
    g->timed_out = 0;
}

int drive_gate_update_goal(drive_align_gate_t *g,
                           const float goal_rad[DRIVE_GATE_NUM_MODULES],
                           const float speed[DRIVE_GATE_NUM_MODULES])
{
    int i;
    int driving = 0;
    float max_delta = 0.0f;

    if (g->trigger_rad <= 0.0f) {
        g->pending = 0;
        return 0;
    }

    for (i = 0; i < DRIVE_GATE_NUM_MODULES; i++) {
        float d = fabsf(goal_rad[i] - g->prev_goal[i]);
        if (d > max_delta) max_delta = d;
        if (fabsf(speed[i]) > DRIVE_GATE_SPEED_EPS) driving = 1;
    }

    g->driving = driving;
    if (driving && (!g->has_prev || max_delta > g->trigger_rad)) {
        if (!g->pending) g->hold_ticks = 0;
        g->pending = 1;
        g->timed_out = 0;   /* 새 방향 전환: 타임아웃 이력 초기화 */
    }

    for (i = 0; i < DRIVE_GATE_NUM_MODULES; i++) {
        g->prev_goal[i] = goal_rad[i];
    }
    g->has_prev = 1;
    return g->pending;
}

int drive_gate_allow(drive_align_gate_t *g,
                     const float goal_rad[DRIVE_GATE_NUM_MODULES],
                     const float actual_rad[DRIVE_GATE_NUM_MODULES],
                     int actual_ok)
{
    int i;
    float max_err = 0.0f;

    if (g->trigger_rad <= 0.0f) return 1;

    if (actual_ok) {
        for (i = 0; i < DRIVE_GATE_NUM_MODULES; i++) {
            float e = fabsf(goal_rad[i] - actual_rad[i]);
            if (e > max_err) max_err = e;
        }
    }

    /* 실측 트리거: 조금씩 돌려서 (1) 을 피한 큰 방향 전환 */
    if (!g->pending && !g->timed_out && g->driving && actual_ok &&
        max_err > g->trigger_rad) {
        g->pending = 1;
        g->hold_ticks = 0;
    }

    if (!g->pending) return 1;

    if (actual_ok && max_err <= g->release_rad) {
        g->pending = 0;
        g->hold_ticks = 0;
        return 1;
    }

    g->hold_ticks++;
    if (g->max_hold_ticks > 0 && g->hold_ticks >= g->max_hold_ticks) {
        g->pending = 0;
        g->hold_ticks = 0;
        g->timed_out = 1;
        return 1;
    }
    return 0;
}

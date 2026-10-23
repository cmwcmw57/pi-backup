/*
 * drive_align_gate.h
 *
 * 큰 방향 전환에서 조향이 끝날 때까지 구동을 보류하는 게이트.
 * 순수 C (HAL 없음) -> 호스트 gcc 테스트 가능, STM32 에 그대로 편입.
 *
 * 기본 규칙은 sim/rl/env.py 의 drive_align_gate 와 같다 (시뮬은 이 C 가 아니라
 * 파이썬으로 따로 구현돼 있다 -- 아래 "시뮬과 다른 점" 참고):
 *   1) 매 제어주기 목표 조향각(접은 뒤)을 직전 목표와 비교한다. 어느 한 바퀴라도
 *      trigger_rad(기본 60도) 넘게 바뀌었고 구동 명령이 0 이 아니면 대기 시작.
 *   2) 대기 중에는 네 바퀴 구동을 0 으로 둔다.
 *   3) 네 바퀴의 **실제** 조향각이 모두 목표의 release_rad(기본 10도) 안에 들면 해제.
 *
 * 이유: 한 바퀴만 먼저 새 방향으로 구동하면 의도하지 않은 요(yaw)가 생긴다.
 * 조향 가동범위에 이음매가 있으므로 180도 강제 반전은 반드시 일어난다.
 *
 * 각도는 **wrap 하지 않고** 그대로 뺀다. +179 와 -179 는 방향으로는 2도지만
 * 서보는 358도를 돌아야 하므로 wrap 하면 너무 일찍 해제된다.
 *
 * 시뮬과 다른 점 (펌웨어에만 있는 안전장치):
 *   a) 첫 구동 명령은 무조건 대기로 시작해 실제각으로 판정한다. 전원 투입 후
 *      서보가 어디를 보는지 모르므로 (이미 맞으면 바로 해제된다).
 *   b) 실측 트리거: 대기 중이 아니어도 구동 중에 |목표 - 실제| 가 trigger_rad 를
 *      넘으면 대기한다. 조이스틱을 40도씩 세 번 돌리면 (1) 은 한 번도 안 걸리는데
 *      실제로는 120도 방향 전환이라, 서보가 도는 동안 바퀴들이 제각각 구동된다.
 *   c) 보류 타임아웃: max_hold_ticks 넘게 정렬이 안 되면 (모래에서 정지 조향이
 *      안 먹힘, 기구 간섭 등) 보류를 풀고 timed_out 을 세운다. 이후 목표가 다시
 *      크게 바뀔 때까지 (b) 를 끈다 -- 안 그러면 1틱 구동 / 1.5초 정지를 반복한다.
 */

#ifndef DRIVE_ALIGN_GATE_H
#define DRIVE_ALIGN_GATE_H

#ifdef __cplusplus
extern "C" {
#endif

#define DRIVE_GATE_NUM_MODULES 4

typedef struct {
    float trigger_rad;                         /* 대기 시작 기준 (직전 목표 대비 변화량) */
    float release_rad;                         /* 해제 기준 (실제각 - 목표각) */
    float prev_goal[DRIVE_GATE_NUM_MODULES];   /* 직전 주기 목표각 */
    int   has_prev;                            /* prev_goal 이 유효한가 */
    int   pending;                             /* 1 = 구동 보류 중 */
    int   driving;                             /* 이번 주기 구동 명령이 있는가 */
    unsigned max_hold_ticks;                   /* 0 = 타임아웃 없음 */
    unsigned hold_ticks;                       /* 현재 보류가 이어진 주기 수 */
    int   timed_out;                           /* 1 = 타임아웃으로 풀렸다 (다음 목표 급변까지) */
} drive_align_gate_t;

/* trigger_rad <= 0 이면 게이트 전체를 끈다 (항상 구동 허용).
 * max_hold_ticks: 이 주기 수 넘게 정렬이 안 되면 강제로 푼다 (0 = 무한 대기). */
void drive_gate_init(drive_align_gate_t *g, float trigger_rad, float release_rad,
                     unsigned max_hold_ticks);

/* 매 제어주기 목표각을 넣는다. speed 는 모듈별 구동 명령 (부호 무관, 0 판정만 쓴다).
 * 반환: 이 호출 뒤 대기 중이면 1. */
int drive_gate_update_goal(drive_align_gate_t *g,
                           const float goal_rad[DRIVE_GATE_NUM_MODULES],
                           const float speed[DRIVE_GATE_NUM_MODULES]);

/* 매 제어주기 update_goal 다음에 한 번 부른다. 실제 조향각으로 실측 트리거와
 * 해제를 판정한다.
 * actual_ok = 0 (서보 읽기 실패) 이면 해제하지 않는다 -- 한 바퀴라도 어디를
 * 보는지 모르면 구동하지 않는 쪽이 안전하다 (타임아웃은 그대로 센다).
 * 반환: 구동 허용이면 1, 보류면 0. */
int drive_gate_allow(drive_align_gate_t *g,
                     const float goal_rad[DRIVE_GATE_NUM_MODULES],
                     const float actual_rad[DRIVE_GATE_NUM_MODULES],
                     int actual_ok);

#ifdef __cplusplus
}
#endif

#endif /* DRIVE_ALIGN_GATE_H */

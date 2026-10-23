/* 스워브 역기구학 구현. 차체 속도 -> 바퀴 4개의 조향각/속도. 규약은 swerve_kinematics.h 참고. */
#include "swerve_kinematics.h"

#define SWERVE_FOLD_EPS_RAD 1e-4f   /* 0.006 deg: 가동범위 경계 반올림 허용 */
#include <math.h>

#define SWERVE_PI      3.14159265358979323846f
#define SWERVE_HALF_PI (SWERVE_PI * 0.5f)
#define SWERVE_TWO_PI  (SWERVE_PI * 2.0f)

/* 이 크기 미만의 모듈 요구 속도는 "0" 으로 본다 -> atan2 노이즈를 쫓지 말고
 * 조향각을 유지한다. 1 mm/s 는 이 로버에서 의미 있는 값보다 한참 아래다. */
#define SWERVE_VEL_EPS_MPS 0.001f

/* 임의의 실수 각도를 (-pi, pi] 로 접는다. 크기와 무관하게 O(1)
 * (last_angle 은 여러 바퀴에 걸쳐 무한정 누적될 수 있다). */
static float wrap_to_pi(float a)
{
    a = fmodf(a + SWERVE_PI, SWERVE_TWO_PI);
    if (a < 0.0f) {
        a += SWERVE_TWO_PI;
    }
    return a - SWERVE_PI;
}

static float copysign_local(float mag, float sgn)
{
    return sgn < 0.0f ? -mag : mag;
}

void swerve_ik_compute(float vx, float vy, float omega,
                        const swerve_module_pos_t modules[SWERVE_NUM_MODULES],
                        float max_wheel_mps,
                        swerve_module_state_t state[SWERVE_NUM_MODULES],
                        swerve_module_cmd_t out[SWERVE_NUM_MODULES])
{
    int i;
    float max_abs_speed = 0.0f;

    for (i = 0; i < SWERVE_NUM_MODULES; i++) {
        float xi = modules[i].x;
        float yi = modules[i].y;

        /* 강체 점속도: v_i = v_center + omega x r_i.
         * omega 가 +z 축이면 omega x r_i = (-omega*yi, omega*xi, 0) */
        float vx_i = vx - omega * yi;
        float vy_i = vy + omega * xi;
        float speed_i = sqrtf(vx_i * vx_i + vy_i * vy_i);

        if (speed_i < SWERVE_VEL_EPS_MPS) {
            /* 데드밴드: 마지막 조향각 유지, 구동 속도는 0 */
            if (!state[i].initialized) {
                state[i].last_angle_rad = 0.0f;
                state[i].initialized = 1;
            }
            out[i].angle_rad = state[i].last_angle_rad;
            out[i].speed_mps = 0.0f;
            continue;
        }

        {
            float raw_angle = atan2f(vy_i, vx_i);
            float speed = speed_i;
            float delta = wrap_to_pi(raw_angle - state[i].last_angle_rad);

            /* 최단경로 최적화: 90도를 넘게 조향하지 않는다.
             * 나머지 90도는 바퀴 회전 방향을 뒤집어서 대신한다 */
            if (fabsf(delta) > SWERVE_HALF_PI) {
                delta -= copysign_local(SWERVE_PI, delta);
                speed = -speed;
            }

            state[i].last_angle_rad += delta; /* continuous, unwrapped */
            state[i].initialized = 1;

            out[i].angle_rad = state[i].last_angle_rad;
            out[i].speed_mps = speed;
        }

        if (fabsf(out[i].speed_mps) > max_abs_speed) {
            max_abs_speed = fabsf(out[i].speed_mps);
        }
    }

    /* desaturation: 어느 모듈이든 max_wheel_mps 를 넘게 요구하면, 모든 모듈
     * 속도를 같은 비율로 줄인다. 그래야 명령한 운동의 모양(방향/곡률)이
     * 보존된다 */
    if (max_wheel_mps > 0.0f && max_abs_speed > max_wheel_mps) {
        float scale = max_wheel_mps / max_abs_speed;
        for (i = 0; i < SWERVE_NUM_MODULES; i++) {
            out[i].speed_mps *= scale;
        }
    }
}

void swerve_fold_to_limit(float limit_rad, float unwind_rad, float slow_mps,
                          swerve_module_state_t state[SWERVE_NUM_MODULES],
                          swerve_module_cmd_t out[SWERVE_NUM_MODULES])
{
    float lo[SWERVE_NUM_MODULES];
    float hi[SWERVE_NUM_MODULES];
    int i;
    if (limit_rad < SWERVE_HALF_PI) {
        limit_rad = SWERVE_HALF_PI; /* 이보다 좁으면 표현 불가능한 방향이 생긴다 */
    }
    for (i = 0; i < SWERVE_NUM_MODULES; i++) {
        lo[i] = -limit_rad;
        hi[i] = limit_rad;
    }
    swerve_fold_to_range(lo, hi, unwind_rad, slow_mps, state, out);
}

void swerve_fold_to_range(const float lo_rad[SWERVE_NUM_MODULES],
                          const float hi_rad[SWERVE_NUM_MODULES],
                          float unwind_rad, float slow_mps,
                          swerve_module_state_t state[SWERVE_NUM_MODULES],
                          swerve_module_cmd_t out[SWERVE_NUM_MODULES])
{
    int i;
    for (i = 0; i < SWERVE_NUM_MODULES; i++) {
        float lo = lo_rad[i];
        float hi = hi_rad[i];
        float center = 0.5f * (lo + hi);
        float th = out[i].angle_rad;
        float ref = state[i].initialized ? state[i].last_cmd_rad : 0.0f;
        /* 직전 명령각에 가장 가까운 등가각 th + n*PI */
        float n = floorf((ref - th) / SWERVE_PI + 0.5f);
        float folded = th + n * SWERVE_PI;
        /* 가동범위를 벗어나면 한 칸 되돌린다 (폭 >= PI 면 반드시 안에 든다).
         * 2026-10-14: 경계를 반올림 수준(SWERVE_FOLD_EPS_RAD)만 넘은 것은 범위 안으로 본다 (아래에서
         * 경계에 붙임).  순수 옆방향(vx=0)이면 IK 누적각이 90° 를 1 ulp 위아래로 오가서, 예전에는
         * +90/-90 을 매 주기 번갈아 내고 구동 정렬 게이트가 계속 걸렸다. */
        if (folded > hi + SWERVE_FOLD_EPS_RAD) {
            n -= 1.0f;
        } else if (folded < lo - SWERVE_FOLD_EPS_RAD) {
            n += 1.0f;
        }
        folded = th + n * SWERVE_PI;
        /* 폭 < PI 면 어떤 등가각도 범위 밖인 방향이 있다. 기구 보호가 우선이라
         * 경계에 붙인다 (그 바퀴의 방향은 틀어진다). */
        if (folded > hi) {
            folded = hi;
        } else if (folded < lo) {
            folded = lo;
        }
        /* 이음매에서 멀어지도록 미리 푼다: 범위 중앙에서 멀고 이 바퀴가 구동
         * 중이 아니면 중앙에 가까운 등가각으로 넘어간다 (그때는 180도 회전이 공짜다). */
        if (unwind_rad > 0.0f && fabsf(folded - center) > unwind_rad
            && fabsf(out[i].speed_mps) < slow_mps) {
            float alt = folded - (folded > center ? SWERVE_PI : -SWERVE_PI);
            if (fabsf(alt - center) < fabsf(folded - center) && alt >= lo && alt <= hi) {
                n += (folded > center ? -1.0f : 1.0f);
                folded = alt;
            }
        }
        /* n 이 홀수면 180도 뒤집힌 것이므로 바퀴 속도 부호를 반전한다 */
        if (((int)fabsf(n)) % 2 == 1) {
            out[i].speed_mps = -out[i].speed_mps;
        }
        out[i].angle_rad = folded;
        state[i].last_cmd_rad = folded;
    }
}

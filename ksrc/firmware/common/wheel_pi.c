/*
 * wheel_pi.c -- 바퀴 속도 FF + 기준 모델 PI.  설명은 wheel_pi.h.
 */

#include "wheel_pi.h"

#include <math.h>

void wheel_pi_default_cfg(wheel_pi_cfg_t *cfg)
{
    cfg->dt = 0.020f;
    cfg->k_duty = 0.0097f;      /* D1: 200 duty -> 1.94, 400 -> 3.88 rad/s (4개 평균) */
    cfg->f_kin = 5.0f;          /* D1 직선 절편 2–10 duty */
    cfg->kick = 40.0f;          /* D3 기동 35–52 duty (f_kin 과 합쳐 45) */
    cfg->w_still = 0.01f;
    cfg->w_zero = 0.001f;
    /* 극점-영점 상쇄 (ki/kp = 1/τ, τ = 40 ms), 루프 대역 20 rad/s: kp = 20·τ/k_duty.
     * 설계 시뮬(common/tests/wheel_pi_design.py): 계단 오버슈트 0.5%, 0→2 rad/s 90% 37 ms,
     * 경사 부하 80 duty 계단에 속도 처짐 0.41 rad/s 를 130 ms 에 회복.
     * τ 30–50 ms, 기울기 ×0.85–1.05, 정지마찰 35–55 에서 오버슈트 6% 이하. */
    cfg->kp = 82.5f;
    cfg->ki = 2062.0f;
    cfg->i_max = 200.0f;        /* 20° 경사 바퀴당 ~80 + 전압 -15% 중속 ~60 + 개체차 */
    cfg->tau_m = 0.040f;        /* = 실측 기계 시정수 */
    cfg->duty_max = 799.0f;
    /* 보호 (wheel_pi.h).  corr 250: 부하 80 duty + 전압 -15% 중속 ~60 + 가속 과도에 여유.
     * 정상 시험(계단·저속·부하·역전) 에서 |c| 최대는 test_wheel_pi.c 가 출력. */
    cfg->corr_max = 250.0f;
    cfg->stall_w = 0.05f;
    cfg->stall_t = 0.5f;
    cfg->enc_duty = 150.0f;
    cfg->enc_w = 0.3f;
    cfg->enc_t = 0.3f;
    cfg->enc_retry_t = 1.0f;
    cfg->enc_window_t = 10.0f;
    cfg->enc_trips_max = 3.0f;
}

void wheel_pi_reset(wheel_pi_state_t *s)
{
    s->r_m = 0.0f;
    s->i = 0.0f;
    s->stall_n = 0;
    s->enc_n = 0;
    s->stall = 0;
}

void wheel_pi_clear_fault(wheel_pi_state_t *s)
{
    wheel_pi_reset(s);
    s->enc_hold = 0;
    s->enc_win = 0;
    s->enc_trips = 0;
    s->enc_dir = 0;
}

static float clampf(float x, float lim)
{
    return (x > lim) ? lim : ((x < -lim) ? -lim : x);
}

static float sgnf(float x)
{
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

int16_t wheel_pi_step(const wheel_pi_cfg_t *cfg, wheel_pi_state_t *s, float w_ref, float w_meas)
{
    if (!isfinite(w_ref) || !isfinite(w_meas) || fabsf(w_ref) < cfg->w_zero)
    {
        wheel_pi_reset(s);
        return 0;
    }
    if (cfg->tau_m > 0.0f)
        s->r_m += (w_ref - s->r_m) * cfg->dt / (cfg->tau_m + cfg->dt);
    else
        s->r_m = w_ref;
    const float e = s->r_m - w_meas;
    const float sg = sgnf(w_ref);
    const float ff0 = w_ref / cfg->k_duty + sg * cfg->f_kin;

    /* 보호 3 창: 첫 걸림부터 enc_window_t 가 지나면 횟수를 비운다 */
    if (s->enc_trips > 0 && !s->enc_dir)
    {
        s->enc_win++;
        if ((float)s->enc_win * cfg->dt >= cfg->enc_window_t - 1e-4f)
        {
            s->enc_trips = 0;
            s->enc_win = 0;
        }
    }
    /* 보호 2·3 이 걸린 동안: FF 만 (스톨은 바퀴가 다시 돌면, 방향 재시도는 시간이 지나면 풀림) */
    if (s->enc_hold > 0)
        s->enc_hold--;
    else if (s->stall && !s->enc_dir && fabsf(w_meas) >= 0.5f * fabsf(w_ref) && sgnf(w_meas) == sg)
    {
        s->stall = 0;
        s->stall_n = 0;
    }
    if (s->stall || s->enc_dir || s->enc_hold > 0)
    {
        s->i = 0.0f;
        return (int16_t)clampf(ff0, cfg->duty_max);
    }

    float ff = ff0;
    if (fabsf(w_meas) < cfg->w_still)
        ff += sg * cfg->kick;

    /* 조건부 적분: 출력이나 보정량이 오차 방향으로 포화면 쌓지 않는다 */
    const float c_pre = cfg->kp * e + s->i;
    const float u_pre = ff + c_pre;
    const int sat = ((fabsf(u_pre) >= cfg->duty_max) || (fabsf(c_pre) >= cfg->corr_max)) && (sgnf(u_pre) == sgnf(e));
    if (!sat)
        s->i += cfg->ki * e * cfg->dt;
    s->i = clampf(s->i, cfg->i_max);

    const float c = clampf(cfg->kp * e + s->i, cfg->corr_max);
    const float u = clampf(ff + c, cfg->duty_max);

    /* 보호 2: 진행 방향으로 보정 상한에 붙었는데 바퀴가 안 돎 */
    if (c * sg >= cfg->corr_max && fabsf(w_meas) < cfg->stall_w)
        s->stall_n++;
    else
        s->stall_n = 0;
    if ((float)s->stall_n * cfg->dt >= cfg->stall_t - 1e-4f)
    {
        s->stall = 1;
        s->i = 0.0f;
    }
    /* 보호 3: 큰 duty 와 반대로 도는 바퀴 -> 재시도 대기, 창 안에서 enc_trips_max 번이면 잠금 */
    if (fabsf(u) >= cfg->enc_duty && fabsf(w_meas) >= cfg->enc_w && sgnf(w_meas) == -sgnf(u))
        s->enc_n++;
    else
        s->enc_n = 0;
    if ((float)s->enc_n * cfg->dt >= cfg->enc_t - 1e-4f)
    {
        s->enc_n = 0;
        s->i = 0.0f;
        if (s->enc_trips == 0)
            s->enc_win = 0;
        if (s->enc_trips < 255)
            s->enc_trips++;
        if ((float)s->enc_trips >= cfg->enc_trips_max)
            s->enc_dir = 1;
        else
            s->enc_hold = (uint16_t)(cfg->enc_retry_t / cfg->dt + 0.5f);
    }
    return (int16_t)u;
}

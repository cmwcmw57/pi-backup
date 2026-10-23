/*
 * 바퀴 속도 PI 호스트 테스트. STM32/HAL 불필요:
 *   gcc -std=c99 -Wall -Wextra -I.. -o /tmp/t_wheel_pi test_wheel_pi.c ../wheel_pi.c -lm && /tmp/t_wheel_pi
 *
 * 1) 식 하나하나 (FF, 킥, 정지 명령, 보정량 상한, 조건부 적분, 보호: 스톨·엔코더 역방향)
 * 2) 실측 플랜트 모델(2026-10-13)과 닫힌 루프: 계단 오버슈트·상승시간·정상오차·저속 기동·부하 회복.
 *    플랜트는 설계 시뮬(wheel_pi_design.py)과 같은 식:
 *      tau*dw/dt = k*(duty - load - fric(w)) - w,  fric = 5 + 41*exp(-|w|/0.13),
 *      멈춘 상태에서 |duty-load| <= 47 이면 안 움직임.  엔코더 8384 틱/회전, 1 ms 적분.
 */
#include "wheel_pi.h"
#include <math.h>
#include <stdio.h>

#define PI_D 3.14159265358979323846

static int g_fail = 0;

static void expect(const char *label, int ok, double got)
{
    printf("%s %s: %.4f\n", ok ? "ok  " : "FAIL", label, got);
    if (!ok) g_fail = 1;
}

/* ---- 플랜트 -------------------------------------------------------------- */
typedef struct { double w, th, k, tau, stat; int stuck; } plant_t;

static void plant_init(plant_t *p, double kscale, double tau, double stat)
{
    p->w = 0; p->th = 0; p->k = 0.0097 * kscale; p->tau = tau; p->stat = stat; p->stuck = 1;
}

static double sgn(double x) { return (x > 0) - (x < 0); }

static void plant_step(plant_t *p, double duty, double load, double dt)
{
    double net = duty - load, s;
    if (p->stuck || p->w == 0.0)
    {
        p->stuck = 1;
        if (fabs(net) <= p->stat) { p->w = 0; return; }
        p->stuck = 0;
        s = sgn(net);
    }
    else s = sgn(p->w);
    double fr = 5.0 + 41.0 * exp(-fabs(p->w) / 0.13);
    double wn = p->w + (p->k * (net - s * fr) - p->w) / p->tau * dt;
    if (p->w != 0 && sgn(wn) != sgn(p->w))
    {
        wn = 0;
        if (fabs(net) <= p->stat) p->stuck = 1;
    }
    p->w = wn;
    p->th += p->w * dt;
}

static long ticks(const plant_t *p) { return (long)floor(p->th / (2 * PI_D) * 8384.0); }

/* w_ref(t) 를 주고 T 초 돌려서 1 ms 속도 기록. load(t) 는 duty 단위. */
#define NMAX 5000
static double g_w[NMAX];
static double g_cmax;      /* 정상 시험에서 보정량 |kp*e+i| 최대 (보호 1 여유 확인) */
static int g_prot;         /* 보호 2·3 가 한 번이라도 걸렸나 */
static int g_enc_sign = 1; /* -1 = 엔코더 부호가 뒤집힌 배선 */

typedef double (*fn_t)(double);

static void sim(fn_t ref, fn_t load, double T, double kscale, double tau, double stat)
{
    wheel_pi_cfg_t c;
    wheel_pi_state_t s;
    wheel_pi_default_cfg(&c);
    wheel_pi_clear_fault(&s);
    plant_t p;
    plant_init(&p, kscale, tau, stat);
    long last = 0;
    int16_t u = 0;
    g_cmax = 0;
    g_prot = 0;
    int n = (int)(T * 1000.0 + 0.5);
    for (int i = 0; i < n; i++)
    {
        double t = i * 1e-3;
        if (i % 20 == 0)
        {
            long tk = ticks(&p);
            float w_meas = (float)(g_enc_sign * (tk - last) / 8384.0 * 2 * PI_D / 0.020);
            last = tk;
            u = wheel_pi_step(&c, &s, (float)ref(t), w_meas);
            double cc = fabs(c.kp * (s.r_m - w_meas) + s.i);
            if (ref(t) != 0.0 && cc > g_cmax) g_cmax = cc;
            g_prot |= s.stall | ((s.enc_dir | (s.enc_trips > 0)) << 1);
        }
        plant_step(&p, u, load ? load(t) : 0.0, 1e-3);
        g_w[i] = p.w;
    }
}

static double mean(int i0, int i1) { double a = 0; for (int i = i0; i < i1; i++) a += g_w[i]; return a / (i1 - i0); }
static double maxw(int i0, int i1) { double a = -1e9; for (int i = i0; i < i1; i++) if (g_w[i] > a) a = g_w[i]; return a; }
static double minw(int i0, int i1) { double a = 1e9; for (int i = i0; i < i1; i++) if (g_w[i] < a) a = g_w[i]; return a; }
static int rise_ms(int i0, int i1, double target)
{
    for (int i = i0; i < i1; i++) if (g_w[i] >= 0.9 * target) return i - i0;
    return -1;
}

static double r2(double t) { return t >= 0.2 ? 2.0 : 0.0; }
static double r03(double t) { return t >= 0.2 ? 0.3 : 0.0; }
static double rrev(double t) { return t < 0.2 ? 0.0 : (t < 1.0 ? 2.0 : -2.0); }
static double ld80(double t) { return t >= 1.0 ? 80.0 : 0.0; }
static double ldblock(double t) { return (t >= 1.0 && t < 2.0) ? 600.0 : 0.0; }

int main(void)
{
    wheel_pi_cfg_t c;
    wheel_pi_state_t s = {0};
    wheel_pi_default_cfg(&c);

    /* 1) 식 */
    wheel_pi_reset(&s);
    int16_t d = wheel_pi_step(&c, &s, 2.0f, 0.0f);
    /* r_m = 2*0.02/0.06 = 0.667, e = 0.667: ff 206.2+5+40, p 55.0, i 27.5 -> 333 */
    expect("first step duty ~333", d >= 331 && d <= 335, d);
    wheel_pi_reset(&s);
    d = wheel_pi_step(&c, &s, 2.0f, 0.5f);           /* 움직이는 중: 킥 없음. e = 0.667-0.5 -> 211.2+13.8+6.9 */
    expect("no kick when moving ~231", d >= 229 && d <= 233, d);
    wheel_pi_reset(&s);
    d = wheel_pi_step(&c, &s, -2.0f, 0.0f);
    expect("reverse is symmetric", d >= -335 && d <= -331, d);
    s.i = 50.0f; s.r_m = 1.0f;
    d = wheel_pi_step(&c, &s, 0.0f, 1.0f);
    expect("zero ref -> duty 0", d == 0, d);
    expect("zero ref resets state", s.i == 0.0f && s.r_m == 0.0f, s.i);
    wheel_pi_reset(&s);
    for (int k = 0; k < 20; k++) d = wheel_pi_step(&c, &s, 1.0f, 0.0f);    /* 막힌 바퀴 0.4 s */
    /* ff = 103.1+5+킥40 = 148, 보정 상한 250 -> 398 */
    expect("blocked 0.4 s: duty <= ff + corr_max (398)", d >= 396 && d <= 398, d);
    expect("blocked 0.4 s: not yet stall", s.stall == 0, s.stall);
    /* 보정이 상한에 닿는 데 ~5 주기, 거기서 0.5 s -> 시작부터 ~0.6 s */
    int k_stall = 20;
    while (!s.stall && k_stall < 100) { d = wheel_pi_step(&c, &s, 1.0f, 0.0f); k_stall++; }
    expect("blocked: stall at 0.55-0.65 s", s.stall == 1 && k_stall >= 28 && k_stall <= 32, k_stall * 0.02);
    d = wheel_pi_step(&c, &s, 1.0f, 0.0f);
    expect("stall: FF only, no kick (108)", d == 108, d);
    d = wheel_pi_step(&c, &s, 1.0f, 0.3f);
    expect("stall: wheel at 30% -> still FF", s.stall == 1 && d == 108, d);
    d = wheel_pi_step(&c, &s, 1.0f, 0.6f);
    expect("stall: wheel at 60% -> back to PI", s.stall == 0 && d != 108, d);
    for (int k = 0; k < 30; k++) wheel_pi_step(&c, &s, 1.0f, 0.0f);
    wheel_pi_step(&c, &s, 0.0f, 0.0f);
    expect("stall cleared by zero command", s.stall == 0, s.stall);

    wheel_pi_clear_fault(&s);
    for (int k = 0; k < 14; k++) d = wheel_pi_step(&c, &s, 2.0f, -1.0f);   /* 반대로 도는 바퀴 0.28 s */
    expect("reverse wheel 0.28 s: no trip yet", s.enc_trips == 0, s.enc_trips);
    d = wheel_pi_step(&c, &s, 2.0f, -1.0f);
    expect("reverse wheel 0.3 s: trip 1, retry hold (not locked)", s.enc_trips == 1 && s.enc_hold > 0 && !s.enc_dir,
           s.enc_trips);
    int k_hold = 0;
    while (s.enc_hold > 0 && k_hold < 100) { d = wheel_pi_step(&c, &s, 2.0f, 2.0f); k_hold++; }
    expect("retry hold: FF only (211) for 1 s", d == 211 && k_hold == 50, k_hold * 0.02);
    d = wheel_pi_step(&c, &s, 2.0f, 1.5f);                                /* 느림 -> PI 보정이 붙어야 */
    expect("after hold: PI again", d > 211 && !s.enc_dir, d);
    /* 일시적 역회전 한 번 (경사에서 밀림): 창 10 s 가 지나면 횟수도 비워짐 */
    for (int k = 0; k < 500; k++) wheel_pi_step(&c, &s, 2.0f, 2.0f);
    expect("one-off reverse: trips cleared after 10 s window", s.enc_trips == 0 && !s.enc_dir, s.enc_trips);
    /* 진짜 배선 문제: 재시도마다 다시 걸림 -> 3 번째에 잠금 */
    wheel_pi_clear_fault(&s);
    int k_lock = 0;
    while (!s.enc_dir && k_lock < 1000) { wheel_pi_step(&c, &s, 2.0f, -1.0f); k_lock++; }
    expect("persistent reverse: locked at 3rd trip (~2.9 s)", s.enc_dir && s.enc_trips == 3 && k_lock >= 140 && k_lock <= 150,
           k_lock * 0.02);
    d = wheel_pi_step(&c, &s, 2.0f, -1.0f);
    expect("locked: FF only (211)", d == 211, d);
    wheel_pi_step(&c, &s, 0.0f, 0.0f);
    d = wheel_pi_step(&c, &s, 2.0f, 2.0f);
    expect("lock survives zero command", s.enc_dir == 1 && d == 211, d);
    wheel_pi_clear_fault(&s);
    d = wheel_pi_step(&c, &s, 2.0f, 2.0f);
    expect("clear_fault -> PI again", s.enc_dir == 0 && s.enc_trips == 0 && d != 211, d);

    wheel_pi_reset(&s);
    for (int k = 0; k < 200; k++) d = wheel_pi_step(&c, &s, 8.0f, 0.0f);   /* 최고속 넘는 명령 */
    expect("saturated output clipped", d == 799, d);
    expect("no windup while saturated", s.i < 1.0f, s.i);
    d = wheel_pi_step(&c, &s, NAN, 0.0f);
    expect("NaN ref -> 0", d == 0, d);

    /* 2) 닫힌 루프 (공칭 플랜트) */
    sim(r2, NULL, 2.0, 1.0, 0.040, 47.0);
    expect("step 0->2: overshoot < 3%", (maxw(200, 2000) - 2.0) / 2.0 < 0.03, (maxw(200, 2000) - 2.0) / 2.0);
    expect("step 0->2: rise90 < 50 ms", rise_ms(200, 2000, 2.0) >= 0 && rise_ms(200, 2000, 2.0) < 50,
           rise_ms(200, 2000, 2.0));
    expect("step 0->2: steady error < 0.01", fabs(mean(1000, 2000) - 2.0) < 0.01, mean(1000, 2000) - 2.0);

    sim(r03, NULL, 3.0, 1.0, 0.040, 47.0);   /* 개루프 duty 31 이면 정지마찰에 막혀 안 도는 속도 */
    expect("low speed 0.3: starts < 60 ms", rise_ms(200, 3000, 0.3) >= 0 && rise_ms(200, 3000, 0.3) < 60,
           rise_ms(200, 3000, 0.3));
    expect("low speed 0.3: steady error < 0.01", fabs(mean(1000, 3000) - 0.3) < 0.01, mean(1000, 3000) - 0.3);

    sim(r2, NULL, 2.0, 0.85, 0.040, 47.0);   /* 배터리 -15% */
    expect("voltage -15%: steady error < 0.01", fabs(mean(1000, 2000) - 2.0) < 0.01, mean(1000, 2000) - 2.0);

    sim(r2, ld80, 3.0, 1.0, 0.040, 47.0);    /* 경사 부하 */
    expect("load 80 duty: drop < 0.5", 2.0 - minw(1000, 3000) < 0.5, 2.0 - minw(1000, 3000));
    expect("load 80 duty: steady error < 0.01", fabs(mean(2000, 3000) - 2.0) < 0.01, mean(2000, 3000) - 2.0);

    sim(r2, ldblock, 3.0, 1.0, 0.040, 47.0); /* 1 s 막혔다 풀림: 적분 상한으로 튐 제한 */
    expect("block release: peak < 3.3", maxw(2000, 3000) < 3.3, maxw(2000, 3000));

    sim(rrev, NULL, 2.0, 1.0, 0.040, 47.0);
    expect("reverse 2->-2: settles", fabs(mean(1500, 2000) + 2.0) < 0.01, mean(1500, 2000) + 2.0);
    expect("reverse 2->-2: |corr| max < corr_max", g_cmax < 250.0 && !g_prot, g_cmax);

    /* 정상 시험에서 보호가 안 걸리고, 보정량이 상한 아래인지 */
    {
        fn_t refs[3] = {r2, r03, r2};
        fn_t loads[3] = {NULL, NULL, ld80};
        const char *names[3] = {"0->2", "0->0.3", "load 80"};
        for (int k = 0; k < 3; k++)
        {
            char lbl[80];
            sim(refs[k], loads[k], 3.0, 1.0, 0.040, 47.0);
            snprintf(lbl, sizeof lbl, "%s: no protection trip, |corr| max < 250", names[k]);
            expect(lbl, g_cmax < 250.0 && !g_prot, g_cmax);
        }
        sim(r2, NULL, 2.0, 0.85, 0.040, 55.0);
        expect("voltage -15% static 55: no trip", g_cmax < 250.0 && !g_prot, g_cmax);
    }

    /* 엔코더 부호가 뒤집힌 배선: 보호 없으면 양의 되먹임으로 최고속 */
    g_enc_sign = -1;
    sim(r2, NULL, 4.0, 1.0, 0.040, 47.0);   /* 걸림 0.5 / 1.8 / 3.1 s -> 3 번째에 잠금 */
    g_enc_sign = 1;
    expect("reversed encoder: direction mismatch trips", (g_prot & 2) != 0, g_prot);
    expect("reversed encoder: speed stays < 5 rad/s (FF+250 at most)", maxw(0, 4000) < 5.0, maxw(0, 4000));
    expect("reversed encoder: locked, ends at FF speed ~2", fabs(mean(3500, 4000) - 2.0) < 0.3, mean(3500, 4000));

    /* 3) 플랜트 변동: τ 30/50 ms, 기울기 ×0.85/1.05 에서 오버슈트 < 8% */
    const double taus[2] = {0.030, 0.050}, ks[2] = {0.85, 1.05};
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
        {
            char lbl[64];
            sim(r2, NULL, 2.0, ks[b], taus[a], 47.0);
            snprintf(lbl, sizeof lbl, "robust tau %.0f ms k x%.2f: overshoot < 8%%", taus[a] * 1e3, ks[b]);
            expect(lbl, (maxw(200, 2000) - 2.0) / 2.0 < 0.08, (maxw(200, 2000) - 2.0) / 2.0);
        }

    printf(g_fail ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return g_fail;
}

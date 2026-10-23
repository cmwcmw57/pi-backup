/* ================================================================
 * imu_mean_time.h
 * 설명: MCU millisecond 읽기 시각을 누적해 평균과 제어 시각의 signed offset을 만든다.
 * 사용법: imu_time_add(&acc, read_ms); mean=imu_time_take(&acc);
 * ================================================================ */
#ifndef ROVER_IMU_MEAN_TIME_H
#define ROVER_IMU_MEAN_TIME_H
#include <stdint.h>
#include <string.h>

/* 평균 구간의 읽기 시각을 wrap에 안전한 상대값으로 보관한다. */
typedef struct {
    uint64_t sum_delta_ms;
    uint32_t first_ms, last_ms;
    uint16_t count;
    uint8_t bad;
} imu_time_accumulator_t;

/* 마지막 평균에 사용한 표본들의 대표 읽기 시각과 유효성을 보관한다. */
typedef struct {
    uint32_t first_ms, last_ms, mean_delta_us;
    uint8_t count, valid;
} imu_mean_time_t;

static inline void imu_time_add(imu_time_accumulator_t *a, uint32_t read_ms)
{
    if (a->count >= 255) { a->bad=1; return; }
    if (!a->count) a->first_ms=read_ms;
    uint32_t delta=read_ms-a->first_ms;
    if (delta > INT32_MAX || (a->count && (uint32_t)(read_ms-a->last_ms)>INT32_MAX)) a->bad=1;
    a->last_ms=read_ms;
    a->sum_delta_ms+=delta;
    a->count++;
}

static inline imu_mean_time_t imu_time_take(imu_time_accumulator_t *a)
{
    imu_mean_time_t out={0};
    if (a->count && !a->bad) {
        uint64_t mean=(a->sum_delta_ms*1000+a->count/2)/a->count;
        if (mean <= UINT32_MAX) {
            out.first_ms=a->first_ms; out.last_ms=a->last_ms;
            out.mean_delta_us=(uint32_t)mean; out.count=(uint8_t)a->count; out.valid=1;
        }
    }
    memset(a,0,sizeof *a);
    return out;
}

static inline int imu_mean_offset_us(const imu_mean_time_t *mean, uint32_t reference_ms, int16_t *offset)
{
    *offset=0;
    if (!mean->valid) return 0;
    uint32_t wrapped=mean->first_ms-reference_ms;
    int64_t delta=(wrapped <= INT32_MAX ? (int64_t)wrapped : (int64_t)wrapped-INT64_C(4294967296));
    int64_t us=delta*1000+mean->mean_delta_us;
    if (us < INT16_MIN || us > INT16_MAX) return 0;
    *offset=(int16_t)us;
    return 1;
}
#endif

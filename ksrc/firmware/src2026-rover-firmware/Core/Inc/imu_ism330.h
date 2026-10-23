/**
 * @file    imu_ism330.h
 * @brief   ISM330DHCX 6축 IMU (SPI3 마스터) 드라이버
 *
 * 핀 (rover_motor.ioc 와 동일):
 *   PC10 SPI3_SCK, PC11 SPI3_MISO, PC12 SPI3_MOSI, PC9 IMU_CS (GPIO, 평소 High)
 * 모드 0, 8비트, MSB first, fPCLK1/16 = 1MHz (센서 상한 10MHz).
 *
 * 설정: 가속도 208Hz ±4g (0.122 mg/LSB), 자이로 208Hz ±500dps (17.5 mdps/LSB),
 *       BDU=1 (상·하위 바이트가 다른 샘플에서 섞이지 않게), 주소 자동증가.
 * 50Hz 제어주기에서 가장 최근 샘플 하나를 읽는다. 값은 **센서 좌표계 원시값**이며
 * 차체 좌표계로의 회전(장착 방향)은 Pi 쪽에서 한다.
 *
 * SPI 마스터는 센서가 없어도 오류가 나지 않고 0x00/0xFF 를 읽는다. 그래서
 * WHO_AM_I(0x6B) 를 주기적으로 다시 확인해 유효 여부를 판단한다.
 *
 * CubeMX 가 만드는 MX_SPI3_Init() 은 호출하지 말 것 (레지스터로 직접 설정).
 */

#ifndef IMU_ISM330_H
#define IMU_ISM330_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/** SPI3·CS 핀 초기화 + 센서 리셋·설정.  @return 1 = WHO_AM_I 확인 성공 */
int Imu_Init(void);

/**
 * @brief  최신 샘플 읽기
 * @param  out  gyro x,y,z, acc x,y,z (원시 i16)
 * @return 1 = 유효, 0 = 센서 미확인 상태 (out 은 0)
 */
int Imu_Read(int16_t out[6]);

/**
 * @brief  WHO_AM_I 와 설정 레지스터 재확인. 어긋났으면(센서 단독 리셋 등) 바로 재설정한다.
 *         1초에 한 번 정도 부른다 (배선 흔들림·핫플러그 대비).
 * @return 1 = 유효
 */
int Imu_Check(void);

/**
 * 새 샘플(가속도·자이로 둘 다)이 있으면 읽어서 이번 제어주기 합에 더한다. 메인 루프에서 1 ms 마다.
 * ODR 208 Hz 라 20 ms 주기에 3–4 개. 서보 읽기(~3 ms) 동안 하나쯤 놓쳐도 평균은 유지된다.
 * 이유: 50 Hz 순간값 하나는 차체 진동이 겹쳐 보여(aliasing) 자이로 적분·중력 필터가 틀어진다
 * (시뮬 거친 지형: 순간값 적분 평균 24° 오차, 주기 평균이면 0.8°. MOTOR_MODELING_PLAN §11).
 * @param sample  NULL 이 아니면 이번에 읽은 원시 샘플 (ICAP 고속 기록용)
 * @return 이번에 더했으면 1
 */
int Imu_Poll(int16_t sample[6]);

/**
 * 이번 주기 평균을 꺼내고 합을 비운다. 쌓인 게 없으면 Imu_Read 순간값.
 * @param n  평균에 들어간 샘플 수 (0 = 순간값) -> 텔레메트리 imu_n
 * @return   IMU 유효 여부
 */
int Imu_TakeMean(int16_t out[6], uint8_t *n);

/* 평균 표본 읽기 시각의 offset. 입력 clock은 HAL_GetTick; 정확도는 us 단위와 다르다.
 * 범위 초과/표본 없음이면 0 반환·offset=0. 센서 내부 acquisition 시각은 아니다. */
int Imu_MeanReadOffsetUs(uint32_t reference_ms, int16_t *offset);

#ifdef __cplusplus
}
#endif

#endif /* IMU_ISM330_H */

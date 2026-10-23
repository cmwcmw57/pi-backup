/*
 * swerve_kinematics.h
 *
 * 4모듈 독립 구동+조향 스워브 역기구학.
 *
 * 순수 C 라서 HAL/보드 의존성이 전혀 없다 -> 호스트(gcc)에서 그대로 컴파일해
 * 테스트할 수 있고, STM32CubeIDE 의 Core/Src 에 그대로 떨어뜨려도 된다.
 *
 * 차체 좌표계 규약 (ROS REP-103 및 KSRC 제안서의 슬립 공식
 * u_i = (v_B + omega_z * k x r_i) * t_i 와 동일):
 *   x = 전진, y = 좌측, omega = +가 반시계 (위에서 봤을 때, +z 축 오른손 법칙).
 *
 * 모듈 장착 위치 (x, y) 는 로봇 회전중심(보통 기하중심/무게중심) 기준 미터 단위.
 */

#ifndef SWERVE_KINEMATICS_H
#define SWERVE_KINEMATICS_H

#ifdef __cplusplus
extern "C" {
#endif

#define SWERVE_NUM_MODULES 4

typedef struct {
    float x; /* m, 로봇 중심에서 +가 전진 */
    float y; /* m, 로봇 중심에서 +가 좌측 */
} swerve_module_pos_t;

/* 모듈별로 유지되는 상태. 각도 wrap 최적화와 영속도 조향 데드밴드에 필요하다.
 * 시작할 때 한 번 0 으로 초기화할 것
 * (예: `swerve_module_state_t state[SWERVE_NUM_MODULES] = {0};`). */
typedef struct {
    float last_angle_rad; /* 마지막으로 명령한 조향각(연속, wrap 안 됨) */
    float last_cmd_rad;   /* 마지막으로 **접어서** 내보낸 각 (가동범위 안) */
    int   initialized;    /* 첫 유효 명령 전까지 0 */
} swerve_module_state_t;

typedef struct {
    float angle_rad;  /* 목표 조향각(차체 기준). 임의의 실수값이며 wrap 되지
                          않는다. 서보 명령 범위로의 변환은 액추에이터
                          매핑 계층에서 할 것 */
    float speed_mps;  /* 부호 있는 바퀴 선속도. 각도 최적화가 180도 조향을
                          피하는 방식이 바로 이 부호 반전이다 */
} swerve_module_cmd_t;

/*
 * SWERVE_NUM_MODULES 개 모듈 전체의 역기구학을 푼다.
 *
 * vx, vy       : 차체 선속도 m/s (차체 좌표계, 위 규약 참고)
 * omega        : 차체 각속도 rad/s, +가 반시계
 * modules      : 모듈 장착 위치. 인덱스는 배선/번호 규칙에 맞춰 쓰면 된다
 * max_wheel_mps: 바퀴 속도 상한 m/s. 어느 한 모듈의 원해가 이를 넘으면
 *                **모든** 모듈 속도를 같은 비율로 줄여서 명령한 운동의
 *                모양을 보존한다 (스워브 표준 "desaturation").
 *                0 이하를 넘기면 비활성화.
 * state        : 모듈별 유지 상태(각도 이력). 매 호출에 같은 배열을 넘길 것
 * out          : 모듈별 결과 명령
 *
 * 조향각 최적화: 각 모듈에서 원 목표각(필요 속도벡터의 atan2)을 그 모듈의
 * 마지막 명령각과 비교한다. 거기까지 돌리는 데 90도를 넘게 회전해야 하면,
 * 각도를 180도 뒤집고 대신 바퀴 속도의 부호를 반전한다. 지면 접촉 속도벡터는
 * 동일하면서 조향 회전은 최대 90도로 줄어든다.
 *
 * 영속도 데드밴드: 요청된 차체 속도가 사실상 0 이면(|vx|,|vy|,|omega| 가 모두
 * 작은 epsilon 미만), 조향각을 0 으로 튕기지 않고 마지막 명령값으로 유지한다.
 * 조종자가 스틱/키에서 손을 뗐을 때 바퀴가 파르르 떠는 걸 막는다.
 */
void swerve_ik_compute(float vx, float vy, float omega,
                        const swerve_module_pos_t modules[SWERVE_NUM_MODULES],
                        float max_wheel_mps,
                        swerve_module_state_t state[SWERVE_NUM_MODULES],
                        swerve_module_cmd_t out[SWERVE_NUM_MODULES]);

/*
 * 조향 가동범위 안으로 접기.
 *
 * swerve_ik_compute 가 내는 angle_rad 는 **wrap 되지 않은 연속각**이라 여러
 * 기동을 거치며 무한정 흘러간다 (실측: 실제 경기장에서 67% 가 ±90도를 벗어남).
 * 그대로 자르면 평균 76.5도 오차가 난다.
 *
 * th 와 th±180도 는 바퀴 속도 부호를 뒤집으면 물리적으로 동일하다. 이 등가성으로
 * 가동범위 안에 넣되, **직전에 내보낸 각에 가장 가까운 대표값**을 고른다.
 *
 * 고정 창((-90,90] 같은)에 접으면 안 된다: 연속각이 89->91도로 2도 움직일 때
 * 명령이 +89 -> -89 로 178도 튀고, 서보가 6도/스텝이면 0.6초 동안 바퀴가 딴
 * 데를 본다. 실측하면 20초 에피소드의 24%를 그 강제 반전에 쓴다.
 *
 * limit_rad 는 **배선이 견디는 조향 가동범위**다 (서보 자체는 다회전 가능하며
 * sts3215_angle_rad_to_ticks_near 가 이음매를 알아서 처리한다). 슬립링이 있으면
 * 제한이 없으므로 이 함수를 부르지 않아도 된다. limit_rad 는 최소 PI/2 여야
 * 모든 방향을 표현할 수 있다.
 *
 * unwind_rad / slow_mps: 이음매(±limit) 처리.  범위가 유한하면 어딘가에 반드시
 * 이음매가 있고, 거기서는 180도 강제 반전을 피할 수 없다.  없앨 수는 없으니
 * **언제 치를지 고른다**: 조향각이 unwind_rad 를 넘었고 그 바퀴의 속도 명령이
 * slow_mps 미만이면, 0 에 가까운 등가각으로 미리 넘어간다.  구동 중이 아닐 때
 * 180도 돌리는 건 공짜다.
 * 안 하면 각이 한계 근처에 눌러앉는다 (실측: 시간의 10%를 |각|>170도 에서 보내고,
 * 90도 넘는 점프의 88%가 거기서 터졌다).  unwind_rad <= 0 이면 비활성화.
 */
void swerve_fold_to_limit(float limit_rad, float unwind_rad, float slow_mps,
                          swerve_module_state_t state[SWERVE_NUM_MODULES],
                          swerve_module_cmd_t out[SWERVE_NUM_MODULES]);

/*
 * swerve_fold_to_limit 의 바퀴별·비대칭 버전. 모듈 i 의 조향각을
 * [lo_rad[i], hi_rad[i]] 안으로 접는다 (swerve_fold_to_limit 은 모든 모듈에
 * [-limit, +limit] 를 넣어 이 함수를 부른다).
 *
 * 바퀴마다 배선·구동모터 위치가 달라 가동범위가 다를 때 쓴다.
 *   - 폭(hi - lo) >= PI 여야 모든 방향을 표현할 수 있다. 폭이 딱 PI 면 방향마다
 *     등가각이 하나뿐이라 이음매를 지날 때마다 180도 반전이 생긴다.
 *   - 폭 < PI 면 표현 불가능한 방향은 가까운 경계로 클램프한다 (기구 보호 우선).
 *   - unwind_rad 는 범위 **중앙**((lo+hi)/2)으로부터의 거리로 해석한다.
 */
void swerve_fold_to_range(const float lo_rad[SWERVE_NUM_MODULES],
                          const float hi_rad[SWERVE_NUM_MODULES],
                          float unwind_rad, float slow_mps,
                          swerve_module_state_t state[SWERVE_NUM_MODULES],
                          swerve_module_cmd_t out[SWERVE_NUM_MODULES]);

#ifdef __cplusplus
}
#endif

#endif /* SWERVE_KINEMATICS_H */

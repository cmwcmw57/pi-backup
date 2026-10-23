/*
 * Stage 1 -- prove firmware/common/*.c (already gcc-host-tested and
 * validated against real vendor protocols) compiles AND runs correctly
 * under STM32duino's ARM toolchain. Still touches zero real I/O
 * (no UART parsing of live bytes, no PWM, no servo bus) -- every call
 * below uses fixed, known inputs with precomputed expected outputs,
 * same values already checked on x86 host in firmware/common/tests/.
 *
 * The PRIMARY pass criterion for Stage 1 is just: this compiles
 * (Sketch > Verify, no board/upload needed). If you do have the board
 * connected and flash this, it also re-checks the results at runtime
 * and reports pass/fail two ways:
 *   - LED_BUILTIN: fast blink (10Hz) = all checks passed,
 *                  slow blink (1Hz)  = something failed
 *   - Serial (115200): prints each check's result if a monitor is open
 *
 * Uses the ksrc_common library (symlinked at
 * ~/Arduino/libraries/ksrc_common/src -> ~/ksrc/firmware/common).
 */
#include <math.h>
#include <string.h>
#include "swerve_kinematics.h"
#include "teleop_protocol.h"
#include "sts3215_protocol.h"
#include "mdd3a_driver.h"

bool g_all_passed = true;

void check(const char *label, bool cond) {
  if (Serial) {
    Serial.print(cond ? "ok   " : "FAIL ");
    Serial.println(label);
  }
  if (!cond) g_all_passed = false;
}

bool near(float a, float b, float tol) {
  return fabsf(a - b) <= tol;
}

void run_checks() {
  // --- swerve_kinematics: pure forward, matches
  // firmware/common/tests/test_swerve_kinematics.c case 1 ---
  {
    swerve_module_pos_t modules[SWERVE_NUM_MODULES] = {
      {0.12f, 0.12f}, {0.12f, -0.12f}, {-0.12f, 0.12f}, {-0.12f, -0.12f}
    };
    swerve_module_state_t state[SWERVE_NUM_MODULES] = {0};
    swerve_module_cmd_t out[SWERVE_NUM_MODULES];
    swerve_ik_compute(1.0f, 0.0f, 0.0f, modules, -1.0f, state, out);
    bool ok = true;
    for (int i = 0; i < SWERVE_NUM_MODULES; i++) {
      ok = ok && near(out[i].angle_rad, 0.0f, 1e-4f) && near(out[i].speed_mps, 1.0f, 1e-4f);
    }
    check("swerve_ik_compute: pure forward", ok);
  }

  // --- teleop_protocol: encode -> parse -> decode round trip against
  // the known cross-checked frame (vx=1.5, vy=-2.25, omega=3.0) ---
  {
    uint8_t buf[TELEOP_VELOCITY_CMD_FRAME_LEN];
    int n = teleop_encode_velocity_cmd(1.5f, -2.25f, 3.0f, buf, sizeof buf);
    teleop_parser_t p;
    teleop_frame_t frame;
    teleop_velocity_cmd_t cmd;
    teleop_parser_init(&p);
    bool decoded = false;
    for (int i = 0; i < n; i++) {
      if (teleop_parser_feed_byte(&p, buf[i], &frame)) decoded = true;
    }
    bool ok = decoded && teleop_decode_velocity_cmd(&frame, &cmd)
              && near(cmd.vx, 1.5f, 1e-5f) && near(cmd.vy, -2.25f, 1e-5f) && near(cmd.omega, 3.0f, 1e-5f);
    check("teleop_protocol: encode/parse/decode round trip", ok);
  }

  // --- sts3215_protocol: known byte vector, id=1 pos=2048 speed=1000 acc=50 ---
  {
    uint8_t buf[32];
    int n = sts3215_build_write_pos_packet(1, 2048, 1000, 50, buf, sizeof buf);
    uint8_t want[] = {0xff,0xff,0x01,0x0a,0x03,0x29,0x32,0x00,0x08,0x00,0x00,0xe8,0x03,0xa3};
    bool ok = (n == (int)sizeof(want)) && (memcmp(buf, want, sizeof want) == 0);
    check("sts3215_protocol: write_pos byte vector", ok);
  }

  // --- mdd3a_driver: forward/reverse mutual exclusion ---
  {
    mdd3a_channel_cmd_t fwd = mdd3a_speed_to_duty(0.6f);
    mdd3a_channel_cmd_t rev = mdd3a_speed_to_duty(-0.6f);
    bool ok = near(fwd.duty_a, 0.6f, 1e-6f) && near(fwd.duty_b, 0.0f, 1e-6f)
              && near(rev.duty_a, 0.0f, 1e-6f) && near(rev.duty_b, 0.6f, 1e-6f);
    check("mdd3a_driver: fwd/rev duty", ok);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) {
    // give a monitor up to 2s to attach; proceed either way
  }
  run_checks();
}

void loop() {
  int delay_ms = g_all_passed ? 100 : 500;
  digitalWrite(LED_BUILTIN, HIGH);
  delay(delay_ms);
  digitalWrite(LED_BUILTIN, LOW);
  delay(delay_ms);
}

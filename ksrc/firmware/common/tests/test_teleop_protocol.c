/*
 * 텔레옵 프로토콜 호스트 테스트.
 *   gcc -std=c99 -Wall -Wextra -o /tmp/teleop_test test_teleop_protocol.c teleop_protocol.c && /tmp/teleop_test
 *
 * --hexdump 옵션도 있다: /tmp/teleop_test --hexdump
 * 고정 테스트 벡터(vx=1.5, vy=-2.25, omega=3.0)의 인코딩 결과를 16진수로
 * 출력한다. pi/common/nucleo_link.py 의 파이썬 인코더와 바이트 단위로
 * 대조하기 위한 것이다.
 */
#include "teleop_protocol.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;

static void expect_near(const char *label, float got, float want, float tol)
{
    if (fabsf(got - want) > tol) {
        printf("FAIL %s: got %.6f want %.6f\n", label, got, want);
        g_fail = 1;
    } else {
        printf("ok   %s: %.6f\n", label, got);
    }
}

static void expect_eq_int(const char *label, int got, int want)
{
    if (got != want) {
        printf("FAIL %s: got %d want %d\n", label, got, want);
        g_fail = 1;
    } else {
        printf("ok   %s: %d\n", label, got);
    }
}

static void feed_frame_and_expect(teleop_parser_t *p, const uint8_t *buf, int len,
                                   int expect_success)
{
    teleop_frame_t frame;
    int i, got = 0;
    for (i = 0; i < len; i++) {
        if (teleop_parser_feed_byte(p, buf[i], &frame)) {
            got = 1;
        }
    }
    expect_eq_int("frame_decoded", got, expect_success);
}

static void print_hexdump(void)
{
    uint8_t buf[TELEOP_VELOCITY_CMD_FRAME_LEN];
    int n = teleop_encode_velocity_cmd(1.5f, -2.25f, 3.0f, buf, sizeof buf);
    int i;
    for (i = 0; i < n; i++) {
        printf("%02x ", buf[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--hexdump") == 0) {
        print_hexdump();
        return 0;
    }

    /* 1) round trip: encode then decode gives back the same floats */
    {
        uint8_t buf[TELEOP_VELOCITY_CMD_FRAME_LEN];
        teleop_parser_t p;
        teleop_frame_t frame;
        teleop_velocity_cmd_t cmd;
        int n, i, decoded = 0;

        n = teleop_encode_velocity_cmd(0.75f, -1.2f, 2.5f, buf, sizeof buf);
        expect_eq_int("encode_len", n, TELEOP_VELOCITY_CMD_FRAME_LEN);

        teleop_parser_init(&p);
        for (i = 0; i < n; i++) {
            if (teleop_parser_feed_byte(&p, buf[i], &frame)) decoded = 1;
        }
        expect_eq_int("roundtrip_decoded", decoded, 1);
        expect_eq_int("roundtrip_decode_velocity", teleop_decode_velocity_cmd(&frame, &cmd), 1);
        expect_near("roundtrip.vx", cmd.vx, 0.75f, 1e-6f);
        expect_near("roundtrip.vy", cmd.vy, -1.2f, 1e-6f);
        expect_near("roundtrip.omega", cmd.omega, 2.5f, 1e-6f);
        expect_eq_int("no_crc_errors", (int)p.crc_error_count, 0);
        expect_eq_int("no_framing_errors", (int)p.framing_error_count, 0);
    }

    /* 2) garbage bytes before a valid frame must not break the parser */
    {
        uint8_t garbage[] = {0x00, 0xFF, 0xAA, 0x12, 0x55, 0xAA}; /* junk incl. lone sync bytes */
        uint8_t buf[TELEOP_VELOCITY_CMD_FRAME_LEN];
        uint8_t full[sizeof(garbage) + TELEOP_VELOCITY_CMD_FRAME_LEN];
        teleop_parser_t p;

        teleop_encode_velocity_cmd(1.0f, 0.0f, 0.0f, buf, sizeof buf);
        memcpy(full, garbage, sizeof garbage);
        memcpy(full + sizeof garbage, buf, sizeof buf);

        teleop_parser_init(&p);
        feed_frame_and_expect(&p, full, sizeof full, 1);
    }

    /* 3) corrupted CRC must be rejected, not silently accepted */
    {
        uint8_t buf[TELEOP_VELOCITY_CMD_FRAME_LEN];
        teleop_parser_t p;

        teleop_encode_velocity_cmd(1.0f, 2.0f, 3.0f, buf, sizeof buf);
        buf[10] ^= 0xFF; /* flip a payload byte, leave CRC as-is -> mismatch */

        teleop_parser_init(&p);
        feed_frame_and_expect(&p, buf, sizeof buf, 0);
        expect_eq_int("crc_error_counted", (int)p.crc_error_count, 1);
    }

    /* 4) parser resyncs after a rejected frame and decodes the next one */
    {
        uint8_t good[TELEOP_VELOCITY_CMD_FRAME_LEN];
        uint8_t bad[TELEOP_VELOCITY_CMD_FRAME_LEN];
        uint8_t stream[2 * TELEOP_VELOCITY_CMD_FRAME_LEN];
        teleop_parser_t p;
        teleop_frame_t frame;
        teleop_velocity_cmd_t cmd;
        int i, count = 0;

        teleop_encode_velocity_cmd(9.0f, 9.0f, 9.0f, bad, sizeof bad);
        bad[9] ^= 0x01; /* corrupt -> should be dropped */
        teleop_encode_velocity_cmd(4.0f, 5.0f, 6.0f, good, sizeof good);

        memcpy(stream, bad, sizeof bad);
        memcpy(stream + sizeof bad, good, sizeof good);

        teleop_parser_init(&p);
        for (i = 0; i < (int)sizeof stream; i++) {
            if (teleop_parser_feed_byte(&p, stream[i], &frame)) {
                count++;
                teleop_decode_velocity_cmd(&frame, &cmd);
            }
        }
        expect_eq_int("resync_frame_count", count, 1);
        expect_near("resync.vx", cmd.vx, 4.0f, 1e-6f);
        expect_near("resync.vy", cmd.vy, 5.0f, 1e-6f);
        expect_near("resync.omega", cmd.omega, 6.0f, 1e-6f);
    }

    /* 5) watchdog staleness, including rollover safety */
    {
        expect_eq_int("watchdog_fresh", teleop_link_is_stale(1000, 900, 300), 0);
        expect_eq_int("watchdog_stale", teleop_link_is_stale(1500, 900, 300), 1);
        /* now_ms has wrapped past last_valid_frame_ms (near UINT32_MAX) */
        expect_eq_int("watchdog_rollover_fresh",
                       teleop_link_is_stale(50u, 4294967200u /* ~96ms before wrap */, 300), 0);
    }

    if (g_fail) {
        printf("\nSOME TESTS FAILED\n");
        return 1;
    }
    printf("\nALL TESTS PASSED\n");
    return 0;
}

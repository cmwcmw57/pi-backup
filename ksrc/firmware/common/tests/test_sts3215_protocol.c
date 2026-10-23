/*
 * gcc -std=c99 -Wall -Wextra -o /tmp/sts_test test_sts3215_protocol.c sts3215_protocol.c -lm && /tmp/sts_test
 *
 * STS3215 패킷 조립 호스트 테스트.
 *
 * 아래 기대 바이트 벡터는 FEETECH 공식 SCS.cpp/SMS_STS.cpp 수식을 파이썬으로
 * 처음부터 재구현해서 **독립적으로 교차검증**한 값이다. "이 C 코드가 자기
 * 자신과 일치한다" 수준이 아니다.
 */
#include "sts3215_protocol.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define TEST_PI 3.14159265358979323846f

static int g_fail = 0;

static void expect_bytes(const char *label, const uint8_t *got, int got_len,
                          const uint8_t *want, int want_len)
{
    if (got_len != want_len || memcmp(got, want, (size_t)want_len) != 0) {
        int i;
        printf("FAIL %s: got  ", label);
        for (i = 0; i < got_len; i++) printf("%02x ", got[i]);
        printf("\n         want ");
        for (i = 0; i < want_len; i++) printf("%02x ", want[i]);
        printf("\n");
        g_fail = 1;
    } else {
        printf("ok   %s (%d bytes)\n", label, got_len);
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

int main(void)
{
    /* 1) WritePosEx equivalent: id=1 pos=2048 speed=1000 acc=50 */
    {
        uint8_t buf[32];
        int n = sts3215_build_write_pos_packet(1, 2048, 1000, 50, buf, sizeof buf);
        uint8_t want[] = {0xff,0xff,0x01,0x0a,0x03,0x29,0x32,0x00,0x08,0x00,0x00,0xe8,0x03,0xa3};
        expect_bytes("write_pos", buf, n, want, sizeof want);
    }

    /* 2) torque enable: id=1 enable=1 */
    {
        uint8_t buf[32];
        int n = sts3215_build_torque_enable_packet(1, 1, buf, sizeof buf);
        uint8_t want[] = {0xff,0xff,0x01,0x04,0x03,0x28,0x01,0xce};
        expect_bytes("torque_enable", buf, n, want, sizeof want);
    }

    /* 3) ping: id=1 */
    {
        uint8_t buf[32];
        int n = sts3215_build_ping_packet(1, buf, sizeof buf);
        uint8_t want[] = {0xff,0xff,0x01,0x02,0x01,0xfb};
        expect_bytes("ping", buf, n, want, sizeof want);
    }

    /* 4) angle -> ticks: 0, +180deg, -90deg (wraps to 270deg), full turn wraps to 0 */
    expect_eq_int("angle_0deg",    sts3215_angle_rad_to_ticks(0.0f), 0);
    expect_eq_int("angle_180deg",  sts3215_angle_rad_to_ticks(TEST_PI), 2048);
    expect_eq_int("angle_neg90deg_wraps_270", sts3215_angle_rad_to_ticks(-TEST_PI/2.0f), 3072);
    expect_eq_int("angle_360deg_wraps_0", sts3215_angle_rad_to_ticks(2.0f*TEST_PI), 0);

    /* 5) reply parser: round trip against the known-good ack frame */
    {
        uint8_t frame[] = {0xff,0xff,0x01,0x02,0x00,0xfc}; /* id=1 status=0, no params */
        sts3215_reply_parser_t p;
        sts3215_reply_t reply;
        int i, decoded = 0;

        sts3215_reply_parser_init(&p);
        for (i = 0; i < (int)sizeof frame; i++) {
            if (sts3215_reply_parser_feed_byte(&p, frame[i], &reply)) decoded = 1;
        }
        expect_eq_int("reply_decoded", decoded, 1);
        expect_eq_int("reply.id", reply.id, 1);
        expect_eq_int("reply.status", reply.status, 0);
        expect_eq_int("reply.param_len", reply.param_len, 0);
    }

    /* 6) reply parser resyncs past garbage and rejects corrupted CRC */
    {
        uint8_t garbage[] = {0x00, 0xFF, 0x12, 0x34}; /* deliberately does not end in FF FF */
        uint8_t bad[] = {0xff,0xff,0x02,0x02,0x00,0xfb}; /* id=2, but corrupt status below */
        uint8_t good[] = {0xff,0xff,0x03,0x02,0x00,0xfa}; /* id=3, valid */
        uint8_t stream[64];
        int off = 0, i, count = 0;
        sts3215_reply_parser_t p;
        sts3215_reply_t reply;

        bad[4] ^= 0x01; /* corrupt status byte, checksum now mismatches */

        memcpy(stream+off, garbage, sizeof garbage); off += (int)sizeof garbage;
        memcpy(stream+off, bad, sizeof bad); off += (int)sizeof bad;
        memcpy(stream+off, good, sizeof good); off += (int)sizeof good;

        sts3215_reply_parser_init(&p);
        for (i = 0; i < off; i++) {
            if (sts3215_reply_parser_feed_byte(&p, stream[i], &reply)) count++;
        }
        expect_eq_int("resync_decoded_count", count, 1);
        expect_eq_int("resync_decoded_id", reply.id, 3);
        expect_eq_int("resync_crc_error_seen", (int)p.crc_error_count >= 1, 1);
    }

    if (g_fail) {
        printf("\nSOME TESTS FAILED\n");
        return 1;
    }
    printf("\nALL TESTS PASSED\n");
    return 0;
}

# ================================================================
# build_patch.py
# 설명: 고정 KSRC baseline에서 기존 wire 길이를 유지하는 IMU 읽기 시각 패치를 만든다.
# 사용법: python build_patch.py /path/to/ksrc-baseline [--profile latest --output new.patch]
# ================================================================
import difflib
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def replace_once(text: str, old: str, new: str) -> str:
    """예상 source 한 곳만 바꿔 다른 firmware revision에 잘못 적용하지 않는다."""
    if text.count(old)!=1: raise ValueError(f'baseline context mismatch: {old[:60]}')
    return text.replace(old,new,1)


def main(baseline: Path, profile: str = 'legacy', output: Path | None = None) -> None:
    """baseline 파일은 읽기만 하고 portable header와 변경 diff만 출력한다."""
    manifest=json.loads((HERE/('LATEST_BASELINE.json' if profile == 'latest' else 'BASELINE.json')).read_text())
    original={}
    for name,checksum in manifest['sha256'].items():
        data=(baseline/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=checksum: raise ValueError(f'baseline hash mismatch: {name}')
        original[name]=data.decode()
    changed=dict(original)
    header='firmware/src2026-rover-firmware/Core/Inc/imu_mean_time.h'
    changed[header]=(HERE/'imu_mean_time.h').read_text()
    name='firmware/src2026-rover-firmware/Core/Src/imu_ism330.c'
    s=changed[name]
    s=replace_once(s,'#include "imu_ism330.h"','#include "imu_ism330.h"\n#include "imu_mean_time.h"')
    s=replace_once(s,'static int32_t s_sum[6];','static int32_t s_sum[6];\nstatic imu_time_accumulator_t s_read_times;\nstatic imu_mean_time_t s_last_mean_time;')
    s=replace_once(s,'    if (!Imu_Read(v)) return 0;\n    if (s_n == 255)',
                   '    if (!Imu_Read(v)) return 0;\n    uint32_t read_ms = HAL_GetTick();\n    if (s_n == 255)')
    s=replace_once(s,'    s_n++;\n    if (sample != NULL)','    s_n++;\n    imu_time_add(&s_read_times, read_ms);\n    if (sample != NULL)')
    s=replace_once(s,'        return Imu_Read(out);          /* 이번 주기에 못 쌓았으면 순간값 하나 */',
'''        int ok = Imu_Read(out);       /* 순간값의 읽기 완료 시각; acquisition 시각이 아님 */
        imu_time_accumulator_t instant = {0};
        if (ok) imu_time_add(&instant, HAL_GetTick());
        s_last_mean_time = imu_time_take(&instant);
        return ok;''')
    s=replace_once(s,'    s_n = 0;\n    return s_valid;','    s_last_mean_time = imu_time_take(&s_read_times);\n    s_n = 0;\n    return s_valid;')
    s+='''
/* 마지막 평균의 대표 읽기 시각 - 제어 reference, microsecond 단위. */
int Imu_MeanReadOffsetUs(uint32_t reference_ms, int16_t *offset)
{
    return imu_mean_offset_us(&s_last_mean_time, reference_ms, offset);
}
'''
    changed[name]=s
    name='firmware/src2026-rover-firmware/Core/Inc/imu_ism330.h'
    changed[name]=replace_once(changed[name],'int Imu_TakeMean(int16_t out[6], uint8_t *n);',
'''int Imu_TakeMean(int16_t out[6], uint8_t *n);

/* 평균 표본 읽기 시각의 offset. 입력 clock은 HAL_GetTick; 정확도는 us 단위와 다르다.
 * 범위 초과/표본 없음이면 0 반환·offset=0. 센서 내부 acquisition 시각은 아니다. */
int Imu_MeanReadOffsetUs(uint32_t reference_ms, int16_t *offset);''')
    name='firmware/common/spi_link.h'
    s=changed[name]
    if profile == 'latest':
        s=replace_once(s,' *   [140] wpi_corr u16         지금 바퀴 PI 보정량 상한 [duty] (10-14, 타입 0x03 확인용. 이전 펌웨어는 0)',
                       ' *   [140] tagged u16/i16       bit12=0: wpi_corr; bit12=1: imu_mean_read_offset_us (PI readback 없음)')
    else:
        s=replace_once(s,' *   [140..141] 예약 (0)',' *   [140..141] imu_mean_read_offset_us i16 (IMU_READ_TIME_VALID flag일 때만 유효)')
    s=replace_once(s,'typedef enum {','#define SPI_TLM_FLAG_IMU_READ_TIME_VALID (1u << 12) /* MCU 평균 표본 읽기 시각 유효 */\n\ntypedef enum {')
    s=replace_once(s,'} spi_link_telemetry_t;','    int16_t imu_mean_read_offset_us; /* t_ms 기준 signed us, 실제 tick 정밀도는 1ms */\n} spi_link_telemetry_t;')
    changed[name]=s
    name='firmware/common/spi_link.c'
    s=changed[name]
    if profile == 'latest':
        s=replace_once(s,'    put_u16(&frame[140], t->wpi_corr);',
'''    put_u16(&frame[140], (t->flags & SPI_TLM_FLAG_IMU_READ_TIME_VALID)
        ? (uint16_t)t->imu_mean_read_offset_us : t->wpi_corr);''')
        s=replace_once(s,'    out->wpi_corr = get_u16(&frame[140]);',
'''    out->wpi_corr = (out->flags & SPI_TLM_FLAG_IMU_READ_TIME_VALID) ? 0 : get_u16(&frame[140]);
    out->imu_mean_read_offset_us = (out->flags & SPI_TLM_FLAG_IMU_READ_TIME_VALID)
        ? (int16_t)get_u16(&frame[140]) : 0;''')
    else:
        s=replace_once(s,'    frame[139] = t->imu_n;','    frame[139] = t->imu_n;\n    put_u16(&frame[140], (uint16_t)t->imu_mean_read_offset_us);')
        s=replace_once(s,'    out->imu_n = frame[139];','    out->imu_n = frame[139];\n    out->imu_mean_read_offset_us = (int16_t)get_u16(&frame[140]);')
    changed[name]=s
    name='firmware/src2026-rover-firmware/Core/Src/main.c'
    s=changed[name]
    s=replace_once(s,'        memset(&t, 0, sizeof t);',
'''        memset(&t, 0, sizeof t);
        int imu_read_time_valid = s_imu_valid &&
            Imu_MeanReadOffsetUs(now, &t.imu_mean_read_offset_us);''')
    s=replace_once(s,'            (s_res_active       ? SPI_TLM_FLAG_RES_ACTIVE     : 0u));',
'''            (s_res_active       ? SPI_TLM_FLAG_RES_ACTIVE     : 0u) |
            (imu_read_time_valid ? SPI_TLM_FLAG_IMU_READ_TIME_VALID : 0u));''')
    changed[name]=s
    name='pi/common/nucleo_link.py'
    s=changed[name]
    s=replace_once(s,'# [2]..[141] 영역:', 'TLM_FLAG_IMU_READ_TIME_VALID = 1 << 12  # 평균 표본의 MCU 읽기 시각 offset\n\n# [2]..[141] 영역:')
    if profile == 'latest':
        s = '''# ================================================================
# nucleo_link.py
# 설명: KSRC serial/SPI 프레임을 처리하고 timing flag로 IMU 시각과 PI readback을 구분한다.
# 사용법: from nucleo_link import SpiLink, decode_spi_telemetry
# ================================================================
''' + s
        s=replace_once(s,'        "wpi_corr": v[58],',
                       '        "wpi_corr": None if v[4] & TLM_FLAG_IMU_READ_TIME_VALID else v[58],')
        s=replace_once(s,'        "imu_n": v[57],',
'''        "imu_timing_valid": (v[4] & (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID)) == (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID),
        "imu_mean_read_offset_us": struct.unpack_from("<h", frame, 140)[0] if (v[4] & (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID)) == (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID) else None,
        "imu_time_source": "mcu_mean_sample_read" if (v[4] & (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID)) == (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID) else None,
        "imu_n": v[57],''')
    else:
        s=replace_once(s,'4B4B4BBB2x','4B4B4BBBh')
        s=replace_once(s,'        "imu_n": v[57],',
'''        "imu_timing_valid": (v[4] & (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID)) == (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID),
        "imu_mean_read_offset_us": v[58] if (v[4] & (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID)) == (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID) else None,
        "imu_time_source": "mcu_mean_sample_read" if (v[4] & (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID)) == (TLM_FLAG_IMU_VALID | TLM_FLAG_IMU_READ_TIME_VALID) else None,
        "imu_n": v[57],''')
    changed[name]=s
    if profile == 'latest':
        # board가 실제 빌드하는 사본도 common과 같은 내용으로 유지한다.
        for suffix, folder in (('c', 'Src'), ('h', 'Inc')):
            changed[f'firmware/src2026-rover-firmware/Core/{folder}/spi_link.{suffix}'] = changed[f'firmware/common/spi_link.{suffix}']
        for name in ('pi/common/fake_nucleo.py', 'pi/common/test_spi_link.py'):
            changed[name] = replace_once(changed[name], '("wpi_corr", ctypes.c_uint16)]',
                '("wpi_corr", ctypes.c_uint16), ("imu_mean_read_offset_us", ctypes.c_int16)]')
        name = 'pi/common/test_spi_link.py'
        changed[name] = replace_once(changed[name], '        t.wpi_corr = rng.randrange(800)',
            '        t.wpi_corr = rng.randrange(800)\n        t.imu_mean_read_offset_us = rng.randrange(-32768, 32768)')
        changed[name] = replace_once(changed[name], '        assert d["wpi_corr"] == t.wpi_corr',
'''        tagged = bool(t.flags & nl.TLM_FLAG_IMU_READ_TIME_VALID)
        valid = tagged and bool(t.flags & nl.TLM_FLAG_IMU_VALID)
        assert d["wpi_corr"] == (None if tagged else t.wpi_corr)
        assert d["imu_timing_valid"] == valid
        assert d["imu_mean_read_offset_us"] == (t.imu_mean_read_offset_us if valid else None)''')
    parts=[]
    for name in sorted(changed):
        parts.extend(difflib.unified_diff(original.get(name,'').splitlines(True),changed[name].splitlines(True),
                     fromfile='a/'+name if name in original else '/dev/null',tofile='b/'+name))
    target = output or HERE/('ksrc_imu_mean_time_latest.patch' if profile == 'latest' else 'ksrc_imu_mean_time.patch')
    target.write_text(''.join(parts))
    print(f'Generated patch for {len(changed)} files; baseline preserved')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--profile', choices=['legacy', 'latest'], default='legacy')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.baseline, args.profile, args.output)

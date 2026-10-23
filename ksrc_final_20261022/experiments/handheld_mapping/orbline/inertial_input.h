// ================================================================
// inertial_input.h
// 설명: offline IMU CSV 표본을 검증하고 영상별 적분 배치를 구성한다.
// 사용법: #include "inertial_input.h"
// ================================================================
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rover {
struct ImuSample {
    // SI 단위의 각속도·가속도와 원본 nanosecond 시각을 보관한다.
    std::int64_t stamp_ns;
    std::array<float, 3> gyro, accel;
};
inline std::vector<ImuSample> read_imu_csv(std::istream& input) {
    std::string line;
    std::getline(input, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "timestamp_ns,gx_rad_s,gy_rad_s,gz_rad_s,ax_m_s2,ay_m_s2,az_m_s2")
        throw std::runtime_error("IMU CSV header does not match SI-unit contract");
    std::vector<ImuSample> samples;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream row(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(row, field, ',')) fields.push_back(field);
        if (!line.empty() && line.back() == ',') fields.emplace_back();
        if (fields.size() != 7) throw std::runtime_error("IMU CSV requires seven columns");
        ImuSample sample;
        std::size_t used = 0;
        try { sample.stamp_ns = std::stoll(fields[0], &used); }
        catch (const std::exception&) { throw std::runtime_error("invalid IMU timestamp"); }
        if (used != fields[0].size() || sample.stamp_ns < 0)
            throw std::runtime_error("invalid IMU timestamp");
        for (int i = 0; i < 6; ++i) {
            float value;
            try { value = std::stof(fields[i + 1], &used); }
            catch (const std::exception&) { throw std::runtime_error("invalid IMU numeric value"); }
            if (used != fields[i + 1].size()) throw std::runtime_error("invalid IMU numeric value");
            if (!std::isfinite(value)) throw std::runtime_error("IMU values must be finite");
            if (i < 3) sample.gyro[i] = value;
            else sample.accel[i - 3] = value;
        }
        if (!samples.empty()) {
            if (sample.stamp_ns <= samples.back().stamp_ns)
                throw std::runtime_error("IMU timestamps must be strictly increasing");
            if (sample.stamp_ns - samples.back().stamp_ns > 100000000)
                throw std::runtime_error("IMU timestamp gap exceeds 100 ms");
        }
        samples.push_back(sample);
    }
    if (samples.empty()) throw std::runtime_error("IMU CSV is empty");
    return samples;
}
class ImuBatches {
    // 이전 영상 경계 표본을 이어서 프레임별 IMU 구간을 제공한다.
public:
    explicit ImuBatches(const std::vector<ImuSample>& samples) : samples_(samples) {}
    std::vector<ImuSample> next_for_orb(std::int64_t stamp_ns) {
        if (started_ && stamp_ns <= previous_ns_)
            throw std::runtime_error("frame timestamps must be strictly increasing");
        const auto right = std::upper_bound(samples_.begin(), samples_.end(), stamp_ns,
            [](std::int64_t stamp, const ImuSample& sample) { return stamp < sample.stamp_ns; });
        if (right == samples_.end()) throw std::runtime_error("IMU coverage requires a sample after frame timestamp");
        if (right - samples_.begin() < 2) throw std::runtime_error("first frame requires two preceding IMU samples");
        auto begin = started_ ? samples_.begin() + submitted_end_ : right - 2;
        const auto end = right + 1;
        if (started_) {
            const auto lower = [this](std::int64_t stamp) {
                return std::lower_bound(samples_.begin(), samples_.end(), stamp,
                    [](const ImuSample& sample, std::int64_t time) { return sample.stamp_ns < time; });
            };
            // pinned core는 이전영상-1ms 이전을 버리고 현재영상-1ms 이상에서 멈춘다.
            // 실제 적분 구간이 부족한 영상은 queue 상태를 바꾸지 않고 건너뛴다.
            if (lower(stamp_ns - 1000000) - lower(previous_ns_ - 1000000) < 1 || end <= begin)
                return {};
        }
        submitted_end_ = static_cast<std::size_t>(end - samples_.begin());
        previous_ns_ = stamp_ns;
        started_ = true;
        // 현재영상 우측의 실제 표본을 미리 넣되 다음 영상에서 다시 넣지 않는다.
        return {begin, end};
    }
private:
    const std::vector<ImuSample>& samples_;
    std::size_t submitted_end_ = 0;
    std::int64_t previous_ns_ = 0;
    bool started_ = false;
};
}

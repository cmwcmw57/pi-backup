// ================================================================
// runner.cpp
// 설명: rectified stereo 파일 또는 point stdin을 추적하고 pose·최종 keyframe을 내보낸다.
// 사용법: rover_orb3_compare ORBvoc LSDvoc settings.yaml pairs.tsv|- output_dir [imu.csv]
// ================================================================
#include <System.h>
#include <KeyFrame.h>
#include <Map.h>
#include <MapPoint.h>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef ROVER_POINT_BASELINE
#include "inertial_input.h"
#include "map_export.h"
// 오래된 runtime이 복구 설정을 조용히 무시하지 않도록 실행 시 지원을 확인한다.
extern "C" bool orbslam3_has_stereo_recovery_wait() __attribute__((weak));
extern "C" bool orbslam3_has_initial_stereo_relocalization() __attribute__((weak));
extern "C" bool orbslam3_has_safe_presave_observations() __attribute__((weak));
#endif

int main(int argc, char** argv) {
#ifdef ROVER_POINT_BASELINE
    if (argc != 6 && argc != 7) { std::cerr << "ORBvoc LSDvoc yaml pairs.tsv output_dir [imu.csv]\n"; return 2; }
#else
    if (argc != 6) { std::cerr << "ORBvoc LSDvoc yaml pairs.tsv output_dir\n"; return 2; }
#endif
    try {
        const bool live = std::string(argv[4]) == "-";
#ifndef ROVER_POINT_BASELINE
        if (live) throw std::runtime_error("stdin live mode requires point baseline");
#endif
#ifdef ROVER_POINT_BASELINE
        const bool imu_requested = argc == 7;
#ifdef ROVER_IMU_ONLY
        if (!imu_requested) throw std::runtime_error("IMU comparison runner requires imu.csv");
#endif
        if (live && imu_requested) throw std::runtime_error("IMU CSV requires offline stereo input");
        std::vector<rover::ImuSample> imu_samples;
        if (imu_requested) {
            std::ifstream imu_file(argv[6]);
            if (!imu_file) throw std::runtime_error("cannot open IMU CSV");
            imu_samples = rover::read_imu_csv(imu_file);
        }
        rover::ImuBatches imu_batches(imu_samples);
        size_t imu_samples_submitted = 0;
        size_t imu_frames_skipped = 0;
        std::int64_t replay_first_ns = -1;
        std::chrono::steady_clock::time_point replay_started;
        ORB_SLAM3::Map* final_tracked_map = nullptr;
        {
            const cv::FileStorage settings(argv[3], cv::FileStorage::READ);
            if (!settings["Tracking.StereoRelocalizationWaitSeconds"].empty()
                && (!orbslam3_has_stereo_recovery_wait || !orbslam3_has_stereo_recovery_wait()
                    || !orbslam3_has_initial_stereo_relocalization || !orbslam3_has_initial_stereo_relocalization()))
                throw std::runtime_error("Stereo recovery patch missing or outdated: apply orbslam3_stereo_recovery_wait.patch and rebuild native runtime");
        }
        if (!orbslam3_has_safe_presave_observations || !orbslam3_has_safe_presave_observations())
            throw std::runtime_error("Atlas save safety patch missing: apply orbslam3_presave_observation_safety.patch and rebuild native runtime");
        ORB_SLAM3::System slam(argv[1], argv[3], imu_requested
            ? ORB_SLAM3::System::IMU_STEREO : ORB_SLAM3::System::STEREO, false);
#else
        ORB_SLAM3::System slam(argv[1], argv[2], argv[3], ORB_SLAM3::System::STEREO, false);
#endif
        std::ifstream pair_file;
        std::istream* input = &std::cin;
        if (!live) { pair_file.open(argv[4]); input = &pair_file; }
        std::ofstream timing(std::string(argv[5]) + "/timings.csv");
        if (!*input || !timing) throw std::runtime_error("cannot open input/output");
        std::ofstream poses;
        if (live) {
            poses.open(std::string(argv[5]) + "/live_poses.csv");
            if (!poses) throw std::runtime_error("cannot open live poses");
            poses << "stamp_ns,state,map_id,process_ms";
            for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) poses << ",m" << i << j;
            poses << '\n' << std::setprecision(17); poses.flush();
        }
        timing << "stamp_ns,state,map_id,process_ms";
#ifdef ROVER_POINT_BASELINE
        timing << ",imu_initialized,imu_samples";
#endif
        timing << '\n' << std::setprecision(17);
        std::map<unsigned long, ORB_SLAM3::Map*> maps;
        std::string line;
        size_t processed = 0;
        while (std::getline(*input, line)) {
            std::istringstream row(line);
            std::string stamp, left, right;
            std::getline(row, stamp, '\t'); std::getline(row, left, '\t'); std::getline(row, right);
#ifdef ROVER_POINT_BASELINE
            std::int64_t frame_ns = 0;
            if (imu_requested) {
                size_t used = 0;
                frame_ns = std::stoll(stamp, &used);
                if (used != stamp.size() || frame_ns < 0) throw std::runtime_error("invalid stereo timestamp");
                if (replay_first_ns < 0) {
                    replay_first_ns = frame_ns;
                    replay_started = std::chrono::steady_clock::now();
                }
                // 비동기 LocalMapping에도 원본 촬영과 같은 처리 시간을 제공한다.
                std::this_thread::sleep_until(replay_started + std::chrono::nanoseconds(frame_ns - replay_first_ns));
            }
#endif
            cv::Mat image_left = cv::imread(left, cv::IMREAD_GRAYSCALE);
            cv::Mat image_right = cv::imread(right, cv::IMREAD_GRAYSCALE);
            if (image_left.empty() || image_right.empty()) throw std::runtime_error("missing stereo image");
            auto start = std::chrono::steady_clock::now();
#ifdef ROVER_POINT_BASELINE
            std::vector<ORB_SLAM3::IMU::Point> imu_batch;
            if (imu_requested) {
                const auto samples = imu_batches.next_for_orb(frame_ns);
                if (samples.empty()) {
                    ++imu_frames_skipped;
                    std::cerr << "Skipping frame " << stamp << ": insufficient effective IMU integration interval\n";
                    continue;
                }
                for (const auto& sample : samples)
                    imu_batch.emplace_back(sample.accel[0], sample.accel[1], sample.accel[2],
                        sample.gyro[0], sample.gyro[1], sample.gyro[2], static_cast<double>(sample.stamp_ns)*1e-9);
                imu_samples_submitted += imu_batch.size();
            }
            auto tracked_pose = slam.TrackStereo(image_left, image_right, std::stod(stamp)*1e-9, imu_batch);
#else
            slam.TrackStereo(image_left, image_right, std::stod(stamp)*1e-9);
#endif
            double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
            int state = slam.GetTrackingState();
            long long map_id = -1;
#ifdef ROVER_POINT_BASELINE
            bool imu_initialized = false;
            final_tracked_map = nullptr;
#endif
            if (state == 2) {
                for (auto* point : slam.GetTrackedMapPoints()) {
                    if (!point || point->isBad()) continue;
                    auto* map = point->GetMap();
                    if (!map || map->IsBad()) continue;
                    map_id = map->GetInitKFid(); maps[map_id] = map;
#ifdef ROVER_POINT_BASELINE
                    imu_initialized = map->isImuInitialized();
                    final_tracked_map = map;
#endif
                    break;
                }
            }
            timing << stamp << ',' << state << ',' << map_id << ',' << milliseconds;
#ifdef ROVER_POINT_BASELINE
            timing << ',' << (imu_initialized ? 1 : 0) << ',' << imu_batch.size();
#endif
            timing << '\n';
            timing.flush();
#ifdef ROVER_POINT_BASELINE
            if (live) {
                const Eigen::Matrix4f world_camera = tracked_pose.inverse().matrix();
                poses << stamp << ',' << state << ',' << map_id << ',' << milliseconds;
                for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) poses << ',' << world_camera(i,j);
                poses << '\n'; poses.flush();
                if (!poses) throw std::runtime_error("live pose write failed");
            }
#endif
            ++processed;
        }
        slam.Shutdown();
        size_t exported = 0;
#ifdef ROVER_POINT_BASELINE
        std::vector<unsigned long> exported_imu_initialized_maps;
#endif
#ifdef ROVER_POINT_BASELINE
        for (auto* map : rover::unique_map_pointers(maps)) {
#else
        for (const auto& item : maps) {
            auto* map = item.second;
#endif
            if (map->IsBad()) continue;
            auto id = map->GetInitKFid();
            std::ofstream frames(std::string(argv[5]) + "/map_" + std::to_string(id) + "_keyframes.csv");
            frames << "map_id,keyframe_id,source_stamp,x,y,z,qx,qy,qz,qw\n" << std::setprecision(17);
#ifdef ROVER_POINT_BASELINE
            const auto previously_exported = exported;
#endif
            for (auto* frame : map->GetAllKeyFrames()) {
                if (!frame || frame->isBad() || frame->GetMap() != map) continue;
                Eigen::Matrix3d rotation;
#ifdef ROVER_POINT_BASELINE
                auto se3 = frame->GetPoseInverse();
                cv::Mat pose(4, 4, CV_32F);
                for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) pose.at<float>(i,j) = se3.matrix()(i,j);
#else
                cv::Mat pose = frame->GetPoseInverse();
#endif
                for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) rotation(i,j) = pose.at<float>(i,j);
                Eigen::Quaterniond q(rotation); q.normalize();
                frames << id << ',' << frame->mnId << ',' << frame->mTimeStamp;
                for (int i=0; i<3; ++i) frames << ',' << pose.at<float>(i,3);
                frames << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << '\n';
                ++exported;
            }
            if (!frames) throw std::runtime_error("keyframe export failed");
#ifdef ROVER_POINT_BASELINE
            if (exported > previously_exported && map->isImuInitialized()) exported_imu_initialized_maps.push_back(id);
#endif
        }
        if (!timing || processed == 0 || exported == 0) throw std::runtime_error("no valid final keyframes");
        std::ofstream complete(std::string(argv[5]) + "/runner_complete.json");
        complete << "{\"status\":\"complete\",\"frames\":" << processed
                 << ",\"keyframes\":" << exported;
#ifdef ROVER_POINT_BASELINE
        const bool imu_initialized = !exported_imu_initialized_maps.empty();
        complete << ",\"imu_requested\":" << (imu_requested ? "true" : "false")
                 << ",\"imu_initialized\":" << (imu_initialized ? "true" : "false")
                 << ",\"imu_fused\":" << (imu_requested && imu_initialized ? "true" : "false")
                 << ",\"imu_samples_submitted\":" << imu_samples_submitted
                 << ",\"imu_frames_skipped\":" << imu_frames_skipped
                 << ",\"imu_replay_paced\":" << (imu_requested ? "true" : "false")
                 << ",\"active_map_imu_initialized\":null"
                 << ",\"final_tracked_map_initialized\":";
        if (final_tracked_map && !final_tracked_map->IsBad())
            complete << (final_tracked_map->isImuInitialized() ? "true" : "false");
        else complete << "null";
        complete << ",\"exported_imu_initialized_maps\":[";
        for (size_t i = 0; i < exported_imu_initialized_maps.size(); ++i) {
            if (i) complete << ',';
            complete << exported_imu_initialized_maps[i];
        }
        complete << ']';
#endif
        complete << "}\n";
        return complete ? 0 : 1;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

// ================================================================
// stream_orb.cpp
// 설명: 시뮬이 파일로 건넨 좌우 영상 요청을 ORB-SLAM3 stereo로 순서대로 처리하고 camera pose를 파일로 돌려준다.
// 사용법: stream_orb vocabulary.txt orb.yaml io_dir   (io_dir에 STOP 파일이 생기면 종료)
//   요청: io_dir/<stamp_ns>.req = "left.png\nright.png\n"    응답: io_dir/<stamp_ns>.pose = "state map_id keypoints tx ty tz qx qy qz qw"
// ================================================================
#include <System.h>
#include <MapPoint.h>
#include <Map.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: stream_orb vocabulary.txt orb.yaml io_dir\n";
        return 2;
    }
    try {
        const fs::path io(argv[3]);
        cv::setNumThreads(1);
        cv::setRNGSeed(0);
        ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::STEREO, false);
        std::set<long long> done;
        std::ofstream(io / "READY") << "ready\n";
        for (;;) {
            if (fs::exists(io / "STOP")) break;
            std::vector<long long> pending;
            for (const auto& entry : fs::directory_iterator(io)) {
                if (entry.path().extension() != ".req") continue;
                const long long stamp = std::stoll(entry.path().stem().string());
                if (!done.count(stamp)) pending.push_back(stamp);
            }
            if (pending.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            std::sort(pending.begin(), pending.end());
            const long long stamp = pending.front();
            std::ifstream request(io / (std::to_string(stamp) + ".req"));
            std::string left_path, right_path;
            std::getline(request, left_path);
            std::getline(request, right_path);
            if (left_path.empty() || right_path.empty()) {  // 아직 다 쓰이지 않은 요청
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            const cv::Mat left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
            const cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
            if (left.empty() || right.empty()) throw std::runtime_error("missing stereo image for " + std::to_string(stamp));
            const auto camera_from_world = slam.TrackStereo(left, right, stamp * 1e-9);
            const int state = slam.GetTrackingState();
            long long map_id = -1;
            for (auto* point : slam.GetTrackedMapPoints())
                if (point && !point->isBad()) map_id = static_cast<long long>(point->GetMap()->GetId());
            const auto pose = camera_from_world.inverse();
            const auto p = pose.translation();
            const auto q = pose.unit_quaternion();
            const fs::path tmp = io / (std::to_string(stamp) + ".pose.tmp");
            {
                std::ofstream out(tmp);
                out << std::setprecision(17) << state << ' ' << map_id << ' ' << slam.GetTrackedKeyPointsUn().size() << ' '
                    << p.x() << ' ' << p.y() << ' ' << p.z() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
            }
            fs::rename(tmp, io / (std::to_string(stamp) + ".pose"));
            done.insert(stamp);
        }
        slam.Shutdown();
        std::ofstream(io / "FINISHED") << "finished\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "STREAM_ORB_FAILED " << error.what() << std::endl;
        std::ofstream(fs::path(argv[3]) / "FAILED") << error.what() << '\n';
        return 1;
    }
}

// ================================================================
// compat.h
// 설명: ORB-LINE upstream의 OpenCV3 상수를 OpenCV4와 연결한다.
// 사용법: CMake에서 -include compat.h로 적용한다.
// ================================================================
#include <unistd.h>
#ifdef __cplusplus
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs.hpp>
#ifndef CV_LOAD_IMAGE_UNCHANGED
#define CV_LOAD_IMAGE_UNCHANGED cv::IMREAD_UNCHANGED
#endif
#endif

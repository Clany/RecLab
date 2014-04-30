#ifndef AUTO_CALIB_H
#define AUTO_CALIB_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "clany/clany_macros.h"

_CLANY_BEGIN

cv::Mat autoCalibrate(const std::vector<cv::Mat>& cam_mat_vec);

_CLANY_END

#endif // AUTO_CALIB_H
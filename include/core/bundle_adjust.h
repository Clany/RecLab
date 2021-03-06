#ifndef BUNDLE_ADJUST_H
#define BUNDLE_ADJUST_H

#include <opencv2/opencv.hpp>
#include "clany/point_types.h"


_CLANY_BEGIN

using PtCloud = PointCloud<Point3d>;
using ImgPtCorrs = vector<vector<int>>;

void adjustBundle(PtCloud& cloud, ImgPtCorrs& img_pts, cv::Matx33d& K, vector<cv::Matx34d>& P_vec,
                  const vector<vector<cv::KeyPoint>>& key_pts_vec);


_CLANY_END

#endif // BUNDLE_ADJUST_H
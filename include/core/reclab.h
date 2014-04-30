#ifndef RECLAB_H
#define RECLAB_H

#include <deque>
#include <functional>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "clany/point_types.h"

#include "feature_matcher.h"
#include "core/auto_calib.h"


_CLANY_BEGIN
class RecLab
{
public:
    RecLab(cv::InputArray calib_mat = cv::noArray());

    void operator()(const cv::Mat& frame);

    void writeResult() const;

    bool isSuccess() const { return point_cloud.size() != 0; }

private:
    using PtCloud = PointCloud<pair<PointXYZ, vector<int>>>;

    // Calibrated case
    bool initModel(const cv::Mat& frame);
    bool addView(const cv::Mat& frame);

    bool findCameraMatrices(const vector<cv::Point2d>& l_pts,
                            const vector<cv::Point2d>& r_pts,
                            cv::Matx34d& P0, cv::Matx34d& P1);

    bool decompEssentialMat(cv::InputArray E, cv::Matx33d& R1, cv::Matx33d& R2,
                            cv::Matx31d& t1, cv::Matx31d& t2) const;

    cv::Matx34d getCameraMat(cv::InputArray R, cv::InputArray t) const;
    void getRTFromCamMat(cv::InputArray P, cv::Vec3d& rvec, cv::Vec3d& tvec) const;

    bool triangulate(const vector<cv::Point2d>& l_pts,
                     const vector<cv::Point2d>& r_pts,
                     cv::Matx34d& P0, cv::Matx34d& P1,
                     PointCloud<PointXYZ>& point_cloud);

    bool testTriangulate(const vector<cv::Point2d>& l_pts,
                         const vector<cv::Point2d>& r_pts,
                         cv::Matx34d& P0, cv::Matx34d& P1,
                         PointCloud<PointXYZ>& point_cloud);

    bool isInFrontOfCam(const PointCloud<PointXYZ>& point_cloud, const cv::Matx34d& P) const;

    // Uncalibrated case
    bool initModelUC(const cv::Mat& frame);
    bool addViewUC(const cv::Mat& frame);

    // Common functions
    void getFeaturePoints();
    void getCorrespondences(vector<cv::Point2d>& l_pts, vector<cv::Point2d>& r_pts);
    void matToPointCloud(const cv::InputArray mat_cloud, PtCloud& point_cloud) const;
    cv::Matx41d triangulationImpl(const cv::Point2d& l_pt, const cv::Point2d& r_pt,
                                  const cv::Matx34d& P0, const cv::Matx34d& P1) const;

private:
    function<bool(const cv::Mat& frame)> processor;
    shared_ptr<FeatureMatcher> feature_matcher;

    cv::Matx33d K;
    vector<cv::Matx34d> cam_mat_vec;

    int frame_count;
    deque<cv::Mat> frame_seq;
    cv::Mat prev_frame;
    cv::Mat curr_frame;

    vector<cv::KeyPoint> curr_key_pts;
    vector<cv::KeyPoint> prev_key_pts;

    vector<vector<cv::KeyPoint>> key_pts_vec;
    vector<vector<cv::DMatch>> matches_vec;

    PtCloud point_cloud;
    PointCloud<pair<cv::Point3f, int>> prev_3dto2d_corresp;
};


class ConvertExcept : public runtime_error
{
public:
    ConvertExcept(const string& err_msg) : runtime_error(err_msg) {};
};
_CLANY_END

#endif // RECLAB_H
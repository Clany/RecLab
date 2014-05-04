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
    enum class MatchMethod { RichFeature, OpticalFlow };

    RecLab(cv::InputArray calib_mat = cv::noArray());

    RecLab(int rows, int cols);

    void setMatchMethod(MatchMethod method);

    void operator()(const cv::Mat& frame);

    void postProcess();

    void writeResult() const;

    bool isSuccess() const { return point_cloud.size() != 0; }

private:
    using PtCloud = PointCloud<Point3d>;
    using ImgPtCorrs = vector<vector<int>>;

    void getFeaturePoints();
    bool initModel(const cv::Mat& frame);
    bool addView(const cv::Mat& frame);
    void denseReconstruct();

    void getCorrespondences(vector<cv::Point2d>& l_pts, vector<cv::Point2d>& r_pts);

    bool findCameraMatrices(const vector<cv::Point2d>& l_pts,
                            const vector<cv::Point2d>& r_pts,
                            cv::Matx34d& P0, cv::Matx34d& P1);

    bool decompEssentialMat(cv::InputArray E, cv::Matx33d& R1, cv::Matx33d& R2,
                            cv::Matx31d& t1, cv::Matx31d& t2) const;

    cv::Matx34d getCameraMat(cv::InputArray R, cv::InputArray t) const;

    void getOrientation(cv::InputArray P, cv::Vec3d& rvec, cv::Vec3d& tvec) const;

    bool testTriangulate(const vector<cv::Point2d>& l_pts,
                         const vector<cv::Point2d>& r_pts,
                         cv::Matx34d& P0, cv::Matx34d& P1,
                         PtCloud& point_cloud);

    bool triangulate(const vector<cv::Point2d>& l_pts,
                     const vector<cv::Point2d>& r_pts,
                     cv::Matx34d& P0, cv::Matx34d& P1,
                     PtCloud& point_cloud) const;

    cv::Matx41d triangulationImpl(const cv::Point2d& l_pt, const cv::Point2d& r_pt,
                                  const cv::Matx34d& P0, const cv::Matx34d& P1) const;

    bool isInFrontOfCam(const PtCloud& point_cloud, const cv::Matx34d& P) const;

    double pointDepth(const Point3d& point, const cv::Matx34d& P) const;

    void denseMatch(const cv::Mat& left_img, const cv::Mat& right_img,
                    const cv::Matx33d& H1, const cv::Matx33d H2,
                    vector<cv::Point2d>& l_pts, vector<cv::Point2d>& r_pts);

    void writePointCloud(const string& file_name, const PtCloud& cloud) const;

private:
    function<bool(const cv::Mat& frame)> processor;
    shared_ptr<FeatureMatcher> OF_matcher;
    shared_ptr<FeatureMatcher> RF_matcher;
    shared_ptr<FeatureMatcher> feature_matcher;

    cv::Matx33d K;
    vector<cv::Matx34d> cam_mat_vec;

    int frame_count;
    vector<cv::Mat> frame_vec;
    cv::Mat prev_frame;
    cv::Mat curr_frame;

    vector<cv::KeyPoint> prev_key_pts;
    vector<cv::KeyPoint> curr_key_pts;

    vector<vector<cv::KeyPoint>> key_pts_vec;
    vector<vector<cv::DMatch>> matches_vec;

    PtCloud point_cloud;
    PtCloud dense_cloud;
    // Every reconstructed 3D point corresponds to a set of image points
    ImgPtCorrs corr_imgpts;
};
_CLANY_END

#endif // RECLAB_H
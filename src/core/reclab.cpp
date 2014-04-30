#pragma warning(disable: 4244)

#include <numeric>
#include <opencv2/viz/vizcore.hpp>
#include "clany/file_operation.hpp"

#include "core/reclab.h"
#include "core/bundle_adjust.h"

using namespace std;
using namespace cv;
using namespace clany;
using namespace placeholders;


const float EPSILON = 0.0001f;

RecLab::RecLab(InputArray _calib_mat) :
    feature_matcher(MatcherFactory::create("RichFeature", "PyramidFAST"))
    /*feature_matcher(MatcherFactory::create("OpticalFlow", "GFTT"))*/, frame_count(0)
{
    if (!_calib_mat.empty()) {
        K = _calib_mat.getMat();
        processor = bind(&RecLab::initModel, this, _1);
    } else {
        processor = bind(&RecLab::initModelUC, this, _1);
    }
}


void RecLab::operator()(const Mat& frame)
{
    curr_frame = frame;
// #ifndef NDEBUG
//     curr_key_pts.clear();
//     FileStorage ifs("features" + to_string(frame_count) + ".xml", FileStorage::READ);
//     read(ifs["KeyPoints"], curr_key_pts);
// #else
//     getFeaturePoints();
// #endif
    getFeaturePoints();

//     FileStorage ofs("features" + to_string(frame_count) + ".xml", FileStorage::WRITE);
//     write(ofs, "KeyPoints", curr_key_pts);
//     ++frame_count;
//     return;

    if (processor(frame)) {
        frame_seq.push_back(frame);
    }
    prev_key_pts = move(curr_key_pts);
    prev_frame = frame;
    ++frame_count;
}


void RecLab::writeResult() const
{
    Mat result(1, point_cloud.size(), CV_32FC3);
    float* data = result.ptr<float>(0);
    for (const auto& point : point_cloud) {
        data[0] = point.first.x;
        data[1] = point.first.y;
        data[2] = point.first.z;
        data += 3;
    }
    viz::writeCloud("result.obj", result);
}


//////////////////////////////////////////////////////////////////////////
// Calibrated case
bool RecLab::initModel(const Mat& frame)
{
    // Skip first frame
    if (!frame_count) {
        key_pts_vec.push_back(curr_key_pts);
        return true;
    }

    // Perform initial reconstruction from two views
    vector<Point2d> prev_fr_pts, curr_fr_pts;
    getCorrespondences(prev_fr_pts, curr_fr_pts);

    Matx34d p0(Matx34d::eye());
    Matx34d p1(Matx34d::eye());
    if (findCameraMatrices(prev_fr_pts, curr_fr_pts, p0, p1)) {
        key_pts_vec.push_back(curr_key_pts);
        cam_mat_vec.push_back(p0);
        cam_mat_vec.push_back(p1);
        adjustBundle(point_cloud, K, cam_mat_vec, key_pts_vec);
        processor = bind(&RecLab::addView, this, _1);
        return true;
    }

    return false;
}


bool RecLab::addView(const Mat& frame)
{
    vector<DMatch> matches;
    vector<Point2d> l_pts, r_pts;
    feature_matcher->match(curr_frame, prev_frame, curr_key_pts, prev_key_pts, matches);

    // Get 3D-2D correspondences
    vector<Point3f> obj_pts;
    vector<Point2f> img_pts;
    vector<DMatch> non_corresp;
    for (const auto& m : matches) {
//         l_pts.push_back(prev_key_pts[m.trainIdx].pt);
//         r_pts.push_back(curr_key_pts[m.queryIdx].pt);

        auto pt_iter = find_if(point_cloud.begin(), point_cloud.end(),
                               [&m, this](const auto& point) {
            return point.second.back() == m.trainIdx && point.second.size() == frame_count;
        });
        if (pt_iter != point_cloud.end()) {
            obj_pts.emplace_back(pt_iter->first.x, pt_iter->first.y, pt_iter->first.z);
            img_pts.push_back(curr_key_pts[m.queryIdx].pt);
            pt_iter->second.push_back(m.queryIdx);
        } else {
            l_pts.push_back(prev_key_pts[m.trainIdx].pt);
            r_pts.push_back(curr_key_pts[m.queryIdx].pt);
            non_corresp.push_back(m);
        }
    }
    for (auto& point : point_cloud) {
        if (point.second.size() == frame_count) {
            point.second.push_back(-1);
        }
    }
    matches_vec.push_back(move(matches));

    // last estimated camera matrix
    Matx34d prev_P = cam_mat_vec.back();
    Vec3d rvec, tvec;
    getRTFromCamMat(prev_P, rvec, tvec);

    // Estimate current camera matrix
    vector<int> inliers;
    double min_val, max_val;
    minMaxIdx(img_pts, &min_val, &max_val);
    solvePnPRansac(obj_pts, img_pts, K, noArray(), rvec, tvec, true, 1000,
                   0.006 * max_val, 0.25 * img_pts.size(), inliers, CV_EPNP);
    // Not enough inliers
    if (inliers.size() * 5 < img_pts.size()) return false;

    Matx33d R;
    Rodrigues(rvec, R);
    Matx34d curr_P = getCameraMat(R, tvec);
    cam_mat_vec.push_back(curr_P);

    PointCloud<PointXYZ> cloud;
    triangulate(l_pts, r_pts, K*prev_P, K*curr_P, cloud);

    for (size_t i = 0; i < cloud.size(); ++i) {
        PointXYZ X = cloud[i];
        vector<int> img_pts(frame_count, -1);
        img_pts.back() = non_corresp[i].trainIdx;
        img_pts.push_back(non_corresp[i].queryIdx);
        point_cloud.push_back({X, img_pts});
    }
    key_pts_vec.push_back(move(curr_key_pts));
    adjustBundle(point_cloud, K, cam_mat_vec, key_pts_vec);

    return true;
}


bool RecLab::findCameraMatrices(const vector<Point2d>& l_pts,
                                const vector<Point2d>& r_pts,
                                Matx34d& P0, Matx34d& P1)
{
    // Too few correspondences
    const size_t pt_size = l_pts.size();
    if (pt_size < 100) return false;


    double max_val;
    cv::minMaxIdx(l_pts, nullptr, &max_val);
    Matx33d F = findFundamentalMat(l_pts, r_pts, FM_RANSAC, 0.006 * max_val);
    Matx33d E = K.t() * F * K;
    Matx33d H1, H2;
    stereoRectifyUncalibrated(l_pts, r_pts, F, curr_frame.size(), H1, H2);
    Mat rec_curr, rec_prev;
    warpPerspective(prev_frame, rec_prev, H1, curr_frame.size());
    warpPerspective(curr_frame, rec_curr, H2, curr_frame.size());
    imwrite("rec_03.jpg", rec_prev);
    imwrite("rec_05.jpg", rec_curr);
    system("Pause");

    // Determinant of E should be 0
    if (abs(determinant(E)) > 1e-6) return false;

    // Four possible solutions
    Matx33d R1, R2;
    Matx31d t1, t2;
    if (!decompEssentialMat(E, R1, R2, t1, t2)) return false;

    // det(R) should be +-1
    double det_R = determinant(R1);
    if (abs(abs(det_R) - 1.0) > 1e-6) return false;
    // If det(R1) = -1, reverse sign of E to make det(R) = 1
    if (det_R < 0) decompEssentialMat(-E, R1, R2, t1, t2);

    // Normalize image points first
    vector<Point2d> l_pts_norm, r_pts_norm;
    undistortPoints(l_pts, l_pts_norm, K, noArray());
    undistortPoints(r_pts, r_pts_norm, K, noArray());

    // Test four cases
    PointCloud<PointXYZ> pt_cloud;
    P0 = Matx34d::eye();

    P1 = getCameraMat(R1, t1);
    if (testTriangulate(l_pts_norm, r_pts_norm, P0, P1, pt_cloud)) return true;

    P1 = getCameraMat(R1, t2);
    if (testTriangulate(l_pts_norm, r_pts_norm, P0, P1, pt_cloud)) return true;

    P1 = getCameraMat(R2, t1);
    if (testTriangulate(l_pts_norm, r_pts_norm, P0, P1, pt_cloud)) return true;

    P1 = getCameraMat(R2, t2);
    if (testTriangulate(l_pts_norm, r_pts_norm, P0, P1, pt_cloud)) return true;

    return false;
}


bool RecLab::decompEssentialMat(InputArray E, Matx33d& R1, Matx33d& R2,
                                Matx31d& t1, Matx31d& t2) const
{
    Vec3d D;
    Matx33d U, Vt;
    SVD::compute(E, D, U, Vt);

    //First and second singular values sbould be the same
    double ratio = D[0] < D[1] ? abs(D[0] / D[1]) : abs(D[1] / D[0]);
    if (ratio < 0.7) return false;

    const Matx33d W(0, -1, 0,
                    1,  0, 0,
                    0,  0, 1);
    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;
    t1 = U.col(2);
    t2 = -t1;

    return true;
}


Matx34d RecLab::getCameraMat(InputArray _R, InputArray _t) const
{
    Matx33d R = _R.getMat();
    Matx31d t = _t.getMat();

    return Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0),
                   R(1, 0), R(1, 1), R(1, 2), t(1),
                   R(2, 0), R(2, 1), R(2, 2), t(2));
}


void RecLab::getRTFromCamMat(cv::InputArray _P, cv::Vec3d& rvec, cv::Vec3d& tvec) const
{
    Mat P = _P.getMat();
    Mat R = P(Rect(0, 0, 3, 3));
    Rodrigues(R, rvec);
    tvec = P.col(3);
}


bool RecLab::triangulate(const vector<Point2d>& l_pts,
                         const vector<Point2d>& r_pts,
                         Matx34d& P0, Matx34d& P1,
                         PointCloud<PointXYZ>& pt_cloud)
{
    size_t pc_size = l_pts.size();
    pt_cloud.resize(pc_size);
#if 0
    Mat pt_cloud_homo, pt_cloud_inhomo;
    triangulatePoints(P0, P1, l_pts, r_pts, pt_cloud_homo);
    convertPointsFromHomogeneous(pt_cloud_homo.reshape(4, 1), pt_cloud_inhomo);

    // Calculate reprojection error
    Matx31d r_vec;
    Rodrigues(Mat(P0)(Rect(0, 0, 3, 3)), r_vec);
    Matx31d t_vec = P0.col(3);
    vector<Point2f> reproj_img_pts;
    projectPoints(pt_cloud_inhomo, r_vec, t_vec, Matx33d::eye(), noArray(), reproj_img_pts);
    double mse = inner_product(l_pts.begin(), l_pts.end(), reproj_img_pts.begin(),
                               0.0, plus<double>(), [](const Point2f& pt_a, const Point2f& pt_b) {
        return norm(pt_a - pt_b);
    }) / reproj_img_pts.size();
    matToPointCloud(pt_cloud_inhomo, pt_cloud);
#else
    double mse = 0;
    for (size_t i = 0; i < pc_size; i++) {
        Matx41d X = triangulationImpl(l_pts[i], r_pts[i], P0, P1);
        {
            Matx31d img_pt_homo = P0 * X;
            Point2d img_pt(img_pt_homo(0) / img_pt_homo(2),
                           img_pt_homo(1) / img_pt_homo(2));
            mse += norm(img_pt - l_pts[i]);
        }
        {
            Matx31d img_pt_homo = P1 * X;
            Point2d img_pt(img_pt_homo(0) / img_pt_homo(2),
                           img_pt_homo(1) / img_pt_homo(2));
            mse += norm(img_pt - r_pts[i]);
        }
        pt_cloud[i] = PointXYZ(X(0), X(1), X(2));
    }
    mse /= pc_size * 2;
#endif
    return mse < 100 && isInFrontOfCam(pt_cloud, P0) && isInFrontOfCam(pt_cloud, P1);
}


bool RecLab::testTriangulate(const vector<Point2d>& l_pts,
                             const vector<Point2d>& r_pts,
                             Matx34d& P0, Matx34d& P1,
                             PointCloud<PointXYZ>& pt_cloud)
{
    if (triangulate(l_pts, r_pts, P0, P1, pt_cloud)) {
        auto& curr_match = matches_vec.back();
        for (size_t i = 0; i < curr_match.size(); ++i) {
            PointXYZ X = pt_cloud[i];
            int l_imgpt_idx = curr_match[i].trainIdx;
            int r_imgpt_idx = curr_match[i].queryIdx;
            point_cloud.push_back({X, {l_imgpt_idx, r_imgpt_idx}});
        }
        return true;
    }

    return false;
}


bool RecLab::isInFrontOfCam(const PointCloud<PointXYZ>& point_cloud, const Matx34d& P)  const
{
//     Matx33d M = Mat(P)(Rect(0, 0, 3, 3));
//     double w = (P.row(2) * Matx41d(pt.x, pt.y, pt.z, 1))(0);
//     double depth = w / norm(M.row(2));
//     if (determinant(M) < 0) depth = -depth;

//     const PointXYZ& pt = point_cloud[0];
//     double w = (P.row(2) * Matx41d(pt.x, pt.y, pt.z, 1))(0);
//     double det = determinant(Mat(P)(Rect(0, 0, 3, 3)));
//     return w * det > 0;

    Matx14d p3 = P.row(2);
    double det_R = determinant(Mat(P)(Rect(0, 0, 3, 3)));
    double num_ifc = count_if(point_cloud.begin(), point_cloud.end(), [&p3, &det_R](const auto& pt) {
        double w = p3.dot(Matx14d(pt.x, pt.y, pt.z, 1));
        return w * det_R > 0;
    });

    return num_ifc / point_cloud.size() > 0.75;
}


//////////////////////////////////////////////////////////////////////////
// Uncalibrated case
bool RecLab::initModelUC(const Mat& frame)
{

    processor = bind(&RecLab::addViewUC, this, _1);

    return true;
}


bool RecLab::addViewUC(const Mat& frame)
{
    return true;
}


//////////////////////////////////////////////////////////////////////////
// Common functions
void RecLab::getFeaturePoints()
{
    feature_matcher->detect(curr_frame, curr_key_pts);
}


void RecLab::getCorrespondences(vector<Point2d>& l_pts, vector<Point2d>& r_pts)
{
    vector<DMatch> matches;
    feature_matcher->match(curr_frame, prev_frame, curr_key_pts, prev_key_pts, matches);

    if (frame_seq.back().data == prev_frame.data) {
        for (const auto& m : matches) {
            r_pts.push_back(curr_key_pts[m.queryIdx].pt);
            l_pts.push_back(prev_key_pts[m.trainIdx].pt);
        }
        matches_vec.push_back(move(matches));
    } else {
        auto& last_matchs = matches_vec.back();
        auto& l_key_pts = key_pts_vec.back();
        vector<DMatch> lr_matches;
        for (const auto& m : matches) {
            auto iter = find_if(last_matchs.begin(), last_matchs.end(), [&m](const DMatch& lm) {
                return lm.queryIdx == m.trainIdx;
            });
            if (iter != last_matchs.end()) {
                r_pts.push_back(curr_key_pts[m.queryIdx].pt);
                l_pts.push_back(l_key_pts[iter->trainIdx].pt);
                lr_matches.push_back(DMatch(m.queryIdx, iter->trainIdx, -1));
            }
        }
        matches_vec.pop_back();
        matches_vec.push_back(move(lr_matches));
    }

#ifndef NDEBUG
    Mat img_matches;
    drawMatches(curr_frame, curr_key_pts, frame_seq.back(), key_pts_vec.back(), matches_vec.back(), img_matches);
#endif
}


void RecLab::matToPointCloud(const InputArray _mat_cloud, PtCloud& point_cloud) const
{
    Mat mat_cloud = _mat_cloud.getMat();
    int pt_size = mat_cloud.checkVector(3);
    if (mat_cloud.depth() != CV_32F || pt_size < 0) {
        throw ConvertExcept("Fail to convert from cv::Mat to PointCloud");
    }

//     if (point_cloud.size() != pt_size) point_cloud.resize(pt_size);
//     memmove(point_cloud.points.data(), mat_cloud.data, pt_size * 3 * sizeof(float));
}


Matx41d RecLab::triangulationImpl(const Point2d& l_pt, const Point2d& r_pt,
                                  const Matx34d& P0, const Matx34d& P1) const
{
    double w1 = 1, w2 = 1;
    Matx31d X;
    for (int i = 0; i < 10; ++i) {
        Matx43d A(l_pt.x*P0(2, 0) - P0(0, 0), l_pt.x*P0(2, 1) - P0(0, 1), l_pt.x*P0(2, 2) - P0(0, 2),
                  l_pt.y*P0(2, 0) - P0(1, 0), l_pt.y*P0(2, 1) - P0(1, 1), l_pt.y*P0(2, 2) - P0(1, 2),
                  r_pt.x*P1(2, 0) - P1(0, 0), r_pt.x*P1(2, 1) - P1(0, 1), r_pt.x*P1(2, 2) - P1(0, 2),
                  r_pt.y*P1(2, 0) - P1(1, 0), r_pt.y*P1(2, 1) - P1(1, 1), r_pt.y*P1(2, 2) - P1(1, 2));
        Matx41d B(P0(0, 3) - l_pt.x*P0(2, 3),
                  P0(1, 3) - l_pt.y*P0(2, 3),
                  P1(0, 3) - r_pt.x*P1(2, 3),
                  P1(1, 3) - r_pt.y*P1(2, 3));
        solve(A, B, X, DECOMP_SVD);

        //recalculate weights
        Matx14d X_homo(X(0), X(1), X(2), 1.f);
        float w1_ = P0.row(2).dot(X_homo);
        float w2_ = P1.row(2).dot(X_homo);

        //breaking point
        if (abs(w1 - w1_) <= EPSILON && abs(w2 - w2_) <= EPSILON) break;

        w1 = w1_;
        w2 = w2_;

        //reweight equations and solve
        Matx43d A_((l_pt.x*P0(2, 0) - P0(0, 0)) / w1, (l_pt.x*P0(2, 1) - P0(0, 1)) / w1, (l_pt.x*P0(2, 2) - P0(0, 2)) / w1,
                   (l_pt.y*P0(2, 0) - P0(1, 0)) / w1, (l_pt.y*P0(2, 1) - P0(1, 1)) / w1, (l_pt.y*P0(2, 2) - P0(1, 2)) / w1,
                   (r_pt.x*P1(2, 0) - P1(0, 0)) / w2, (r_pt.x*P1(2, 1) - P1(0, 1)) / w2, (r_pt.x*P1(2, 2) - P1(0, 2)) / w2,
                   (r_pt.y*P1(2, 0) - P1(1, 0)) / w2, (r_pt.y*P1(2, 1) - P1(1, 1)) / w2, (r_pt.y*P1(2, 2) - P1(1, 2)) / w2);
        Matx41d B_((P0(0, 3) - l_pt.x*P0(2, 3)) / w1,
                   (P0(1, 3) - l_pt.y*P0(2, 3)) / w1,
                   (P1(0, 3) - r_pt.x*P1(2, 3)) / w2,
                   (P1(1, 3) - r_pt.y*P1(2, 3)) / w2);
        solve(A_, B_, X, DECOMP_SVD);
    }
    return Matx41d(X(0), X(1), X(2), 1.f);
}
#pragma warning(disable: 4244)

#include <numeric>
#include "clany/file_operation.hpp"
#include "clany/timer.hpp"

#include "core/reclab.h"
#include "core/bundle_adjust.h"

using namespace std;
using namespace cv;
using namespace clany;
using namespace placeholders;


const double EPSILON = 1e-7;

RecLab::RecLab(InputArray calib_mat)
    : OF_matcher(MatcherFactory::create("OpticalFlow", "PyramidGFTT")),
      RF_matcher(MatcherFactory::create("RichFeature", "FAST")),
      feature_matcher(RF_matcher), frame_count(0)
{
    K = calib_mat.getMat();
    processor = bind(&RecLab::initModel, this, _1);
}


RecLab::RecLab(int rows, int cols)
    : OF_matcher(MatcherFactory::create("OpticalFlow", "PyramidGFTT")),
      RF_matcher(MatcherFactory::create("RichFeature", "FAST")),
      feature_matcher(RF_matcher), frame_count(0)
{
    int focal = max(rows, cols);
    K = Matx33f(focal, 0,     cols / 2.f,
                0,     focal, rows / 2.f,
                0,     0,     1);
    processor = bind(&RecLab::initModel, this, _1);
}


void RecLab::setMatchMethod(MatchMethod method)
{
    switch (method) {
    case MatchMethod::RichFeature:
        feature_matcher = RF_matcher;
        break;
    case MatchMethod::OpticalFlow:
        feature_matcher = OF_matcher;
        break;
    default:
        feature_matcher = RF_matcher;
        break;
    }
}


void RecLab::operator()(const Mat& frame)
{
    curr_frame = frame;
    getFeaturePoints();
// #ifndef NDEBUG
//     curr_key_pts.clear();
//     FileStorage ifs("features" + to_string(frame_count) + ".xml", FileStorage::READ);
//     read(ifs["KeyPoints"], curr_key_pts);
// #else
//     getFeaturePoints();
// #endif
    cout << curr_key_pts.size() << " key points found in current frame" << endl;

//     FileStorage ofs("features" + to_string(frame_count) + ".xml", FileStorage::WRITE);
//     write(ofs, "KeyPoints", curr_key_pts);
//     ++frame_count;
//     return;

    if (processor(frame)) {
        frame_vec.push_back(frame);
        ++frame_count;
    }

    prev_key_pts = move(curr_key_pts);
    prev_frame = frame;
}


void RecLab::postProcess()
{
    /*denseReconstruct();*/

    // Prune points that are behind any camera
    vector<Point3d> pruned_cloud;
    for (const auto& point : point_cloud) {
        bool valid_pt = all_of(cam_mat_vec.begin(), cam_mat_vec.end(), [this, &point](const Matx34d& P) {
            return pointDepth(point, P) > 0;
        });
        if (valid_pt) pruned_cloud.emplace_back(point.x, point.y, point.z);
    }

    // Prune points based on depth
    vector<double> depth_vec;
    Matx34d mid_cam = cam_mat_vec[cam_mat_vec.size()/2];
    for (const auto& point : pruned_cloud) {
        depth_vec.push_back(pointDepth(point, mid_cam));
    }

    for (int i = 0; i < 10; ++i) {
        vector<double> tmp_depth;
        vector<Point3d> tmp_cloud;
        Scalar center, std_dev;
        meanStdDev(depth_vec, center, std_dev);
        double max_dist = center[0] + 3 * std_dev[0];
        double min_dist = center[0] - 5 * std_dev[0];
        for (const auto& point : pruned_cloud) {
            double depth = pointDepth(point, mid_cam);
            if (max_dist > depth && depth > min_dist) {
                tmp_depth.push_back(depth);
                tmp_cloud.push_back(point);
            }
        }
        swap(tmp_depth, depth_vec);
        swap(tmp_cloud, pruned_cloud);
    }
    point_cloud.resize(pruned_cloud.size());
    memcpy(point_cloud.points.data(), pruned_cloud.data(), pruned_cloud.size()*sizeof(double)*3);
}


void RecLab::writeResult() const
{
    cout << "Point cloud size: " << point_cloud.size() << endl;
    writePointCloud("Result.obj", point_cloud);
}


//////////////////////////////////////////////////////////////////////////
// Implementation
void RecLab::getFeaturePoints()
{
    feature_matcher->detect(curr_frame, curr_key_pts);
}


bool RecLab::initModel(const Mat& frame)
{
    // Skip first frame
    if (!frame_count) {
        key_pts_vec.push_back(curr_key_pts);
        return true;
    }

    // Perform initial reconstruction from two views
    vector<Point2d> l_pts, r_pts;
    getCorrespondences(l_pts, r_pts);

    Matx34d p0(Matx34d::eye());
    Matx34d p1(Matx34d::eye());
    if (findCameraMatrices(l_pts, r_pts, p0, p1)) {
        cout << "Initial reconstuction succeed" << endl;
        key_pts_vec.push_back(curr_key_pts);
        cam_mat_vec.push_back(p0);
        cam_mat_vec.push_back(p1);
        adjustBundle(point_cloud, corr_imgpts, K, cam_mat_vec, key_pts_vec);
        processor = bind(&RecLab::addView, this, _1);
        return true;
    } else {
        cout << "Initial reconstuction failed, try next pair" << endl;
        matches_vec.pop_back();
        frame_vec[0] = curr_frame;
        key_pts_vec[0] = curr_key_pts;
    }

    return false;
}


bool RecLab::addView(const Mat& frame)
{
    cout << "Adding a new view..." << endl;

    vector<DMatch> matches;
    vector<Point2d> l_pts, r_pts;
    feature_matcher->match(curr_frame, prev_frame, curr_key_pts, prev_key_pts, matches);

    // Get 3D-2D correspondences
    vector<cv::Point3f> obj_pts;
    vector<cv::Point2f> img_pts;
    vector<DMatch> non_corresp;
    vector<vector<int>> old_corr_imgpts(corr_imgpts);
    for (const auto& m : matches) {
        int idx = distance(corr_imgpts.begin(), find_if(corr_imgpts.begin(), corr_imgpts.end(),
                                                        [&m, this](const vector<int>& img_pt) {
            return img_pt.back() == m.trainIdx && img_pt.size() == frame_count;
        }));
        if (idx != corr_imgpts.size()) {
            obj_pts.emplace_back(point_cloud[idx].x, point_cloud[idx].y, point_cloud[idx].z);
            img_pts.emplace_back(curr_key_pts[m.queryIdx].pt);
            corr_imgpts[idx].push_back(m.queryIdx);
        } else {
            l_pts.push_back(prev_key_pts[m.trainIdx].pt);
            r_pts.push_back(curr_key_pts[m.queryIdx].pt);
            non_corresp.push_back(m);
        }
    }
    for (auto& img_pt : corr_imgpts) {
        if (img_pt.size() == frame_count) {
            img_pt.push_back(-1);
        }
    }
    matches_vec.push_back(move(matches));

    // last estimated camera matrix
    Matx34d prev_P = cam_mat_vec.back();
    Vec3d rvec, tvec;
    getOrientation(prev_P, rvec, tvec);

    // Estimate current camera matrix
    vector<int> inliers;
    double min_val, max_val;
    minMaxIdx(img_pts, &min_val, &max_val);
    solvePnPRansac(obj_pts, img_pts, K, noArray(), rvec, tvec, true, 1000,
                   0.006 * max_val, 0.25 * img_pts.size(), inliers, CV_EPNP);
    cout << "Estimating pose, " << inliers.size() << "/" << img_pts.size()
         << " valid correspondeces" << endl;
    // Not enough inliers
    if (inliers.size() * 5 < img_pts.size()) {
        cout << "Adding new view failed" << endl;
        swap(old_corr_imgpts, corr_imgpts);
        return false;
    }

    Matx33d R;
    Rodrigues(rvec, R);
    Matx34d curr_P = getCameraMat(R, tvec);
    cam_mat_vec.push_back(curr_P);

    PtCloud cloud;
    triangulate(l_pts, r_pts, K*prev_P, K*curr_P, cloud);
    point_cloud += cloud;

    int view_count = frame_count + 1;
    vector<int> img_pts_idx(view_count);
    for (const auto& m : non_corresp) {
        img_pts_idx.assign(view_count, -1);
        *(img_pts_idx.rbegin() + 1) = m.trainIdx;
        img_pts_idx.back() = m.queryIdx;

        corr_imgpts.push_back(img_pts_idx);
    }
    key_pts_vec.push_back(move(curr_key_pts));
    adjustBundle(point_cloud, corr_imgpts, K, cam_mat_vec, key_pts_vec);

    return true;
}


void RecLab::denseReconstruct()
{
    // Convert all to gray
    double factor = 480.0 / frame_vec[0].cols;
    for (auto& img : frame_vec) {
        cvtColor(img, img, CV_RGB2GRAY);
    }

    Size img_sz = frame_vec[0].size();
    int pw_size = matches_vec.size();
    for (int i = 0; i < 1/*pw_size*/; ++i) {
        vector<Point2d> l_pts, r_pts;
        for (const auto& m : matches_vec[i]) {
            l_pts.push_back(key_pts_vec[i][m.trainIdx].pt);
            r_pts.push_back(key_pts_vec[i + 1][m.queryIdx].pt);
        }
        double max_val = 0;
        minMaxIdx(l_pts, nullptr, &max_val);
        Matx33d F = findFundamentalMat(l_pts, r_pts, FM_RANSAC, 0.006 * max_val);

        Mat left = frame_vec[i];
        Mat right = frame_vec[i + 1];
        Matx33d H1, H2;
        Mat rec_l, rec_r;
        stereoRectifyUncalibrated(l_pts, r_pts, F, img_sz, H1, H2);
        warpPerspective(frame_vec[i], rec_l, H1, img_sz);
        warpPerspective(frame_vec[i + 1], rec_r, H2, img_sz);

        denseMatch(rec_l, rec_r, H1, H2, l_pts, r_pts);
        PtCloud cloud;
        triangulate(l_pts, r_pts, K*cam_mat_vec[i], K*cam_mat_vec[i + 1], cloud);
        dense_cloud += cloud;
    }

    writePointCloud("dense.obj", dense_cloud);
}


bool RecLab::findCameraMatrices(const vector<Point2d>& l_pts,
                                const vector<Point2d>& r_pts,
                                Matx34d& P0, Matx34d& P1)
{
    // Too few correspondences
    const size_t pt_size = l_pts.size();
    if (pt_size < 100) return false;

    double max_val;
    minMaxIdx(l_pts, nullptr, &max_val);
    Matx33d F = findFundamentalMat(l_pts, r_pts, FM_RANSAC, 0.006 * max_val);
    Matx33d E = K.t() * F * K;

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
    PtCloud pt_cloud;
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


void RecLab::getOrientation(InputArray _P, Vec3d& rvec, Vec3d& tvec) const
{
    Mat P = _P.getMat();
    Mat R = P(Rect(0, 0, 3, 3));
    Rodrigues(R, rvec);
    tvec = P.col(3);
}


bool RecLab::triangulate(const vector<Point2d>& l_pts,
                         const vector<Point2d>& r_pts,
                         const Matx34d& P0, const Matx34d& P1,
                         PtCloud& pt_cloud) const
{
    size_t pt_size = l_pts.size();
    pt_cloud.resize(pt_size);

    double mse = 0;
    for (size_t i = 0; i < pt_size; i++) {
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
        pt_cloud[i] = clany::Point3d(X(0), X(1), X(2));
    }
    mse /= pt_size * 2;

    return mse < 100 && isInFrontOfCam(pt_cloud, P0) && isInFrontOfCam(pt_cloud, P1);
}


bool RecLab::testTriangulate(const vector<Point2d>& l_pts, const vector<Point2d>& r_pts,
                             Matx34d& P0, Matx34d& P1, PtCloud& pt_cloud)
{
    if (triangulate(l_pts, r_pts, P0, P1, pt_cloud)) {
        auto& curr_match = matches_vec.back();
        point_cloud += pt_cloud;
        for (const auto& m : curr_match) {
            corr_imgpts.push_back({m.trainIdx, m.queryIdx});
        }
        return true;
    }

    return false;
}


bool RecLab::isInFrontOfCam(const PtCloud& point_cloud, const Matx34d& P)  const
{

    Matx14d p3 = P.row(2);
    double det_R = determinant(Mat(P)(Rect(0, 0, 3, 3)));
    double num_ifc = count_if(point_cloud.begin(), point_cloud.end(),
                              [&p3, &det_R](const Point3d& pt) {
        double w = p3.dot(Matx14d(pt.x, pt.y, pt.z, 1));
        return w * det_R > 0;
    });

    return num_ifc / point_cloud.size() > 0.9;
}


double RecLab::pointDepth(const clany::Point3d& pt, const cv::Matx34d& P) const
{
    Matx33d M = Mat(P)(Rect(0, 0, 3, 3));
    Matx14d p3 = P.row(2);

    double w = p3.dot(Matx14d(pt.x, pt.y, pt.z, 1));
    double depth = w / norm(M.row(2));
    if (determinant(M) < 0) depth = -depth;

    return depth;
}


void RecLab::getCorrespondences(vector<Point2d>& l_pts, vector<Point2d>& r_pts)
{
    vector<DMatch> matches;
    feature_matcher->match(curr_frame, prev_frame, curr_key_pts, prev_key_pts, matches);

    for (const auto& m : matches) {
        r_pts.push_back(curr_key_pts[m.queryIdx].pt);
        l_pts.push_back(prev_key_pts[m.trainIdx].pt);
    }
    matches_vec.push_back(move(matches));

#ifndef NDEBUG
    Mat img_matches;
    drawMatches(curr_frame, curr_key_pts, frame_vec.back(), key_pts_vec.back(), matches_vec.back(), img_matches);
#endif
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


void RecLab::denseMatch(const Mat& left_img, const Mat& right_img,
                        const Matx33d& H1, const Matx33d H2,
                        vector<Point2d>& l_pts, vector<Point2d>& r_pts)
{
    if (!l_pts.empty()) l_pts.clear();
    if (!r_pts.empty()) r_pts.clear();

    int wind_sz = curr_frame.cols / 100;
    if (wind_sz % 2 == 0) ++wind_sz;
    int num_disp = ((curr_frame.cols / 8) + 15) & -16;

    StereoBM block_match;
    block_match.state->SADWindowSize = wind_sz;
    block_match.state->minDisparity = 0;
    block_match.state->numberOfDisparities = num_disp;

//     StereoSGBM block_match;
//     block_match.SADWindowSize = wind_sz;
//     block_match.P1 = 8 * block_match.SADWindowSize * block_match.SADWindowSize;
//     block_match.P2 = 4 * block_match.P1;
//     block_match.minDisparity = 0;
//     block_match.numberOfDisparities = num_disp;
//     block_match.uniquenessRatio = 10;
//     block_match.speckleWindowSize = 100;
//     block_match.speckleRange = 2;
//     block_match.disp12MaxDiff = 1;

//     StereoVar block_match;
//     block_match.minDisp = -num_disp;
//     block_match.maxDisp = 0;
//     block_match.poly_n = 7;
//     block_match.poly_sigma = 1.1;
//     block_match.fi = 15.0f;
//     block_match.lambda = 0.03f;
//     block_match.flags = StereoVar::USE_SMART_ID | StereoVar::USE_AUTO_PARAMS |
//                         StereoVar::USE_INITIAL_DISPARITY | StereoVar::USE_MEDIAN_FILTERING;

    using DISP_TYPE = short;

    Mat disp;
    block_match(left_img, right_img, disp);

    Matx33d H1_inv = H1.inv();
    Matx33d H2_inv = H2.inv();
    float y_factor = disp.rows / 480.0;
    float x_factor = disp.cols / 640.0;
    for (int i = 0; i < 480; ++i) {
        int y = cvRound(i * y_factor);
        DISP_TYPE* data = disp.ptr<DISP_TYPE>(y);
    	for (int j = 0; j < 640; ++j) {
            int x = cvRound(j * x_factor);
            double disp_val = data[x] / 16.0;
            if (disp_val > 0) {
                Matx31d l_pt_homo = H1_inv * Matx31d(x, y, 1);
                Matx31d r_pt_homo = H2_inv * Matx31d(x - disp_val, y, 1);
                l_pts.emplace_back(l_pt_homo(0) / l_pt_homo(2), l_pt_homo(1) / l_pt_homo(2));
                r_pts.emplace_back(r_pt_homo(0) / r_pt_homo(2), r_pt_homo(1) / r_pt_homo(2));
    		}
    	}
    }
}


void RecLab::writePointCloud(const string& file_name, const PtCloud& cloud) const
{
    ofstream ofs(file_name);
    for (const auto& point : cloud) {
        ofs << "v " << point << endl;
    }
}
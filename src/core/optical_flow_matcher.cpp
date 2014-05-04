#include <set>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "core/optical_flow_matcher.h"

using namespace std;
using namespace cv;
using namespace clany;


OFMatcher::OFMatcher(const string& detector_type)
    : detector(FeatureDetector::create(detector_type)),
      matcher(NORM_L2, true)
{}


void OFMatcher::detect(const Mat& img, vector<KeyPoint>& ft_pts)
{
    if (!ft_pts.empty()) ft_pts.clear();

    detector->detect(img, ft_pts);
}


void OFMatcher::match(const Mat& img_a, const Mat& img_b,
                      vector<KeyPoint>& ft_pts_a, vector<KeyPoint>& ft_pts_b,
                      vector<DMatch>& matches)
{
    if (ft_pts_a.empty()) detector->detect(img_a, ft_pts_a);
    if (ft_pts_b.empty()) detector->detect(img_b, ft_pts_b);

    Mat gray_a, gray_b;
    if (img_a.channels() == 3) {
        cvtColor(img_a, gray_a, CV_RGB2GRAY);
        cvtColor(img_b, gray_b, CV_RGB2GRAY);
    } else {
        gray_a = img_a;
        gray_b = img_b;
    }

    vector<Point2f> pts_a(ft_pts_a.size()), pts_b(ft_pts_a.size());
    KeyPoint::convert(ft_pts_a, pts_a);

    vector<uchar> status(pts_a.size());
    vector<float> error(pts_a.size());
    calcOpticalFlowPyrLK(gray_a, gray_b, pts_a, pts_b, status, error);

    vector<int> good_pts_idx;
    vector<Point2f> good_pts;
    float dist_thresh = img_a.cols / 10.f;
    for (uint i = 0; i < status.size(); ++i) {
        Point2f delta = pts_a[i] - pts_b[i];
        float dist = sqrt(delta.dot(delta));
        if (status[i] && error[i] < 12/* && dist < dist_thresh*/) {
            good_pts_idx.push_back(i);
            good_pts.push_back(pts_b[i]);
        } else {
            status[i] = 0;
        }
    }

    KeyPoint::convert(ft_pts_b, pts_b);
    Mat pts_a_mat = Mat(good_pts).reshape(1);
    Mat pts_b_mat = Mat(pts_b).reshape(1);

    vector<vector<DMatch>> knn_matches;
    matcher.radiusMatch(pts_a_mat, pts_b_mat, knn_matches, 2.f);

    set<int> matched_pt;
    for (const auto& curr_m : knn_matches) {
        DMatch match;
        if (curr_m.size() == 1) {
            match = curr_m[0];
        } else if (curr_m.size() > 1 &&
                   curr_m[0].distance / curr_m[1].distance < 0.7) {
                match = curr_m[0];
        } else {
            continue;
        }

        if (matched_pt.find(match.trainIdx) == matched_pt.end()) {
            match.queryIdx = good_pts_idx[match.queryIdx];
            matches.push_back(match);
            matched_pt.insert(match.trainIdx);
        }
    }

#ifndef NDEBUG
    Mat img_matches;
    drawMatches(img_a, ft_pts_a, img_b, ft_pts_b, matches, img_matches);
#endif
}
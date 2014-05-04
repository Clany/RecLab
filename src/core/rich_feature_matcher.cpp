#include <set>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "core/rich_feature_matcher.h"

using namespace std;
using namespace cv;
using namespace clany;


RichFTMatcher::RichFTMatcher(const string& detector_type, const string& extractor_type)
    : detector(FeatureDetector::create(detector_type)),
      /*detector(new GridAdaptedFeatureDetector(FeatureDetector::create(detector_type), 10000)),*/
      extractor(DescriptorExtractor::create(extractor_type))
{
    if (extractor_type == "ORB" || extractor_type == "BRISK" || extractor_type == "BRIEF") {
        matcher = make_shared<BFMatcher>(NORM_HAMMING, true);
    } else {
        matcher = make_shared<BFMatcher>(NORM_L2, true);
    }
}


void RichFTMatcher::detect(const Mat& img, vector<KeyPoint>& ft_pts)
{
    if (!ft_pts.empty()) ft_pts.clear();

    detector->detect(img, ft_pts);
    extractor->compute(img, ft_pts, Mat());
}


void RichFTMatcher::match(const Mat& img_a, const Mat& img_b,
                          vector<KeyPoint>& ft_pts_a, vector<KeyPoint>& ft_pts_b,
                          vector<DMatch>& matches)
{
    if (ft_pts_a.empty()) detector->detect(img_a, ft_pts_a);
    if (ft_pts_b.empty()) detector->detect(img_b, ft_pts_b);

    Mat descriptors_a;
    Mat descriptors_b;
    extractor->compute(img_a, ft_pts_a, descriptors_a);
    extractor->compute(img_b, ft_pts_b, descriptors_b);
    matcher->match(descriptors_a, descriptors_b, matches);

    vector<Point2f> pts_a_good, pts_b_good;
    set<int> matched_pt;
    for (auto& m : matches) {
        if (matched_pt.find(m.trainIdx) == matched_pt.end()) {
            pts_a_good.push_back(ft_pts_a[m.queryIdx].pt);
            pts_b_good.push_back(ft_pts_b[m.trainIdx].pt);
            matched_pt.insert(m.trainIdx);
        }
    }

    double max_val;
    cv::minMaxIdx(pts_a_good, nullptr, &max_val);
    vector<uchar> status(pts_a_good.size());
    Matx33d F = findFundamentalMat(pts_a_good, pts_b_good, CV_FM_RANSAC, 0.006 * max_val, 0.99, status);

    vector<DMatch> good_matches(countNonZero(status));
    auto status_iter = status.begin();
    copy_if(matches.begin(), matches.end(), good_matches.begin(), [&status_iter](const DMatch&) {
        return *status_iter++;
    });
#ifndef NDEBUG
    Mat matches_before, matches_after;
    drawMatches(img_a, ft_pts_a, img_b, ft_pts_b, matches, matches_before);
    drawMatches(img_a, ft_pts_a, img_b, ft_pts_b, good_matches, matches_after);
#endif
    matches = move(good_matches);
}
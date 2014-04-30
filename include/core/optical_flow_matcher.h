#ifndef OPTICAL_FLOW_MATCHER_H
#define OPTICAL_FLOW_MATCHER_H

#include "feature_matcher.h"


_CLANY_BEGIN
class OFMatcher : public FeatureMatcher
{
public:
    OFMatcher(const string& detector_type = "GFTT");

    void detect(const cv::Mat& img, vector<cv::KeyPoint>& ft_pts) override;

    void match(const cv::Mat& img_a, const cv::Mat& img_b,
               vector<cv::KeyPoint>& ft_pts_a, vector<cv::KeyPoint>& ft_pts_b,
               vector<cv::DMatch>& matches) override;

private:
    cv::Ptr<cv::FeatureDetector> detector;
    cv::BFMatcher matcher;
};
_CLANY_END

#endif // OPTICAL_FLOW_MATCHER_H
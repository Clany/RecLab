#ifndef RICH_FEATURE_MATCHER_H
#define RICH_FEATURE_MATCHER_H

#include <memory>
#include "feature_matcher.h"


_CLANY_BEGIN
class RichFTMatcher : public FeatureMatcher
{
public:
    RichFTMatcher(const string& detector_type = "FAST", const string& descriptor_extractor_type = "ORB");

    void detect(const cv::Mat& img, vector<cv::KeyPoint>& ft_pts) override;

    void match(const cv::Mat& img_a, const cv::Mat& img_b,
               vector<cv::KeyPoint>& ft_pts_a, vector<cv::KeyPoint>& ft_pts_b,
               vector<cv::DMatch>& matches) override;

private:
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    unique_ptr<cv::BFMatcher> matcher;
};
_CLANY_END

#endif // RICH_FEATURE_MATCHER_H
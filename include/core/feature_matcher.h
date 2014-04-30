#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <memory>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "clany/factory.hpp"


_CLANY_BEGIN
class FeatureMatcher
{
public:
    using Ptr = shared_ptr<FeatureMatcher>;

    FeatureMatcher() = default;
    virtual ~FeatureMatcher() {};
    FeatureMatcher(const FeatureMatcher&) = delete;
    FeatureMatcher& operator=(const FeatureMatcher&) = delete;

    virtual void detect(const cv::Mat& img, vector<cv::KeyPoint>& ft_pts) = 0;

    virtual void match(const cv::Mat& img_a, const cv::Mat& img_b,
                       vector<cv::KeyPoint>& ft_pts_a, vector<cv::KeyPoint>& ft_pts_b,
                       vector<cv::DMatch>& matches) = 0;
};


using MatcherCreator = FeatureMatcher::Ptr(const string&);
using MatcherFactory = ObjFactory<FeatureMatcher, string, MatcherCreator>;
_CLANY_END


#endif // FEATURE_MATCHER_H
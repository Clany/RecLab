#include "core/optical_flow_matcher.h"
#include "core/rich_feature_matcher.h"


using namespace clany;

namespace  {
    const bool ADD_RICH_FEATURE_MATCHER =
        MatcherFactory::addType("RichFeature", Factory<RichFTMatcher>());
    const bool ADD_OPTICAL_FLOW_MATCHER =
        MatcherFactory::addType("OpticalFlow", Factory<OFMatcher>());
}
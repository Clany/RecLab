#include "reclab_config.h"
#include "utilities/img_seq_grabber.h"
#include "utilities/video_grabber.h"
#include "utilities/cam_grabber.h"


using namespace clany;

namespace {
    const bool ADD_IMG_SEQ_GRABBER =
        ImgSeqGrabberFactory::addType("Sequence", Factory<ImgSeqGrabber>());
    const bool ADD_VIDEO_GRABBER =
        VideoGrabberFactory::addType("Video", Factory<VideoGrabber>());
    const bool ADD_CAMERA_GRABBER =
        SensorGrabberFactory::addType("Camera", Factory<CamGrabber>());
}


#if ENABLE_KINECT
#include "utilities/kinect_grabber.h"

namespace  {
    const bool ADD_KINECT_GRABBER =
        SensorGrabberFactory::addType("Kinect", Factory<KinectGrabber>());
}
#endif // ENABLE_KINECT
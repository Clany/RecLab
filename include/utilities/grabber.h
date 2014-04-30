#ifndef GRABBER_H
#define GRABBER_H

#include <stdexcept>
#include <opencv2/highgui/highgui.hpp>
#include "clany/factory.hpp"

_CLANY_BEGIN
class Grabber
{
public:
    using Ptr = shared_ptr<Grabber>;

    Grabber() = default;
    virtual ~Grabber() {};
    Grabber(const Grabber&) = delete;
    Grabber& operator=(const Grabber&) = delete;

    virtual bool isOpened() = 0;
    virtual bool read(cv::Mat& image) = 0;
};


class GrabberExcept : public runtime_error
{
public:
    GrabberExcept(const string& err_msg) : runtime_error(err_msg) {};
};


using SensorGrabberCreator = Grabber::Ptr(int);
using VideoGrabberCreator  = Grabber::Ptr(const string&);
using ImgSeqGrabberCreator = Grabber::Ptr(const vector<cv::Mat>& img_seq);

using SensorGrabberFactory = ObjFactory<Grabber, string, SensorGrabberCreator>;
using VideoGrabberFactory  = ObjFactory<Grabber, string, VideoGrabberCreator>;
using ImgSeqGrabberFactory = ObjFactory<Grabber, string, ImgSeqGrabberCreator>;
_CLANY_END

#endif // GRABBER_H

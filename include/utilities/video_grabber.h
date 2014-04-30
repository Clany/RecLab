#ifndef VIDEO_GRABBER_H
#define VIDEO_GRABBER_H

#include "grabber.h"

_CLANY_BEGIN
class VideoGrabber : public Grabber
{
public:
    explicit VideoGrabber(const string& file_name) :video {file_name} {};

    bool isOpened() override { return video.isOpened(); };
    bool read(cv::Mat& frame) override { return video.read(frame); };

protected:
    cv::VideoCapture video;
};
_CLANY_END

#endif // VIDEO_GRABBER_H
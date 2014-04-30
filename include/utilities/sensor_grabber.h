#ifndef FRAME_GRABBER_H
#define FRAME_GRABBER_H

#include <stdexcept>
#include <opencv2/highgui/highgui.hpp>

#include "clany/factory.hpp"
#include "grabber.h"


_CLANY_BEGIN
class SensorGrabber : public Grabber
{
public:
    using Ptr = shared_ptr<SensorGrabber>;
    enum class FrameType { COLOR, DEPTH };

    virtual ~SensorGrabber() {};

    virtual bool setSensor(int idx) = 0;
    virtual bool start() = 0;
    virtual void stop() = 0;
};
_CLANY_END

#endif // FRAME_GRABBER_H
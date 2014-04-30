#ifndef CAM_GRABBER_H
#define CAM_GRABBER_H

#include "sensor_grabber.h"

_CLANY_BEGIN
class CamGrabber : public SensorGrabber
{
public:
    explicit CamGrabber(int sensor_idx) :camera {sensor_idx} {};

    bool setSensor(int idx) override { return camera.open(idx); };
    bool isOpened()         override { return camera.isOpened(); };
    bool start() override { return true; };
    void stop()  override {};

    bool read(cv::Mat& frame) override { return camera.read(frame); };

protected:
    cv::VideoCapture camera;
};
_CLANY_END

#endif // CAM_GRABBER_H
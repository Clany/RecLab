#pragma once

#define NOMINMAX
#include <Windows.h>
#include <NuiApi.h>
#include <string>
#include <memory>

#include "sensor_grabber.h"

_CLANY_BEGIN
class KinectGrabber : public SensorGrabber
{
protected:
    struct SensorInfo
    {
        INuiSensor* NuiSensor;
        HANDLE VideoStreamHandle;
        HANDLE DepthStreamHandle;
        SensorInfo () : NuiSensor(NULL), DepthStreamHandle(NULL), VideoStreamHandle(NULL) {}
    };

public:
    KinectGrabber();
    explicit KinectGrabber(int kinectIdx);
    ~KinectGrabber();

    bool setSensor(int idx) override;
    bool isOpened()         override;
    bool start()            override;
    void stop()             override;

    bool read(cv::Mat& data) override;

    size_t getDeviceCount();
    std::wstring getDeviceName();

    bool resetSensor(int index);


    bool getDepthFrame(uchar*& depth);
    bool getColorFrame(uchar*& color);

    bool isDeviceOK();
    bool isGrabbingStarted();

private:
    void finalize();

protected:
    // Sensor information
    SensorInfo m_Sensor;
    // Coordinate mapper, map depth frame to color space
    unique_ptr<NUI_COLOR_IMAGE_POINT[]> m_pD2CMapper;
};
_CLANY_END
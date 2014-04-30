#include "utilities/kinect_grabber.h"

using namespace clany;
using namespace std;


KinectGrabber::KinectGrabber() :
m_Sensor(), m_pD2CMapper(make_unique<NUI_COLOR_IMAGE_POINT[]>(640*480))
{}


KinectGrabber::KinectGrabber(int kinectIdx) :
m_pD2CMapper(make_unique<NUI_COLOR_IMAGE_POINT[]>(640 * 480))
{
    NuiCreateSensorByIndex(kinectIdx, &m_Sensor.NuiSensor);
    start();
}


KinectGrabber::~KinectGrabber()
{
    finalize();
}


bool KinectGrabber::setSensor(int idx)
{
    if (m_Sensor.NuiSensor) {
        return false;
    }

    if (FAILED(NuiCreateSensorByIndex(idx, &m_Sensor.NuiSensor))) {
        return false;
    }

    return true;
}


bool KinectGrabber::start()
{
    if (!m_Sensor.NuiSensor)
        return false;

    HANDLE hNewDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    HANDLE hNewVideoFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    if (FAILED(m_Sensor.NuiSensor->NuiInitialize(
        NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX |
        NUI_INITIALIZE_FLAG_USES_COLOR)))
        return false;

    if (FAILED(m_Sensor.NuiSensor->NuiImageStreamOpen(
        NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX,
        NUI_IMAGE_RESOLUTION_640x480,
        0,
        2,
        hNewDepthFrameEvent,
        &m_Sensor.DepthStreamHandle)))
        return false;
    m_Sensor.NuiSensor->NuiImageStreamSetImageFrameFlags(m_Sensor.DepthStreamHandle, NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE);


    if (FAILED(m_Sensor.NuiSensor->NuiImageStreamOpen(
        NUI_IMAGE_TYPE_COLOR,
        NUI_IMAGE_RESOLUTION_640x480,
        0,
        2,
        hNewVideoFrameEvent,
        &m_Sensor.VideoStreamHandle)))
        return false;

    return true;
}


void KinectGrabber::stop()
{
    finalize();
}


bool KinectGrabber::isOpened()
{
    return isDeviceOK();
}


bool KinectGrabber::read(cv::Mat& data)
{
    data.create(480, 640, CV_8UC4);
    if (getColorFrame(data.data)) {
        flip(data, data, 1);
        return true;
    }

    return false;
}


bool KinectGrabber::resetSensor(int index)
{
    if (!FAILED(NuiCreateSensorByIndex(index, &m_Sensor.NuiSensor)))
        return true;
    else
        return false;
}


size_t KinectGrabber::getDeviceCount()
{
    int count = 0;
    if (FAILED(NuiGetSensorCount(&count)))
        return 0;
    return (size_t)count;
}


wstring KinectGrabber::getDeviceName()
{
    if (m_Sensor.NuiSensor != NULL) {
        wstring name = m_Sensor.NuiSensor->NuiDeviceConnectionId();
        return name;
    }
    return wstring(L"Unknown Kinect Sensor");
}


bool KinectGrabber::getDepthFrame(uchar*& depth)
{
    if (!isGrabbingStarted())
        return false;

    NUI_IMAGE_FRAME ImageFrame;
    if (FAILED(m_Sensor.NuiSensor->NuiImageStreamGetNextFrame(m_Sensor.DepthStreamHandle, 100, &ImageFrame)))
        return false;

    INuiFrameTexture* pTexture = ImageFrame.pFrameTexture;
    NUI_LOCKED_RECT LockedRect;
    if (FAILED(pTexture->LockRect(0, &LockedRect, NULL, 0)))
        return false;

    depth = LockedRect.pBits;
    pTexture->UnlockRect(0);
    //////////////////////////////////////////////////////////////////////////
    // Mapper
    BOOL bNearMode;
    INuiFrameTexture* pPixelFrameTexture;
    if (FAILED(m_Sensor.NuiSensor->NuiImageFrameGetDepthImagePixelFrameTexture(m_Sensor.DepthStreamHandle, &ImageFrame, &bNearMode, &pPixelFrameTexture)))
        return false;

    NUI_LOCKED_RECT PixelFrameRect;
    if (FAILED(pPixelFrameTexture->LockRect(0, &PixelFrameRect, NULL, 0)))
        return false;

    INuiCoordinateMapper* pMapper;
    m_Sensor.NuiSensor->NuiGetCoordinateMapper(&pMapper);
    pMapper->MapDepthFrameToColorFrame(
        NUI_IMAGE_RESOLUTION_640x480,
        640 * 480,
        (NUI_DEPTH_IMAGE_PIXEL*)PixelFrameRect.pBits,
        NUI_IMAGE_TYPE_COLOR,
        NUI_IMAGE_RESOLUTION_640x480,
        640 * 480,
        m_pD2CMapper.get());

    pPixelFrameTexture->UnlockRect(0);
    //////////////////////////////////////////////////////////////////////////
    m_Sensor.NuiSensor->NuiImageStreamReleaseFrame(m_Sensor.DepthStreamHandle, &ImageFrame);
    return true;
}


bool KinectGrabber::getColorFrame(uchar*& color)
{
    if (!isGrabbingStarted())
        return false;

    NUI_IMAGE_FRAME pImageFrame;
    if (FAILED(m_Sensor.NuiSensor->NuiImageStreamGetNextFrame(m_Sensor.VideoStreamHandle, 100, &pImageFrame)))
        return false;

    INuiFrameTexture* pTexture = pImageFrame.pFrameTexture;
    NUI_LOCKED_RECT LockedRect;
    if (FAILED(pTexture->LockRect(0, &LockedRect, NULL, 0)))
        return false;

    color = LockedRect.pBits;
    pTexture->UnlockRect(0);

    m_Sensor.NuiSensor->NuiImageStreamReleaseFrame(m_Sensor.VideoStreamHandle, &pImageFrame);
    return true;
}


bool KinectGrabber::isDeviceOK()
{
    return m_Sensor.NuiSensor != NULL;
}


bool KinectGrabber::isGrabbingStarted()
{
    return (m_Sensor.NuiSensor != NULL &&
        m_Sensor.DepthStreamHandle != NULL &&
        m_Sensor.VideoStreamHandle != NULL);
}


//////////////////////////////////////////////////////////////////////////
// Private function for internal call
void KinectGrabber::finalize()
{
    if (m_Sensor.NuiSensor)
    {
        m_Sensor.NuiSensor->NuiShutdown();
        m_Sensor.NuiSensor->Release();
        m_Sensor.NuiSensor = NULL;
        m_Sensor.DepthStreamHandle = NULL;
        m_Sensor.VideoStreamHandle = NULL;
    }
}
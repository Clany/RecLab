#include "reclab_app.h"
#include "clany/timer.hpp"

using namespace std;
using namespace cv;
using namespace clany;


RecLabApp::RecLabApp(int cam_idx)
{
    grabber = SensorGrabberFactory::create("Kinect", cam_idx);
    if (!grabber || !grabber->isOpened()) {
        throw GrabberExcept("Cannot open device");
    }
}


RecLabApp::RecLabApp(const string& file_name)
{
    grabber = VideoGrabberFactory::create("Video", file_name);
    if (!grabber || !grabber->isOpened()) {
        throw GrabberExcept("Cannot open file");
    }
}


RecLabApp::RecLabApp(const vector<cv::Mat>& img_seq)
    : rec_lab(img_seq[0].rows, img_seq[0].cols), img_seq_input(true)
{
    grabber = ImgSeqGrabberFactory::create("Sequence", img_seq);
    if (!grabber || !grabber->isOpened()) {
        throw GrabberExcept("Invalid image sequence");
    }
}


void RecLabApp::startProc()
{
    cv::namedWindow("Frame");
    cout << "---------Reconstruction begin---------" << endl;

    Mat frame, display;
    int count = 0;
    CPUTimer timer;
    do{
        if (grabber->read(frame)) {
            double factor = 1000.0 / frame.cols;
            if (frame.cols > 1000) resize(frame, display, Size(), factor, factor);
            else display = frame;
            imshow("Frame", display);
            rec_lab(frame);
        } else {
            if (img_seq_input) break;
        }

        char c = (char)waitKey(30);
        if (c == 'e' || c == 'E') break;
        if (c == 's' || c == 'S') imwrite("kin" + to_string(count++) + ".jpg", frame);
    } while (true);
    timer.elapsed("Time used");
    destroyAllWindows();

    if (rec_lab.isSuccess()) {
        rec_lab.postProcess();
        cout << "---------Reconstruction done---------" << endl;
        rec_lab.writeResult();
    } else {
        cout << "---------Reconstruction fail---------" << endl;
    }
}


void RecLabApp::parseParameters(int argc, char* argv[])
{
    if (argc >= 3 && string(argv[2]) == "--optical_flow") {
        rec_lab.setMatchMethod(RecLab::MatchMethod::OpticalFlow);
    }
}
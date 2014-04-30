#include "reclab_app.h"

using namespace std;
using namespace cv;
using namespace clany;


RecLabApp::RecLabApp(int cam_idx, cv::InputArray calib_mat) :
    rec_lab(calib_mat)
{
    grabber = SensorGrabberFactory::create("Kinect", cam_idx);
    if (!grabber || !grabber->isOpened()) {
        throw GrabberExcept("Cannot open device");
    }
}


RecLabApp::RecLabApp(const string& file_name, cv::InputArray calib_mat) :
    rec_lab(calib_mat)
{
    grabber = VideoGrabberFactory::create("Video", file_name);
    if (!grabber || !grabber->isOpened()) {
        throw GrabberExcept("Cannot open file");
    }
}


RecLabApp::RecLabApp(const vector<cv::Mat>& img_seq, cv::InputArray calib_mat) :
    rec_lab(calib_mat), img_seq_input(true)
{
    grabber = ImgSeqGrabberFactory::create("Sequence", img_seq);
    if (!grabber || !grabber->isOpened()) {
        throw GrabberExcept("Invalid image sequence");
    }
}


void RecLabApp::startProc()
{
    cv::namedWindow("Frame");

    Mat frame;
    char c;
    int count = 0;
    do{
        if (grabber->read(frame)) {
            imshow("Frame", frame);
            rec_lab(frame);
        } else {
            if (img_seq_input) break;
        }

        c = (char)waitKey(30);
        if (c == 'e' || c == 'E') break;
        if (c == 's' || c == 'S') imwrite("kin" + to_string(count++) + ".jpg", frame);
    } while (true);

    destroyAllWindows();

    if (rec_lab.isSuccess()) {
        cout << "Reconstruction done" << endl;
        rec_lab.writeResult();
    } else {
        cout << "Reconstruction fail" << endl;
    }
}
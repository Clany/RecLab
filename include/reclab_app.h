#ifndef RECLAB_APP_H
#define RECLAB_APP_H

#include "utilities/grabber.h"
#include "core/reclab.h"

_CLANY_BEGIN
class RecLabApp
{
public:
    // Camera
    explicit RecLabApp(int cam_idx = 0);

    // Video
    explicit RecLabApp(const string& file_name);

    // Image sequence
    explicit RecLabApp(const vector<cv::Mat>& img_seq);

    void startProc();

    void parseParameters(int argc, char* argv[]);

private:
    shared_ptr<Grabber> grabber;
    RecLab rec_lab;

    bool img_seq_input = false;
};
_CLANY_END

#endif // RECLAB_APP_H
#ifndef IMG_SEQ_GRABBER_H
#define IMG_SEQ_GRABBER_H

#include <vector>

#include "grabber.h"

_CLANY_BEGIN
class ImgSeqGrabber : public Grabber
{
public:
    explicit ImgSeqGrabber(const vector<cv::Mat>& img_seq) :seq {img_seq} {};
    explicit ImgSeqGrabber(vector<cv::Mat>&& img_seq) :seq {move(img_seq)} {};

    bool isOpened() override { return !seq.empty(); };

    bool read(cv::Mat& data) override
    {
        if (curr_idx == seq.size()) {
            return false;
        }
        data = seq[curr_idx];
        ++curr_idx;

        return true;
    };

protected:
    int curr_idx = 0;
    vector<cv::Mat> seq;
};
_CLANY_END

#endif // IMG_SEQ_GRABBER_H
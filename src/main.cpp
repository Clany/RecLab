#include <iostream>
#include <opencv2/viz/vizcore.hpp>
#include "reclab_app.h"
#include "clany/timer.hpp"

using namespace std;
using namespace cv;
using namespace clany;


const float FOCAL = 532.f;
const float X0 = 320.f;
const float Y0 = 260.f;

int main(int argc, char* argv[])
{
    cout << "Welcome to ReconstructLab" << endl;

    vector<Mat> img_seq;
    auto files = Directory::GetListFiles(argv[1], "*.*");
    for (const auto& fn : files) {
        img_seq.push_back(imread(fn));
    }

//     Matx33f K(FOCAL, 0,     X0,
//               0,     FOCAL, Y0,
//               0,     0,     1);
    Size img_sz = img_seq[0].size();
    float focal = static_cast<float>(max(img_sz.width, img_sz.height));
    Matx33f K(focal, 0,     img_sz.width / 2.f,
              0,     focal, img_sz.height / 2.f,
              0,     0,     1);

    try
    {
        RecLabApp rec_app(img_seq, K);
        rec_app.startProc();
    } catch (GrabberExcept& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (ConvertExcept& e) {
        cerr << "Error: " << e.what() << endl;
        return 2;
    } catch (...) {
        cerr << "Error: unknown exception" << endl;
        return 3;
    }

    return 0;
}
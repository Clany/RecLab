#include <iostream>
#include "reclab_app.h"
#include "clany/timer.hpp"

using namespace std;
using namespace cv;
using namespace clany;


int main(int argc, char* argv[])
{
    if (argc != 2 && argc != 3) {
        cout << "usage: " << argv[0] << " dir_name" << " [optional]--optical_flow" << endl;
        return 1;
    }

    vector<Mat> img_seq;
    auto files = Directory::GetListFiles(argv[1], "*.*");
    for (const auto& fn : files) {
        img_seq.push_back(imread(fn));
    }

    try
    {
        RecLabApp rec_app(img_seq);
        rec_app.parseParameters(argc, argv);
        rec_app.startProc();
    } catch (GrabberExcept& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Error: unknown exception" << endl;
        return 3;
    }

    return 0;
}
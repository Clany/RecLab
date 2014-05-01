#define V3DLIB_ENABLE_SUITESPARSE

#pragma warning(disable: 4800)
#pragma warning(disable: 4244)
#pragma warning(disable: 4018)

#include <ssba/Math/v3d_linear.h>
#include <ssba/Base/v3d_vrmlio.h>
#include <ssba/Geometry/v3d_metricbundle.h>

#include "core/bundle_adjust.h"

using namespace std;
using namespace cv;
using namespace V3D;
using namespace clany;


void getRTFromP(const Matx34d& P, Matrix3x3d& R, Vector3d& T)
{
    R[0][0] = P(0, 0); R[0][1] = P(0, 1); R[0][2] = P(0, 2); T[0] = P(0, 3);
    R[1][0] = P(1, 0); R[1][1] = P(1, 1); R[1][2] = P(1, 2); T[1] = P(1, 3);
    R[2][0] = P(2, 0); R[2][1] = P(2, 1); R[2][2] = P(2, 2); T[2] = P(2, 3);
}


void clany::adjustBundle(PtCloud& cloud, ImgPtCorrs& img_pts, Matx33d& _K, vector<Matx34d>& P_vec,
                         const vector<vector<KeyPoint>>& key_pts_vec)
{
    const size_t pc_sz = cloud.size();
    const size_t view_sz = P_vec.size();
    const size_t max_imgpts_sz = img_pts.size() * view_sz;

    double f0 = _K(0, 0);
    Matrix3x3d K_norm;
    memcpy(K_norm[0], _K.val, sizeof(double) * 9);
    scaleMatrixIP(1.0 / f0, K_norm);
    K_norm[2][2] = 1.0;

    vector<Vector3d> Xs(pc_sz);
    memcpy(Xs.data(), cloud.points.data(), sizeof(double) * 3 * pc_sz);

    vector<CameraMatrix> cams(view_sz);
    for (size_t i = 0; i < view_sz; ++i) {
        Matrix3x3d R;
        Vector3d T;
        getRTFromP(P_vec[i], R, T);

        cams[i].setIntrinsic(K_norm);
        cams[i].setRotation(R);
        cams[i].setTranslation(T);
    }

    vector<Vector2d> measurements;
    vector<int> correspondingView;
    vector<int> correspondingPoint;
    measurements.reserve(max_imgpts_sz);
    correspondingView.reserve(max_imgpts_sz);
    correspondingPoint.reserve(max_imgpts_sz);
    for (size_t i = 0; i < pc_sz; ++i) {
        for (size_t j = 0; j < view_sz; ++j) {
            if (img_pts[i][j] >= 0) {
                Point2f key_pt = key_pts_vec[j][img_pts[i][j]].pt;

                measurements.emplace_back(key_pt.x / f0, key_pt.y / f0);
                correspondingPoint.push_back(i);
                correspondingView.push_back(j);
            }
        }
    }

    double thresh = 2.0 / abs(f0);
    Matrix3x3d K_new = cams[0].getIntrinsic();
    bool good_adj = false;
    {
        ScopedBundleExtrinsicNormalizer ext_norm(cams, Xs);
        ScopedBundleIntrinsicNormalizer int_norm(cams, measurements, correspondingView);
        CommonInternalsMetricBundleOptimizer opt(FULL_BUNDLE_FOCAL_LENGTH_PP, thresh, K_new, StdDistortionFunction(), cams, Xs,
                                                 measurements, correspondingView, correspondingPoint);
        opt.tau = 1e-3;
        opt.maxIterations = 50;
        opt.minimize();

        good_adj = (opt.status != 2);
    }

    scaleMatrixIP(f0, K_new);
    K_new[2][2] = 1.0;
    memcpy(_K.val, K_new[0], sizeof(double)* 9);

    if (good_adj) {
        memcpy(cloud.points.data(), Xs.data(), sizeof(double)* 3 * pc_sz);

        //extract adjusted cameras
        for (auto& cam : cams) cam.setIntrinsic(K_new);

        for (int i = 0; i < view_sz; ++i) {
            Matrix3x4d RT = cams[i].getOrientation();
            memcpy(P_vec[i].val, RT[0], sizeof(double) * 12);
        }
    }
}
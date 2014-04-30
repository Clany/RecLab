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


void clany::adjustBundle(PtCloud& cloud, Matx33d& _K, vector<Matx34d>& P_vec,
                         const vector<vector<KeyPoint>>& key_pts_vec)
{
    size_t pc_sz = cloud.size();
    size_t cam_vec_sz = P_vec.size();
    size_t img_pts_sz = 0;
    for (const auto& pt : cloud) {
    	for (const auto& img_pt_idx : pt.second) {
            if (img_pt_idx >= 0) ++img_pts_sz;
    	}
    }

    Matrix3x3d K;
    makeIdentityMatrix(K);
    K[0][0] = _K(0, 0);
    K[1][1] = _K(1, 1);
    K[0][1] = _K(0, 1);
    K[0][2] = _K(0, 2);
    K[1][2] = _K(1, 2);
    double f0 = K[0][0];

    Matrix3x3d K_norm(K);
    scaleMatrixIP(1.0 / f0, K_norm);
    K_norm[2][2] = 1.0;

    //convert to BA datastructs
    vector<int> pointIdFwdMap(pc_sz);
    map<int, int> pointIdBwdMap;
    vector<Vector3d> Xs(pc_sz);
    for (int j = 0; j < pc_sz; ++j) {
        int pointId = j;
        Xs[j][0] = cloud[j].first.x;
        Xs[j][1] = cloud[j].first.y;
        Xs[j][2] = cloud[j].first.z;
        pointIdFwdMap[j] = pointId;
        pointIdBwdMap.insert(make_pair(pointId, j));
    }

    vector<int> camIdFwdMap(cam_vec_sz, -1);
    map<int, int> camIdBwdMap;
    vector<CameraMatrix> cams(cam_vec_sz);
    for (int i = 0; i < cam_vec_sz; ++i) {
        int camId = i;
        Matrix3x3d R;
        Vector3d T;

        Matx34d& P = P_vec[i];

        R[0][0] = P(0, 0); R[0][1] = P(0, 1); R[0][2] = P(0, 2); T[0] = P(0, 3);
        R[1][0] = P(1, 0); R[1][1] = P(1, 1); R[1][2] = P(1, 2); T[1] = P(1, 3);
        R[2][0] = P(2, 0); R[2][1] = P(2, 1); R[2][2] = P(2, 2); T[2] = P(2, 3);

        camIdFwdMap[i] = camId;
        camIdBwdMap.insert(make_pair(camId, i));

        cams[i].setIntrinsic(K_norm);
        cams[i].setRotation(R);
        cams[i].setTranslation(T);
    }

    vector<Vector2d > measurements;
    vector<int> correspondingView;
    vector<int> correspondingPoint;
    measurements.reserve(img_pts_sz);
    correspondingView.reserve(img_pts_sz);
    correspondingPoint.reserve(img_pts_sz);
    for (uint k = 0; k < cloud.size(); ++k) {
        for (uint i = 0; i < cloud[k].second.size(); i++) {
            if (cloud[k].second[i] >= 0) {
                int view_idx = i, pt_idx = k;
                Vector3d p, np;

                Point2f cvp = key_pts_vec[i][cloud[k].second[i]].pt;
                p[0] = cvp.x;
                p[1] = cvp.y;
                p[2] = 1.0;

                if (camIdBwdMap.find(view_idx) != camIdBwdMap.end() &&
                    pointIdBwdMap.find(pt_idx) != pointIdBwdMap.end()) {
                    scaleVectorIP(1.0 / f0, p);
                    measurements.push_back(Vector2d(p[0], p[1]));
                    correspondingView.push_back(camIdBwdMap[view_idx]);
                    correspondingPoint.push_back(pointIdBwdMap[pt_idx]);
                }
            }
        }
    }
    img_pts_sz = measurements.size();

    double thresh = 2.0 / abs(f0);
    Matrix3x3d K0 = cams[0].getIntrinsic();
    StdDistortionFunction distortion;
    bool good_adj = false;
    {
        ScopedBundleExtrinsicNormalizer ext_norm(cams, Xs);
        ScopedBundleIntrinsicNormalizer int_norm(cams, measurements, correspondingView);
        CommonInternalsMetricBundleOptimizer opt(FULL_BUNDLE_FOCAL_LENGTH_PP, thresh, K0, distortion, cams, Xs,
                                                 measurements, correspondingView, correspondingPoint);
        opt.tau = 1e-3;
        opt.maxIterations = 50;
        opt.minimize();

        good_adj = (opt.status != 2);
    }

    for (auto& cam : cams) {
        cam.setIntrinsic(K0);
    }

    Matrix3x3d K_new = K0;
    scaleMatrixIP(f0, K_new);
    K_new[2][2] = 1.0;

    if (good_adj) {
        for (size_t j = 0; j < Xs.size(); ++j) {
            cloud[j].first.x = Xs[j][0];
            cloud[j].first.y = Xs[j][1];
            cloud[j].first.z = Xs[j][2];
        }

        //extract adjusted cameras
        for (int i = 0; i < cam_vec_sz; ++i) {
            Matrix3x3d R = cams[i].getRotation();
            Vector3d T = cams[i].getTranslation();

            Matx34d P;
            P(0, 0) = R[0][0]; P(0, 1) = R[0][1]; P(0, 2) = R[0][2]; P(0, 3) = T[0];
            P(1, 0) = R[1][0]; P(1, 1) = R[1][1]; P(1, 2) = R[1][2]; P(1, 3) = T[1];
            P(2, 0) = R[2][0]; P(2, 1) = R[2][1]; P(2, 2) = R[2][2]; P(2, 3) = T[2];

            P_vec[i] = P;
        }
    }
}
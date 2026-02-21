#include "predict_model.h"

int predict_model(const std::array<double,8>& x) {
    double coeffs[5][8] = {
        {-1.30756100e-01,  2.26512483e-01, -5.59456176e-02,  2.22148535e-01, 2.04895376e+00,  1.11380600e+00,  2.24375124e-01,  4.89936270e-05},
        {-4.12282670e-01, -3.35759042e-01, -3.83939164e-01,  1.69300310e-01, 1.20174107e+00, -7.40071326e-01, -5.24076531e-02, -1.87195918e-04},
        { 5.98759468e-01, -2.87575656e-01,  5.60825854e-01,  1.85928988e-01, 1.23602224e-01, -7.44720954e-02, -2.95365287e-02,  7.75728993e-05},
        {-2.17520808e-01,  3.56221116e-01, -2.53404360e-02, -2.92536933e-01, -4.80518354e+00, -1.21525720e-02, -7.99843147e-02,  2.60833689e-05},
        {1.61800128e-01,  4.06010809e-02, -9.56006300e-02, -2.84840896e-01, 1.43088647e+00, -2.87110010e-01, -6.24464732e-02, -8.27048885e-05}
    };
    double biases[5] = {
        -277.77307081,
        164.22212499,
        -15.48946105,
        32.19425618,
        96.84615069
    };


    std::array<double,5> y;
    /*for (std::size_t i = 0; i < 5; ++i) {
        y[i] = coeffs[i][0]*x[0] + coeffs[i][1]*x[1] + coeffs[i][2]*x[2] + coeffs[i][3]*x[3] + coeffs[i][4]*x[4] + coeffs[i][5]*x[5] + coeffs[i][6]*x[6] + coeffs[i][7]*x[7] + biases[i];
    }*/
    y[0] = coeffs[0][0]*x[0] + coeffs[0][1]*x[1] + coeffs[0][2]*x[2] + coeffs[0][3]*x[3] + coeffs[0][4]*x[4] + coeffs[0][5]*x[5] + coeffs[0][6]*x[6] + coeffs[0][7]*x[7] + biases[0];
    y[1] = coeffs[1][0]*x[0] + coeffs[1][1]*x[1] + coeffs[1][2]*x[2] + coeffs[1][3]*x[3] + coeffs[1][4]*x[4] + coeffs[1][5]*x[5] + coeffs[1][6]*x[6] + coeffs[1][7]*x[7] + biases[1];
    y[2] = coeffs[2][0]*x[0] + coeffs[2][1]*x[1] + coeffs[2][2]*x[2] + coeffs[2][3]*x[3] + coeffs[2][4]*x[4] + coeffs[2][5]*x[5] + coeffs[2][6]*x[6] + coeffs[2][7]*x[7] + biases[2];
    y[3] = coeffs[3][0]*x[0] + coeffs[3][1]*x[1] + coeffs[3][2]*x[2] + coeffs[3][3]*x[3] + coeffs[3][4]*x[4] + coeffs[3][5]*x[5] + coeffs[3][6]*x[6] + coeffs[3][7]*x[7] + biases[3];
    y[4] = coeffs[4][0]*x[0] + coeffs[4][1]*x[1] + coeffs[4][2]*x[2] + coeffs[4][3]*x[3] + coeffs[4][4]*x[4] + coeffs[4][5]*x[5] + coeffs[4][6]*x[6] + coeffs[4][7]*x[7] + biases[4];

    
    for (std::size_t i = 0; i < 5; ++i) {
        #pragma HLS PIPELINE II=1
        y[i] = std::exp(y[i]);
    }

    double Z = y[0]+y[1]+y[2]+y[3]+y[4];

    for (std::size_t i = 0; i < 5; ++i) {
        #pragma HLS PIPELINE II=1
        y[i] /= Z;
    }

    auto it = std::max_element(y.begin(), y.end());
    auto idx = std::distance(y.begin(), it);
    return idx + 1;
}
#include "predict_model.h"

int predict_model(const std::array<double,5>& x) {
    double coeffs[5][5] = {
        {2.20051328e-01,  2.04890955e+00,  1.11016533e+00,  2.23618979e-01, 1.17199286e-04},
        {1.74333332e-01,  1.23152727e+00, -7.34954500e-01, -5.21766994e-02, -1.18635132e-04},
        { 1.81139182e-01,  1.30759400e-01, -7.64210852e-02, -2.95453966e-02, 1.45589518e-04},
        {-2.91914025e-01, -4.81770505e+00, -1.32671175e-02, -7.97117767e-02, 9.41345455e-05},
        {-2.83609800e-01,  1.40650893e+00, -2.85522588e-01, -6.21850304e-02, -1.42141364e-05}
    };
    double biases[5] = {
        -276.65764851,
         163.05908344,
         -15.43233559,
         32.56356667,
         96.46733399
    };


    std::array<double,5> y;
    
    y[0] = coeffs[0][0]*x[0] + coeffs[0][1]*x[1] + coeffs[0][2]*x[2] + coeffs[0][3]*x[3] + coeffs[0][4]*x[4] + biases[0];
    y[1] = coeffs[1][0]*x[0] + coeffs[1][1]*x[1] + coeffs[1][2]*x[2] + coeffs[1][3]*x[3] + coeffs[1][4]*x[4] + biases[1];
    y[2] = coeffs[2][0]*x[0] + coeffs[2][1]*x[1] + coeffs[2][2]*x[2] + coeffs[2][3]*x[3] + coeffs[2][4]*x[4] + biases[2];
    y[3] = coeffs[3][0]*x[0] + coeffs[3][1]*x[1] + coeffs[3][2]*x[2] + coeffs[3][3]*x[3] + coeffs[3][4]*x[4] + biases[3];
    y[4] = coeffs[4][0]*x[0] + coeffs[4][1]*x[1] + coeffs[4][2]*x[2] + coeffs[4][3]*x[3] + coeffs[4][4]*x[4] + biases[4];
    

    for (std::size_t i = 0; i < 5; ++i) {
        #pragma HLS PIPELINE II=1
        y[i] = std::exp(y[i]);
    }
    double Z = y[0] + y[1] +y[2] +y[3] +y[4];

    for (std::size_t i = 0; i < 5; ++i) {
        #pragma HLS PIPELINE II=1
        y[i] /= Z;
    }
    auto it = std::max_element(y.begin(), y.end());
    auto idx = std::distance(y.begin(), it);
    return idx+1;
}
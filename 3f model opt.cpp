#include "predict_model.h"


// Predict 5 outputs from 3 inputs.
int predict_model(const std::array<double,3>& x) {
    double coeffs[5][3] = {
        { 0.05098458,  0.39140812,  0.07987684},
        { 0.21256372, -0.16996399, -0.00882437},
        { 0.08786932, -0.13979567, -0.0182498 },
        {-0.17434198, -0.01781137, -0.02837844},
        {-0.17707563, -0.0638371 , -0.02442422}
    };
    double biases[5] = {
        -89.31607443,
         19.25098301,
         23.97523702,
         21.33804296,
         24.75181145
    };

    std::array<double,5> y;

    
   /* for (std::size_t i = 0; i < 5; ++i) {
        #pragma HLS PIPELINE II=1
        y[i] = coeffs[i][0]*x[0] + coeffs[i][1]*x[1] + coeffs[i][2]*x[2] + biases[i];
    }*/
    y[0] = coeffs[0][0]*x[0] + coeffs[0][1]*x[1] + coeffs[0][2]*x[2] + biases[0];
    y[1] = coeffs[1][0]*x[0] + coeffs[1][1]*x[1] + coeffs[1][2]*x[2] + biases[1];
    y[2] = coeffs[2][0]*x[0] + coeffs[2][1]*x[1] + coeffs[2][2]*x[2] + biases[2];
    y[3] = coeffs[3][0]*x[0] + coeffs[3][1]*x[1] + coeffs[3][2]*x[2] + biases[3];
    y[4] = coeffs[4][0]*x[0] + coeffs[4][1]*x[1] + coeffs[4][2]*x[2] + biases[4];


    for (std::size_t i = 0; i < 5; ++i) {
        #pragma HLS PIPELINE II=1
        y[i] = std::exp(y[i]);
        //cyclic dependence prevent further acceleretion
        //Z += y[i];
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
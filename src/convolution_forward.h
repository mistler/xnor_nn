#ifndef CONVOLUTION_FORWARD_H
#define CONVOLUTION_FORWARD_H

#include "xnor_nn_types.h"

xnor_nn_status_t reference_convolution_forward(
        const float *src, const float *weights, float *dst,
        float alpha, const float *k,
        int MB, int IC, int IH, int IW,
        int OC, int OH, int OW,
        int KH, int KW, int SH, int SW, int PH, int PW);

#endif

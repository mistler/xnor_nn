#ifndef BINARIZE_WEIGHTS_H
#define BINARIZE_WEIGHTS_H

#include "xnor_nn_types.h"

xnor_nn_status_t reference_weights_copy_on_float(const float *from, float *to,
        float *alpha, int OC, int IC, int KH, int KW);

xnor_nn_status_t direct_binarize_weights_char(const float *from,
        unsigned char *to, float *alpha,
        int OC, int IC, int KH, int KW);

#endif

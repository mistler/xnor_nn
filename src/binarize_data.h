#ifndef BINARIZE_DATA_H
#define BINARIZE_DATA_H

#include "xnor_nn_types.h"

xnor_nn_status_t reference_data_copy_on_float(
        const float *from, float *to,
        int MB, int IC, int IH, int IW);

xnor_nn_status_t direct_binarize_char(
        const unsigned int *from, unsigned char *to,
        int MB, int IC, int IH, int IW);

xnor_nn_status_t reference_calculate_k(const float *from, float *a, float *k,
        int MB, int IC, int IH, int IW, int OH, int OW,
        int KH, int KW, int SH, int SW, int PH, int PW);

#endif

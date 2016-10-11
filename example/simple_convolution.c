#include <stdlib.h>
#include <stdio.h>

#include "xnor_nn.h"

int main(void){
    // conv4 from AlexNet
    const int MB = 256;
    const int IC = 384, OC = 384;
    const int IH = 13, IW = 13;
    const int OH = 13, OW = 13;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    void *src_usr = NULL, *weights_usr = NULL;
    void *src_bin = NULL, *weights_bin = NULL;
    void *dst;

    xnor_nn_data_binarizer_t src_binarizer;
    xnor_nn_weights_binarizer_t weights_binarizer;
    xnor_nn_convolution_t convolution;

    xnor_nn_status_t st;
    char st_msg[16];

    // Usr data
    src_usr = malloc(sizeof(float)*MB*IC*IH*IW);
    weights_usr = malloc(sizeof(float)*OC*IC*KH*KW);
    dst = malloc(sizeof(float)*MB*OC*OH*OW);
    if (!src_usr || !weights_usr || !dst) {
        st = xnor_nn_error_memory;
        goto label;
    }

    // Setup
    st = xnor_nn_init_data_binarizer(&src_binarizer, MB, IC, IH, IW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_init_weights_binarizer(&weights_binarizer, OC, IC, KH, KW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_init_convolution(&convolution,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;

    // Internal data
    size_t sz_src_bin = src_binarizer.size(&src_binarizer);
    size_t sz_weights_bin = weights_binarizer.size(&weights_binarizer);

    st = xnor_nn_memory_allocate(&src_bin, sz_src_bin);
    if (st != xnor_nn_success) goto label;
    st = xnor_nn_memory_allocate(&weights_bin, sz_weights_bin);
    if (st != xnor_nn_success) goto label;

    // Execution
    st = weights_binarizer.execute(&weights_binarizer,
            weights_usr, weights_bin);
    if (st != xnor_nn_success) goto label;

    for (int i = 0; i < 3; i++) {
        st = src_binarizer.execute(&src_binarizer, src_usr, src_bin);
        if (st != xnor_nn_success) goto label;

        st = convolution.forward(&convolution, src_bin, weights_bin, dst);
        if (st != xnor_nn_success) goto label;
    }

label:
    free(src_usr);
    free(weights_usr);
    free(dst);

    xnor_nn_memory_free(src_bin);
    xnor_nn_memory_free(weights_bin);

    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
    return 0;
}

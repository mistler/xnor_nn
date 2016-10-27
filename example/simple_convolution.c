#include <stdlib.h>
#include <stdio.h>

#include "xnor_nn.h"

int main(void){
    const int MB = 8;
    const int IC = 64, OC = 128;
    const int IH = 13, IW = 13;
    const int OH = 13, OW = 13;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    xnor_nn_resources_t res = {0};

    xnor_nn_convolution_t convolution;

    xnor_nn_status_t st;
    char st_msg[16];

    // Usr data
    res[xnor_nn_resource_user_src] = malloc(sizeof(float)*MB*IC*IH*IW);
    res[xnor_nn_resource_user_weights] = malloc(sizeof(float)*OC*IC*KH*KW);
    res[xnor_nn_resource_user_dst] = malloc(sizeof(float)*MB*OC*OH*OW);
    if (!res[xnor_nn_resource_user_src] ||
            !res[xnor_nn_resource_user_weights] ||
            !res[xnor_nn_resource_user_dst]) {
        st = xnor_nn_error_memory;
        goto label;
    }

    // Setup
    st = xnor_nn_init_convolution(&convolution, xnor_nn_algorithm_reference,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_allocate_resources(&convolution, res);
    if (st != xnor_nn_success) goto label;

    st = convolution.binarize_weights(&convolution, res);
    if (st != xnor_nn_success) goto label;

    // Execute
    for (int i = 0; i < 3; i++) {
        st = convolution.binarize_data(&convolution, res);
        if (st != xnor_nn_success) goto label;

        st = convolution.calculate_k(&convolution, res);
        if (st != xnor_nn_success) goto label;

        st = convolution.forward(&convolution, res);
        if (st != xnor_nn_success) goto label;
    }

label:
    free(res[xnor_nn_resource_user_src]);
    free(res[xnor_nn_resource_user_weights]);
    free(res[xnor_nn_resource_user_dst]);

    xnor_nn_free_resources(res);

    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
    return st != xnor_nn_success;
}

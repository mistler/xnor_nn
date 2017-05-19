#include <vector>
#include <iostream>

#include "xnor_nn.h"
#include "timer.hpp"

typedef struct {
    xnor_nn_algorithm_t algorithm;
    int MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW;

    void print(std::ostream &stream) const {
        stream
            << "MB: " << MB
            << ", OC: " << OC
            << ", IC: " << IC
            << ", IH: " << IH
            << ", IW: " << IW
            << ", KH: " << KH
            << ", KW: " << KW
            << ", SH: " << SH
            << ", SW: " << SW
            << ", PH: " << PH
            << ", PW: " << PW
            << ", Alg: " << algorithm
            << std::endl;
    }
} convolution_params;


int main(){
    const int N = 1024 * 4 * 2;
    const xnor_nn_algorithm_t alg = xnor_nn_algorithm_bcast;

    // AlexNet conv3
    const convolution_params p{ alg, 1, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 };
    //const convolution_params p { alg, 1, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 };
    p.print(std::cout);

    const int enough = 256*1024*384; // 384mb on float

    float *dst = new float[enough];

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_resources_t res = {0};
    res[xnor_nn_resource_user_dst] = dst;

    xnor_nn_convolution_t convolution;

    st = xnor_nn_init_convolution(&convolution, p.algorithm,
        xnor_nn_data_format_nchw,
        xnor_nn_weights_format_oihw,
        xnor_nn_data_format_nchw,
        p.MB, p.OC, p.IC, p.IH, p.IW, p.KH, p.KW, p.SH, p.SW, p.PH, p.PW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_allocate_resources(&convolution, res);
    if (st != xnor_nn_success) goto label;

    // Execute
    for (int n = 0; n < N; n++) {
        st = convolution.forward(&convolution, res);
        if (st != xnor_nn_success) goto label;
    }

    // Clean up
    xnor_nn_free_resources(res);

label:
    delete[] dst;

    xnor_nn_destroy_convolution(&convolution);

    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
    return st != xnor_nn_success;
}

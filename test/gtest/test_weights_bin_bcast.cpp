#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

TEST(WeightsBinarizeBcast, bcast_precalculated) {
    const int MB = 1;
    const int IC = 2, OC = 8;
    const int IH = 2, IW = 2;
    const int KH = 2, KW = 2;
    const int SH = 1, SW = 1;
    const int PH = 0, PW = 0;

    const float P = 1.f;
    const float N = -1.f;

    // Usr weights (just random)
    const float weights[] = {
        P, P, P, P,
        N, N, N, N,

        P, P, P, P,
        P, P, P, P,

        N, N, N, N,
        N, N, N, N,

        N, N, N, N,
        P, P, P, P,


        P, P, P, P,
        N, N, N, N,

        P, P, P, P,
        P, P, P, P,

        N, N, N, N,
        N, N, N, N,

        N, N, N, N,
        P, P, P, P,
    };

    // Precalculated alpha
    const float alpha = P;

    // Binarizer setup
    xnor_nn_resources_t res = {0};
    float *actual_alpha = (float*)(&res[xnor_nn_resource_alpha]);

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_convolution_t convolution;

    st = xnor_nn_init_convolution(&convolution, xnor_nn_algorithm_bcast,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_allocate_resources(&convolution, res);
    if (st != xnor_nn_success) goto label;

    res[xnor_nn_resource_user_weights] = (void*)weights;

    // Execution
    st = convolution.binarize_weights(&convolution, res);
    if (st != xnor_nn_success) goto label;

    // Chech result
    xnor_nn::test::check_weights_bcast(OC, IC, KH, KW,
            (unsigned char*)res[xnor_nn_resource_bin_weights], weights);

    xnor_nn::test::check_value(*actual_alpha, alpha);

label:
    xnor_nn_free_resources(res);

    xnor_nn_destroy_convolution(&convolution);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}

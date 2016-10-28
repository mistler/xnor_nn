#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

TEST(WeightsBinarizationDirectChar, optimized_precalculated) {
    const int MB = 1;
    const int IC = 2, OC = 2;
    const int IH = 3, IW = 3;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    const float P = 1.f;
    const float N = -1.f;

    // Usr weights (just random)
    const float weights[] = {
        P, P, P,
        N, N, P,
        P, N, N,

        P, P, P,
        P, P, P,
        P, P, P,

        N, N, N,
        N, N, N,
        N, N, N,

        P, P, P,
        N, N, P,
        P, N, N
    };

    // Precalculated weights
    const unsigned char expected_weights_bin[] = {
        0xC0u, 0x40u,
        0xC0u, 0x40u,
        0xC0u, 0x40u,

        0x40u, 0x00u,
        0x40u, 0x00u,
        0xC0u, 0x40u,

        0xC0u, 0x40u,
        0x40u, 0x00u,
        0x40u, 0x00u,
    };

    // Precalculated alpha
    const float alpha = P;

    // Binarizer setup
    xnor_nn_resources_t res = {0};
    float *actual_alpha = (float*)(&res[xnor_nn_resource_alpha]);

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_convolution_t convolution;

    st = xnor_nn_init_convolution(&convolution, xnor_nn_algorithm_optimized,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_allocate_resources(&convolution, res);
    if (st != xnor_nn_success) goto label;

    res[xnor_nn_resource_user_weights] = (void*)weights;

    // Execution
    st = convolution.binarize_weights(&convolution, res);
    if (st != xnor_nn_success) goto label;

    // Check result
    xnor_nn::test::check_weights(OC, (IC + 8 - 1) / 8, KH, KW,
            (unsigned char*)res[xnor_nn_resource_bin_weights],
            expected_weights_bin);
    xnor_nn::test::check_weights(OC, IC, KH, KW,
            (unsigned char*)res[xnor_nn_resource_bin_weights], weights);

    xnor_nn::test::check_value(*actual_alpha, alpha);

label:
    xnor_nn_free_resources(res);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}

#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

TEST(DataBinarizeBcast, bcast_small) {
    const int MB = 1;
    const int IC = 2, OC = 8;
    const int IH = 3, IW = 3;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    const float P = 1.f;
    const float N = -1.f;

    // Usr data (just random)
    const float src[] = {
        P, P, P,
        N, P, N,
        N, N, N,

        N, N, N,
        N, N, N,
        P, P, P
    };

    // Binarizer setup
    xnor_nn_resources_t res = {0};

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_convolution_t convolution;

    st = xnor_nn_init_convolution(&convolution, xnor_nn_algorithm_bcast,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW);
    if (st != xnor_nn_success) goto label;

    st = xnor_nn_allocate_resources(&convolution, res);
    if (st != xnor_nn_success) goto label;

    res[xnor_nn_resource_user_src] = (void*)src;

    // Execution
    st = convolution.binarize_data(&convolution, res);
    if (st != xnor_nn_success) goto label;

    st = convolution.calculate_k(&convolution, res);
    if (st != xnor_nn_success) goto label;

    // Check result
    xnor_nn::test::check_data(MB, IC, IH, IW,
            ((IC + 4 - 1) / 4 + 8 - 1) / 8 * 4 * 8,
            (unsigned char*)res[xnor_nn_resource_bin_src], src);

label:
    xnor_nn_free_resources(res);

    xnor_nn_destroy_convolution(&convolution);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}

#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

TEST(DataWeightsFloatToFloat, simple_precalculated) {
    const int OC = 2, IC = 2, KH = 3, KW = 3;

    const float P = 1.f;
    const float N = -1.f;

    // Usr weights (just random)
    const float weights[OC*IC*KH*KW] = {
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
    // Precalculated src
    const float expected_weights_bin[OC*IC*KH*KW] = {
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

    // Binarizer setup
    xnor_nn_status_t st;
    char st_msg[16];

    size_t sz_weights_bin;
    void *actual_weights_bin = NULL;

    xnor_nn_weights_binarizer_t weights_binarizer;

    st = xnor_nn_init_weights_binarizer(&weights_binarizer, OC, IC, KH, KW);
    if (st != xnor_nn_success) goto label;

    sz_weights_bin = weights_binarizer.size(&weights_binarizer);

    st = xnor_nn_memory_allocate(&actual_weights_bin, sz_weights_bin);
    if (st != xnor_nn_success) goto label;

    // Execution
    st = weights_binarizer.execute(&weights_binarizer,
            weights, actual_weights_bin);
    if (st != xnor_nn_success) goto label;

    // Check result
    xnor_nn::test::check_weights(OC, IC, KH, KW,
            (float*)actual_weights_bin, expected_weights_bin);

label:
    xnor_nn_memory_free(actual_weights_bin);

    EXPECT_EQ(st, xnor_nn_success);
    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
}

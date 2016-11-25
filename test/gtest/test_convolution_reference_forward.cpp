#include "xnor_nn.hpp"

#include "gtest.h"
#include "common.hpp"

TEST(ConvolutionForwardReference, reference_precalculated) {
    const int MB = 1;
    const int IC = 2, OC = 2;
    const int IH = 3, IW = 3;
    const int OH = 3, OW = 3;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    const float P = 1.f;
    const float N = -1.f;

    // Usr data (just random)
    const float src[MB*IC*IH*IW] = {
        P, P, P,
        N, P, N,
        N, N, N,

        N, N, N,
        N, N, N,
        P, P, P
    };
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
    // Precalculated output
    float expected_dst[MB*OC*OH*OW] = {
        8, 12, 8,
        48, 81, 36,
        16, 36, 20,

        16, 36, 16,
        24, 72, 36,
        16, 36, 12
    }; // * 1/9
    for (int i = 0; i < MB*OC*OH*OW; i++) expected_dst[i] /= 9.f;

    float actual_dst[MB*OC*OH*OW] = { 0.f };

    // Convolution setup
    xnor_nn::Convolution convolution{xnor_nn_algorithm_reference,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, weights};

    // Execution
    convolution.forward(src, actual_dst);

    // Check result
    xnor_nn::test::check_4d(MB, OC, OH, OW, actual_dst, expected_dst);
}

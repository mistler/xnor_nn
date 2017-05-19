#include "gtest.h"

#include "common.hpp"

#include "xnor_nn.hpp"

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
    const float src_nchw[MB*IC*IH*IW] = {
        P, P, P,
        N, P, N,
        N, N, N,

        N, N, N,
        N, N, N,
        P, P, P
    };
    const float src_nhwc[MB*IH*IW*IC] = {
        P,N, P,N, P,N,
        N,N, P,N, N,N,
        N,P, N,P, N,P,
    };

    const float weights_oihw[OC*IC*KH*KW] = {
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
    const float weights_hwio[KH*KW*IC*OC] = {
        P,N,P,P, P,N,P,P, P,N,P,P,
        N,N,P,N, N,N,P,N, P,N,P,P,
        P,N,P,P, N,N,P,N, N,N,P,N,
    };
    // Precalculated output
    float expected_dst_nchw[MB*OC*OH*OW] = {
        -16, -48, -16,
        24, 0, 0,
        0, 0, 8,

        0, 0, 0,
        -24, -18, 0,
        0, 0, -8
    }; // * 1/9
    float expected_dst_nhwc[MB*OH*OW*OC] = {
        -16,0, -48,0, -16,0,
        24,-24, 0,-18, 0,0,
        0,0, 0,0, 8,-8,
    }; // * 1/9
    for (int i = 0; i < MB*OC*OH*OW; i++) expected_dst_nchw[i] /= 9.f;
    for (int i = 0; i < MB*OH*OW*OC; i++) expected_dst_nhwc[i] /= 9.f;

    float actual_dst_caffe_nchw[MB*OC*OH*OW] = { 0.f };
    float actual_dst_caffe_nhwc[MB*OH*OW*OC] = { 0.f };
    float actual_dst_tf_nchw[MB*OC*OH*OW] = { 0.f };
    float actual_dst_tf_nhwc[MB*OH*OW*OC] = { 0.f };

    // Convolution setup
    xnor_nn::Convolution convolution_caffe_nchw{xnor_nn_algorithm_reference,
            xnor_nn_data_format_nchw,
            xnor_nn_weights_format_oihw,
            xnor_nn_data_format_nchw,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, weights_oihw};
    xnor_nn::Convolution convolution_caffe_nhwc{xnor_nn_algorithm_reference,
            xnor_nn_data_format_nchw,
            xnor_nn_weights_format_oihw,
            xnor_nn_data_format_nhwc,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, weights_oihw};
    xnor_nn::Convolution convolution_tf_nchw{xnor_nn_algorithm_reference,
            xnor_nn_data_format_nhwc,
            xnor_nn_weights_format_hwio,
            xnor_nn_data_format_nchw,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, weights_hwio};
    xnor_nn::Convolution convolution_tf_nhwc{xnor_nn_algorithm_reference,
            xnor_nn_data_format_nhwc,
            xnor_nn_weights_format_hwio,
            xnor_nn_data_format_nhwc,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, weights_hwio};

    // Execution
    convolution_caffe_nchw.forward(src_nchw, actual_dst_caffe_nchw);
    convolution_caffe_nhwc.forward(src_nchw, actual_dst_caffe_nhwc);
    convolution_tf_nchw.forward(src_nhwc, actual_dst_tf_nchw);
    convolution_tf_nhwc.forward(src_nhwc, actual_dst_tf_nhwc);

    // Check result
    xnor_nn::test::check_4d(MB, OC, OH, OW,
            actual_dst_caffe_nchw, expected_dst_nchw);
    xnor_nn::test::check_4d(MB, OH, OW, OC,
            actual_dst_caffe_nhwc, expected_dst_nhwc);
    xnor_nn::test::check_4d(MB, OC, OH, OW,
            actual_dst_tf_nchw, expected_dst_nchw);
    xnor_nn::test::check_4d(MB, OH, OW, OC,
            actual_dst_tf_nhwc, expected_dst_nhwc);
}

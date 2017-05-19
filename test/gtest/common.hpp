#ifndef COMMON_HPP
#define COMMON_HPP

#include <limits>
#include <cstdint>

#include "gtest.h"

#include "xnor_nn.h"
#include "utils.hpp"

namespace xnor_nn {
namespace test {

typedef struct {
    xnor_nn_tensor_format_t src;
    xnor_nn_tensor_format_t weights;
    xnor_nn_tensor_format_t dst;
} tensor_fmt_t;

typedef struct {
    xnor_nn_algorithm_t algorithm;
    tensor_fmt_t fmt;
    int mb;
    int ic, oc;
    int ih, iw;
    int kh, kw;
    int sh, sw;
    int ph, pw;
    int oh, ow;
} params_t;

int getBICI(const params_t &p) {
#if 1
    return p.ic * p.kh * p.kw > std::numeric_limits<int16_t>::max()
        ? sizeof(int32_t) : sizeof(int16_t);
#else
    (void)p;
    return sizeof(int32_t);
#endif
}

template <typename T>
void check_4d(int S3, int S2, int S1, int S0,
        const T *actual, const T *expected);

template <typename A, typename E>
void check_arrays(int elems, const A *actual, const E *expected);

template <typename A, typename E>
void check_value(const A &actual, const E &expected);

template<> void check_4d<unsigned char>(int S3, int S2, int S1, int S0,
        const unsigned char *a, const unsigned char *e) {
    int wrong = 0;
    for (int s3 = 0; s3 < S3; s3++)
    for (int s2 = 0; s2 < S2; s2++)
    for (int s1 = 0; s1 < S1; s1++)
    for (int s0 = 0; s0 < S0; s0++) {
        unsigned char actual = a[((s3*S2 + s2)*S1 + s1)*S0 + s0];
        unsigned char expected = e[((s3*S2 + s2)*S1 + s1)*S0 + s0];
        EXPECT_EQ(expected, actual) << "s3: " << s3 << ", s2: "
            << s2 << ", s1: " << s1 << ", s0: " << s0 << ". wrong/total: "
            << ++wrong << "/" << S3*S2*S1*S0;
    }
}

void check_data_bcast(xnor_nn_tensor_format_t sfmt,
        int MB, int IC, int IH, int IW, int BICI,
        const unsigned char *a, const float *e) {
    constexpr int SZ = 8;

    const int BIC = xnor_nn::utils::div_up(IC, SZ);
    const int ABIC = xnor_nn::utils::div_up(BIC, BICI) * BICI;

    int wrong = 0;
    for (int mb = 0; mb < MB; mb++)
    for (int h = 0; h < IH; h++)
    for (int w = 0; w < IW; w++)
    for (int c = 0; c < IC; c++) {
        int src_idx = -1;
        switch (sfmt) {
        case xnor_nn_data_format_nchw:
            src_idx = ((mb*IC + c)*IH + h)*IW + w; break;
        case xnor_nn_data_format_nhwc:
            src_idx = ((mb*IH + h)*IW + w)*IC + c; break;
        default: break;
        }

        bool expected = !(bool) (((unsigned int*)e)[src_idx] >> 31);
        bool actual = a[((mb*IH + h)*IW + w)*ABIC + (c / SZ)]
            & (1 << (SZ - 1 - (c % SZ)));

        EXPECT_EQ(expected, actual) << "mb: " << mb << ", c: "
            << c << ", h: " << h << ", w: " << w << ". wrong/total: "
            << ++wrong << "/" << MB*IC*IH*IW;
    }
}

// TODO: check space, filled with ONES
void check_weights_bcast(xnor_nn_tensor_format_t wfmt,
        int OC, int IC, int KH, int KW, int BICI, int VLEN,
        const unsigned char *a, const float *e) {
    constexpr int SZ = 8;

    constexpr int ELEM_SIZE = sizeof(char);
    constexpr int BITS = ELEM_SIZE * SZ;

    const int VLEN_BYTES = (VLEN / SZ);
    const int OCI = VLEN_BYTES / BICI;

    const int BIC = (IC + BITS - 1) / BITS;
    const int ICO = (BIC + BICI - 1) / BICI;

    int wrong = 0;
    for (int oc = 0; oc < OC; oc++)
    for (int ic = 0; ic < IC; ic++)
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++) {
        const int ici = (ic / SZ) % BICI;
        const int ico = (ic / SZ) / BICI;
        const int oci = oc % OCI;
        const int oco = oc / OCI;

        const int a_idx =
            ((((oco*KH + kh)*KW + kw)*ICO + ico)*OCI + oci)*BICI + ici;

        int weights_idx = -1;
        switch (wfmt) {
        case xnor_nn_weights_format_oihw:
            weights_idx = ((oc*IC + ic)*KH + kh)*KW + kw; break;
        case xnor_nn_weights_format_hwio:
            weights_idx = ((kh*KW + kw)*IC + ic)*OC + oc; break;
        default: break;
        }

        bool expected = !(bool)(((unsigned int*)e)[weights_idx] >> 31);
        bool actual = a[a_idx] & ((unsigned char)1) << (SZ - 1 - (ic % SZ));

        EXPECT_EQ(expected, actual) << "oc: " << oc << ", ic: "
            << ic << ", kh: " << kh << ", kw: " << kw << ". wrong/total: "
            << ++wrong << "/" << OC*IC*KH*KW;
    }
}

template<> void check_4d<float>(int S3, int S2, int S1, int S0,
        const float *a, const float *e) {
    const float ERR = 1e-3f;

    int wrong = 0;
    for (int s3 = 0; s3 < S3; s3++)
    for (int s2 = 0; s2 < S2; s2++)
    for (int s1 = 0; s1 < S1; s1++)
    for (int s0 = 0; s0 < S0; s0++) {
        float actual = a[((s3*S2 + s2)*S1 + s1)*S0 + s0];
        float expected = e[((s3*S2 + s2)*S1 + s1)*S0 + s0];
        EXPECT_NEAR(expected, actual, ERR) << "s3: " << s3 << ", s2: "
            << s2 << ", s1: " << s1 << ", s0: " << s0 << ". wrong/total: "
            << ++wrong << "/" << S3*S2*S1*S0;
    }
}

template<> void check_arrays<float, float>(int elems,
        const float *actual, const float *expected) {
    const float ERR = 1e-5f;
    for (int i = 0; i < elems; i++) {
        EXPECT_NEAR(expected[i], actual[i], ERR) << "i: " << i << ", elems: "
            << elems;
    }
}

template<> void check_value<float, float>(const float &actual,
        const float &expected) {
    const float ERR = 1e-5f;
    EXPECT_NEAR(expected, actual, ERR);
}

}
}

#endif

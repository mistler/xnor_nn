#ifndef COMMON_HPP
#define COMMON_HPP

#include "xnor_nn.h"

#include "gtest.h"

namespace xnor_nn {
namespace test {

template <typename T>
void check_data(int MB, int C, int H, int W,
        const T *actual, const T *expected);

template <typename T>
void check_weights(int OC, int IC, int KH, int KW,
        const T *actual, const T *expected);

template <typename T>
void check_arrays(int elems, const T *actual, const T *expected);

template <typename T>
void check_value(const T &actual, const T &expected);


template<> void check_data<float>(int MB, int C, int H, int W,
        const float *a, const float *e) {
    const float ERR = 1e-5f;
    int wrong = 0;
    for (int mb = 0; mb < MB; mb++)
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float actual = a[((mb*C + c)*H + h)*W + w];
        float expected = e[((mb*C + c)*H + h)*W + w];
        EXPECT_NEAR(expected, actual, ERR) << "mb: " << mb << ", c: "
            << c << ", h: " << h << ", w: " << w << ". wrong/total: "
            << ++wrong << "/" << MB*C*H*W;
    }
}

template<> void check_data<unsigned char>(int MB, int C, int H, int W,
        const unsigned char *a, const unsigned char *e) {
    int wrong = 0;
    for (int mb = 0; mb < MB; mb++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
    for (int c = 0; c < C; c++) {
        unsigned char actual = a[((mb*H + h)*W + w)*C + c];
        unsigned char expected = e[((mb*H + h)*W + w)*C + c];
        EXPECT_EQ(expected, actual) << "mb: " << mb << ", c: "
            << c << ", h: " << h << ", w: " << w << ". wrong/total: "
            << ++wrong << "/" << MB*C*H*W;
    }
}

template<> void check_weights<float>(int OC, int IC, int KH, int KW,
        const float *a, const float *e) {
    const float ERR = 1e-5f;
    int wrong = 0;
    for (int oc = 0; oc < OC; oc++)
    for (int ic = 0; ic < IC; ic++)
    for (int kh = 0; kh < KH; kh++)
    for (int kw = 0; kw < KW; kw++) {
        float actual = a[((oc*IC + ic)*KH + kh)*KW + kw];
        float expected = e[((oc*IC + ic)*KH + kh)*KW + kw];
        EXPECT_NEAR(expected, actual, ERR) << "oc: " << oc << ", ic: "
            << ic << ", kh: " << kh << ", kw: " << kw << ". wrong/total: "
            << ++wrong << "/" << OC*IC*KH*KW;
    }
}

template<> void check_arrays<float>(int elems,
        const float *actual, const float *expected) {
    const float ERR = 1e-5f;
    for (int i = 0; i < elems; i++) {
        EXPECT_NEAR(expected[i], actual[i], ERR) << "i: " << i << ", elems: "
            << elems;
    }
}

template<> void check_value<float>(const float &actual, const float &expected) {
    const float ERR = 1e-5f;
    EXPECT_NEAR(expected, actual, ERR);
}

}
}

#endif

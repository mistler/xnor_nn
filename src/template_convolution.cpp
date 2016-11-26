#include "template_convolution.hpp"

#include "utils.h"

#define CHECK(IC, IH, IW, KH, KW, SH, SW, PH, PW) \
    if (IC == c->ic && IH == c->ih && IW == c->iw && KH == c->kh \
            && KW == c->kw && SH == c->sh && SW == c->sw && PH == c->ph \
            && PW == c->pw) \
        return true

#define USE(IC, IH, IW, KH, KW, SH, SW, PH, PW) \
    if (IC == c->ic && IH == c->ih && IW == c->iw && KH == c->kh \
            && KW == c->kw && SH == c->sh && SW == c->sw && PH == c->ph \
            && PW == c->pw) \
        c->forward = exec<IC, IH, IW, KH, KW, SH, SW, PH, PW>

#define C1 CHECK(64, 27, 27, 5, 5, 1, 1, 2, 2)
#define C2 CHECK(192, 13, 13, 3, 3, 1, 1, 1, 1)
#define C3 CHECK(384, 13, 13, 3, 3, 1, 1, 1, 1)
#define C4 CHECK(256, 13, 13, 3, 3, 1, 1, 1, 1)

#define U1 USE(64, 27, 27, 5, 5, 1, 1, 2, 2)
#define U2 USE(192, 13, 13, 3, 3, 1, 1, 1, 1)
#define U3 USE(384, 13, 13, 3, 3, 1, 1, 1, 1)
#define U4 USE(256, 13, 13, 3, 3, 1, 1, 1, 1)

#ifdef __x86_64__
#include <immintrin.h>

namespace {

template<int IC, int IH, int IW, int KH, int KW,
    int SH, int SW, int PH, int PW>
xnor_nn_status_t exec(const xnor_nn_convolution_t *c,
        xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
        || c == nullptr
    ) return xnor_nn_error_invalid_input;
    const unsigned int *src = (unsigned int*)res[xnor_nn_resource_bin_src];
    const unsigned int *weights =
        (unsigned int*)res[xnor_nn_resource_bin_weights];
    float *dst = (float*)res[xnor_nn_resource_user_dst];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];
    const float *k = (float*)res[xnor_nn_resource_k];

    constexpr int OH = (IH + 2*PH - KH) / SH + 1;
    constexpr int OW = (IW + 2*PW - KW) / SW + 1;
    constexpr int BIC = (IC + 8 - 1) / 8;
    constexpr int AIC = ((BIC + VLEN - 1) / VLEN) * VLEN;
    const int MB = c->mb;
    const int OC = c->oc;

    const int ELEM_SIZE = 4;
    const int VECTORS_IN_AIC = AIC / VLEN;

    const int ones[8] = {-1,-1,-1,-1,-1,-1,-1,-1};

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        unsigned long long int dst_i = 0;
        __m256 v_ones = _mm256_loadu_ps((float*)ones);
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            if (oh*SH + kh < (PH > 0 ? PH : 0)) continue;
            if (ow*SW + kw < (PW > 0 ? PW : 0)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            const unsigned int *src_ic =
                src + ((mb*IH + ih)*IW + iw)*AIC/ELEM_SIZE;
            const unsigned int *weights_ic =
                weights + ((kh*KW + kw)*OC + oc)*AIC/ELEM_SIZE;

            for (int aic = 0; aic < VECTORS_IN_AIC; aic++) {
                __m256 v_src = _mm256_load_ps(
                        (float*)src_ic + aic*VLEN/ELEM_SIZE);
                __m256 v_weights = _mm256_load_ps(
                        (float*)weights_ic + aic*VLEN/ELEM_SIZE);

                __m256 v_xor = _mm256_xor_ps(v_src, v_weights);
                __m256i v_xnor =
                    _mm256_castps_si256(_mm256_xor_ps(v_xor, v_ones));

                dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 0));
                dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 1));
                dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 2));
                dst_i += __builtin_popcountll(_mm256_extract_epi64(v_xnor, 3));
            }
        }
        *d = (float)dst_i * *alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

} // namespace

#elif defined __arm__

#include <arm_neon.h>

namespace {

template<int IC, int IH, int IW, int KH, int KW,
    int SH, int SW, int PH, int PW>
xnor_nn_status_t exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
    ) return xnor_nn_error_invalid_input;
    const unsigned int *src = (unsigned int*)res[xnor_nn_resource_bin_src];
    const unsigned int *weights =
        (unsigned int*)res[xnor_nn_resource_bin_weights];
    float *dst = (float*)res[xnor_nn_resource_user_dst];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];
    const float *k = (float*)res[xnor_nn_resource_k];

    constexpr int OH = (IH + 2*PH - KH) / SH + 1;
    constexpr int OW = (IW + 2*PW - KW) / SW + 1;
    constexpr int BIC = (IC + 8 - 1) / 8;
    constexpr int AIC = ((BIC + VLEN - 1) / VLEN) * VLEN;
    const int MB = c->mb;
    const int OC = c->oc;

    const int ELEM_SIZE = 4;
    const int VECTORS_IN_AIC = AIC / VLEN;

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        long long int dst_i = 0;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            if (oh*SH + kh < (PH > 0 ? PH : 0)) continue;
            if (ow*SW + kw < (PW > 0 ? PW : 0)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            const unsigned int *src_ic =
                src + ((mb*IH + ih)*IW + iw)*AIC/ELEM_SIZE;
            const unsigned int *weights_ic =
                weights + ((kh*KW + kw)*OC + oc)*AIC/ELEM_SIZE;

            for (int aic = 0; aic < VECTORS_IN_AIC; aic++) {
                uint32x4_t v_src = vld1q_u32(src_ic + aic*VLEN/ELEM_SIZE);
                uint32x4_t v_weights =
                    vld1q_u32(weights_ic + aic*VLEN/ELEM_SIZE);

                uint32x4_t v_xor = veorq_u32(v_src, v_weights);
                uint32x4_t v_xnor = vmvnq_u32(v_xor);

                uint8x16_t v_cnt16 = vcntq_u8(vreinterpretq_u8_u32(v_xnor));
                uint16x8_t v_cnt8 = vpaddlq_u8(v_cnt16);
                uint32x4_t v_cnt4 = vpaddlq_u16(v_cnt8);
                uint64x2_t v_cnt2 = vpaddlq_u32(v_cnt4);
                dst_i += vgetq_lane_u64(v_cnt2, 0);
                dst_i += vgetq_lane_u64(v_cnt2, 1);
            }
        }
        *d = (float)dst_i * *alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

} // namespace

#else

namespace {

template<int IC, int IH, int IW, int KH, int KW,
    int SH, int SW, int PH, int PW>
xnor_nn_status_t exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
    if (
        res[xnor_nn_resource_bin_src] == nullptr
        || res[xnor_nn_resource_bin_weights] == nullptr
        || res[xnor_nn_resource_user_dst] == nullptr
        || res[xnor_nn_resource_k] == nullptr
    ) return xnor_nn_error_invalid_input;
    const unsigned int *src = (unsigned int*)res[xnor_nn_resource_bin_src];
    const unsigned int *weights =
        (unsigned int*)res[xnor_nn_resource_bin_weights];
    float *dst = (float*)res[xnor_nn_resource_user_dst];
    float *alpha = (float*)&res[xnor_nn_resource_alpha];
    const float *k = (float*)res[xnor_nn_resource_k];

    constexpr int OH = (IH + 2*PH - KH) / SH + 1;
    constexpr int OW = (IW + 2*PW - KW) / SW + 1;
    constexpr int BIC = (IC + 8 - 1) / 8;
    constexpr int AIC = ((BIC + VLEN - 1) / VLEN) * VLEN;
    const int MB = c->mb;
    const int OC = c->oc;

    const int ELEM_SIZE = 4;
    const int VECTORS_IN_AIC = AIC / VLEN;

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            if (oh*SH + kh < (PH > 0 ? PH : 0)) continue;
            if (ow*SW + kw < (PW > 0 ? PW : 0)) continue;

            if (oh*SH + kh >= IH + PH) continue;
            if (ow*SW + kw >= IW + PW) continue;

            const int ih = oh * SH - PH + kh;
            const int iw = ow * SW - PW + kw;

            const unsigned int *src_ic =
                src + ((mb*IH + ih)*IW + iw)*AIC/ELEM_SIZE;
            const unsigned int *weights_ic =
                weights + ((kh*KW + kw)*OC + oc)*AIC/ELEM_SIZE;

            for (int aic = 0; aic < VECTORS_IN_AIC; aic++)
            for (int v = 0; v < VLEN / ELEM_SIZE; v++) {
                int src_idx = aic*VLEN/ELEM_SIZE + v;
                int weights_idx = aic*VLEN/ELEM_SIZE + v;

                unsigned int bsrc = src_ic[src_idx];
                unsigned int bweights = weights_ic[weights_idx];

                unsigned int result = ~(bsrc ^ bweights);
                *d += __builtin_popcount(result);
            }
        }
        *d *= *alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

} // namespace

#endif

namespace xnor_nn {
namespace implementation {

bool TemplateConvolution::isApplicable(
        const xnor_nn_convolution_t *c) const {
    if (c->forward != nullptr) return false;
    if (c->algorithm != xnor_nn_algorithm_template) return false;

    C1;
    C2;
    C3;
    C4;

    return false;
}

void TemplateConvolution::setupConvolution(
        xnor_nn_convolution_t *c) {
    TemplateConvolution *op = new TemplateConvolution;

    const size_t ELEM_SIZE = sizeof(char);
    const size_t BITS = ELEM_SIZE * 8;
    const size_t VEC_LENGTH = VLEN;
    const size_t BIC = (c->ic + BITS - 1) / BITS;

    c->sizeof_element = ELEM_SIZE;
    c->vector_length = VEC_LENGTH;
    c->bic = BIC;
    c->aic = ((BIC + VEC_LENGTH - 1) / VEC_LENGTH) * VEC_LENGTH;

    c->resource_size[xnor_nn_resource_bin_src] =
        c->mb * c->aic * c->ih * c->iw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_bin_weights] =
        c->oc * c->aic * c->kh * c->kw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
    c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);

    U1;
    U2;
    U3;
    U4;

    std::vector<Implementation*> *vec =
        (std::vector<Implementation*>*)c->state;
    vec->push_back(op);
}

TemplateConvolution::~TemplateConvolution() {}

}
}

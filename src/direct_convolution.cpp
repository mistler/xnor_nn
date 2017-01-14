#include "direct_convolution.hpp"

#include "utils.h"

// TODO: log execution

namespace xnor_nn {
namespace implementation {

bool DirectConvolution::isApplicable(
        const xnor_nn_convolution_t *c) const {
    if (c->forward != nullptr) return false;
    if (c->algorithm != xnor_nn_algorithm_direct) return false;
    return true;
}

void DirectConvolution::setupConvolution(
        xnor_nn_convolution_t *c) {
    DirectConvolution *op = new DirectConvolution;

    const int ELEM_SIZE = sizeof(char);
    const int BITS = ELEM_SIZE * 8;
    const int BIC = ((c->ic + BITS - 1) / BITS) * BITS;

    c->bic = BIC;
    c->abic = ((BIC + VLEN - 1) / VLEN) * VLEN;

    c->resource_size[xnor_nn_resource_bin_src] =
        c->mb * c->abic * c->ih * c->iw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_bin_weights] =
        c->oc * c->abic * c->kh * c->kw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
    c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);

    c->forward = op->exec;

    std::vector<Implementation*> *vec =
        (std::vector<Implementation*>*)c->state;
    vec->push_back(op);
}

DirectConvolution::~DirectConvolution() {}

} // namespace implementation
} // namespace xnor_nn

#ifdef ARCH_X86
#include <immintrin.h>

namespace xnor_nn {
namespace implementation {

xnor_nn_status_t DirectConvolution::exec(const xnor_nn_convolution_t *c,
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

    const int MB = c->mb;
    const int IH = c->ih;
    const int IW = c->iw;
    const int OC = c->oc;
    const int OH = c->oh;
    const int OW = c->ow;
    const int KH = c->kh;
    const int KW = c->kw;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

    const int ABIC = c->abic;
    const int ELEM_SIZE = 32;

    const int VECTORS_IN_ABIC = ABIC / VLEN;

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
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const unsigned int *src_ic =
                src + ((mb*IH + ih)*IW + iw)*ABIC/ELEM_SIZE;
            const unsigned int *weights_ic =
                weights + ((kh*KW + kw)*OC + oc)*ABIC/ELEM_SIZE;

            for (int vabic = 0; vabic < VECTORS_IN_ABIC; vabic++) {
                __m256 v_src = _mm256_load_ps(
                        (float*)src_ic + vabic*VLEN/ELEM_SIZE);
                __m256 v_weights = _mm256_load_ps(
                        (float*)weights_ic + vabic*VLEN/ELEM_SIZE);

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

} // namespace implementation
} // namespace xnor_nn

#elif defined ARCH_ARM

#include <arm_neon.h>

namespace xnor_nn {
namespace implementation {

xnor_nn_status_t DirectConvolution::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
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

    const int MB = c->mb;
    const int IH = c->ih;
    const int IW = c->iw;
    const int OC = c->oc;
    const int OH = c->oh;
    const int OW = c->ow;
    const int KH = c->kh;
    const int KW = c->kw;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

    const int ABIC = c->abic;
    const int ELEM_SIZE = 32;

    const int VECTORS_IN_ABIC = ABIC / VLEN;

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        uint32x4_t v_accum = veorq_u32(v_accum, v_accum);
        long long int dst_i = 0;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const unsigned int *src_ic =
                src + ((mb*IH + ih)*IW + iw)*ABIC/ELEM_SIZE;
            const unsigned int *weights_ic =
                weights + ((kh*KW + kw)*OC + oc)*ABIC/ELEM_SIZE;

            for (int vabic = 0; vabic < VECTORS_IN_ABIC; vabic++) {
                uint32x4_t v_src =
                    vld1q_u32(src_ic + vabic*VLEN/ELEM_SIZE);
                uint32x4_t v_weights =
                    vld1q_u32(weights_ic + vabic*VLEN/ELEM_SIZE);

                uint32x4_t v_xor = veorq_u32(v_src, v_weights);
                uint32x4_t v_xnor = vmvnq_u32(v_xor);

                uint8x16_t v_cnt16 = vcntq_u8(vreinterpretq_u8_u32(v_xnor));
                uint16x8_t v_cnt8 = vpaddlq_u8(v_cnt16);
                uint32x4_t v_cnt4 = vpaddlq_u16(v_cnt8);
                v_accum = vaddq_u32(v_cnt4, v_accum);
            }
        }
        uint64x2_t v_cnt2 = vpaddlq_u32(v_accum);
        dst_i += vgetq_lane_u64(v_cnt2, 0);
        dst_i += vgetq_lane_u64(v_cnt2, 1);
        *d = (float)dst_i * *alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

} // namespace implementation
} // namespace xnor_nn

#else

namespace xnor_nn {
namespace implementation {

xnor_nn_status_t DirectConvolution::exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
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

    const int MB = c->mb;
    const int IH = c->ih;
    const int IW = c->iw;
    const int OC = c->oc;
    const int OH = c->oh;
    const int OW = c->ow;
    const int KH = c->kh;
    const int KW = c->kw;
    const int SH = c->sh;
    const int SW = c->sw;
    const int PH = c->ph;
    const int PW = c->pw;

    const int ABIC = c->abic;
    const int ELEM_SIZE = 32;

    const int VECTORS_IN_ABIC = ABIC / VLEN;

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
            const int ih = oh*SH - PH + kh;
            const int iw = ow*SW - PW + kw;

            if (ih < 0 || iw < 0) continue;
            if (ih >= IH || iw >= IW) continue;

            const unsigned int *src_ic =
                src + ((mb*IH + ih)*IW + iw)*ABIC/ELEM_SIZE;
            const unsigned int *weights_ic =
                weights + ((kh*KW + kw)*OC + oc)*ABIC/ELEM_SIZE;

            for (int vabic = 0; vabic < VECTORS_IN_ABIC; vabic++)
            for (int v = 0; v < VLEN / ELEM_SIZE; v++) {
                int src_idx = vabic*VLEN/ELEM_SIZE + v;
                int weights_idx = vabic*VLEN/ELEM_SIZE + v;

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

} // namespace implementation
} // namespace xnor_nn

#endif

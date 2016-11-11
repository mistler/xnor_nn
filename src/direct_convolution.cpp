#include "implementation.hpp"

#ifdef __x86_64__
#include <immintrin.h>

xnor_nn_status_t direct_convolution_forward(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
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

    const int AIC = c->aic;
    const int VEC_LENGTH = c->vector_length;
    const int ELEM_SIZE = 4;

    const int VECTORS_IN_AIC = AIC / VEC_LENGTH;

    int ones[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    __m128i v_ones = _mm_load_si128((__m128i*)ones);

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(3) schedule(static)
    for (int mb = 0; mb < MB; mb++)
    for (int oc = 0; oc < OC; oc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        int dst_idx = ((mb*OC + oc)*OH + oh)*OW + ow;
        float *d = dst + dst_idx;
        *d = 0.f;
        int dst_i = 0;
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
                __m128i *v_ptr_src =
                    (__m128i*)(src_ic + aic*VEC_LENGTH/ELEM_SIZE);
                __m128i *v_ptr_weights =
                    (__m128i*)(weights_ic + aic*VEC_LENGTH/ELEM_SIZE);

                __m128i v_src = _mm_load_si128(v_ptr_src);
                __m128i v_weights = _mm_load_si128(v_ptr_weights);

                __m128i v_xor = _mm_xor_si128(v_src, v_weights);
                __m128i v_xnor = _mm_xor_si128(v_xor, v_ones);

                dst_i += __builtin_popcount(_mm_extract_epi32(v_xnor, 0));
                dst_i += __builtin_popcount(_mm_extract_epi32(v_xnor, 1));
                dst_i += __builtin_popcount(_mm_extract_epi32(v_xnor, 2));
                dst_i += __builtin_popcount(_mm_extract_epi32(v_xnor, 3));
            }
        }
        *d = (float)dst_i * *alpha * k[oh*OW + ow];
    }

    return xnor_nn_success;
}

#elif defined __arm__

#include <arm_neon.h>

xnor_nn_status_t direct_convolution_forward(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
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

    const int AIC = c->aic;
    const int VEC_LENGTH = c->vector_length;
    const int ELEM_SIZE = 4;

    const int VECTORS_IN_AIC = AIC / VEC_LENGTH;

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(3) schedule(static)
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
                uint32x4_t v_src = vld1q_u32(src_ic + aic*VEC_LENGTH/ELEM_SIZE);
                uint32x4_t v_weights =
                    vld1q_u32(weights_ic + aic*VEC_LENGTH/ELEM_SIZE);

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

#else

xnor_nn_status_t direct_convolution_forward(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
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

    const int AIC = c->aic;
    const int VEC_LENGTH = c->vector_length;
    const int ELEM_SIZE = 4;

    const int VECTORS_IN_AIC = AIC / VEC_LENGTH;

    // TODO: potentially loops can be reordered
#   pragma omp parallel for collapse(3) schedule(static)
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
            for (int v = 0; v < VEC_LENGTH / ELEM_SIZE; v++) {
                int src_idx = aic*VEC_LENGTH/ELEM_SIZE + v;
                int weights_idx = aic*VEC_LENGTH/ELEM_SIZE + v;

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

#endif

#include "direct_convolution.hpp"

#include "utils.hpp"

// TODO: log execution

#ifdef ARCH_X86

#define TEMPLATE_CONVOLUTION
#include "direct_convolution_avx.hpp"
#undef TEMPLATE_CONVOLUTION
#include "direct_convolution_avx.hpp"

#elif defined ARCH_ARM

#define TEMPLATE_CONVOLUTION
#include "direct_convolution_neon.hpp"
#undef TEMPLATE_CONVOLUTION
#include "direct_convolution_neon.hpp"

#else

#define TEMPLATE_CONVOLUTION
#include "direct_convolution_default.hpp"
#undef TEMPLATE_CONVOLUTION
#include "direct_convolution_default.hpp"

#endif

#define TRY(OC, IC, IH, IW, KH, KW, SH, SW, PH, PW) \
    if (OC == c->oc && IC == c->ic && IH == c->ih && IW == c->iw \
            && KH == c->kh && KW == c->kw && SH == c->sh && SW == c->sw \
            && PH == c->ph && PW == c->pw) \
    { \
        constexpr int OH = getOH(IH, KH, SH, PH); \
        constexpr int OW = getOW(IH, KH, SH, PH); \
        constexpr int ABIC = getABIC(IC); \
        c->forward = exec_template< \
                OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, OH, OW, ABIC>; \
        return; \
    }

namespace xnor_nn {
namespace implementation {

bool DirectConvolution::isApplicable(const xnor_nn_convolution_t *c) const {
    bool ok = this->DirectBase::isApplicable(c)
        && c->forward == nullptr;
    return ok;
}

void DirectConvolution::setupConvolution(xnor_nn_convolution_t *c) {
    DirectConvolution *op = new DirectConvolution;
    op->DirectBase::setupConvolution(c);
    setState(c, op, xnor_nn_operation_convolution_forward);

    c->resource_size[xnor_nn_resource_bin_src] =
        c->mb * ABIC * c->ih * c->iw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_bin_weights] =
        c->oc * ABIC * c->kh * c->kw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
    c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);

    // OC, IC, IH, IW, KH, KW, SH, SW, PH, PW

    // AlexNet
    TRY(96, 3, 227, 227, 11, 11, 4, 4, 0, 0);
    TRY(256, 96, 27, 27, 5, 5, 1, 1, 2, 2);
    TRY(384, 256, 13, 13, 3, 3, 1, 1, 1, 1);
    TRY(384, 384, 13, 13, 3, 3, 1, 1, 1, 1);
    TRY(256, 384, 13, 13, 3, 3, 1, 1, 1, 1);

    c->forward = exec_simple;
}

DirectConvolution::~DirectConvolution() {}

} // namespace implementation
} // namespace xnor_nn

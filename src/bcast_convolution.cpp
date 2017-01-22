#include "bcast_convolution.hpp"

#include "utils.h"

// TODO: log execution

#ifdef ARCH_X86

#define TEMPLATE_CONVOLUTION
#include "bcast_convolution_avx.hpp"
#undef TEMPLATE_CONVOLUTION
#include "bcast_convolution_avx.hpp"

#elif defined ARCH_ARM

#define TEMPLATE_CONVOLUTION
#include "bcast_convolution_neon.hpp"
#undef TEMPLATE_CONVOLUTION
#include "bcast_convolution_neon.hpp"

#else

#define TEMPLATE_CONVOLUTION
#include "bcast_convolution_default.hpp"
#undef TEMPLATE_CONVOLUTION
#include "bcast_convolution_default.hpp"

#endif

#define TRY(OC, IC, IH, IW, KH, KW, SH, SW, PH, PW) \
    if (OC == c->oc && IC == c->ic && IH == c->ih && IW == c->iw \
            && KH == c->kh && KW == c->kw && SH == c->sh && SW == c->sw \
            && PH == c->ph && PW == c->pw) \
    { \
        constexpr int OH = getOH(IH, KH, SH, PH); \
        constexpr int OW = getOH(IW, KW, SW, PW); \
        constexpr int OCO = getOCO(OC); \
        constexpr int ICO = getICO(IC); \
        c->forward = exec_template<OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, \
                OH, OW, OCO, ICO, OCI>; \
        return; \
    }

namespace xnor_nn {
namespace implementation {

bool BcastConvolution::isApplicable(
        const xnor_nn_convolution_t *c) const {
    bool ok = this->BcastBase::isApplicable(c)
        && c->forward == nullptr;
    return ok;
}

void BcastConvolution::setupConvolution(
        xnor_nn_convolution_t *c) {
    BcastConvolution *op = new BcastConvolution;
    op->BcastBase::setupConvolution(c);
    setState(c, op, xnor_nn_operation_convolution_forward);

    // TODO: move it to base class
    c->resource_size[xnor_nn_resource_bin_src] =
        c->mb * ABIC * c->ih * c->iw * sizeof(char);
    c->resource_size[xnor_nn_resource_bin_weights] =
        OCO * c->kh * c->kw * ICO * OCI * sizeof(int);
    c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
    c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);

    // OC, IC, IH, IW, KH, KW, SH, SW, PH, PW

    // AlexNet
    TRY(192, 64, 27, 27, 5, 5, 1, 1, 2, 2);
    TRY(384, 192, 13, 13, 3, 3, 1, 1, 1, 1);
    TRY(256, 384, 13, 13, 3, 3, 1, 1, 1, 1);
    TRY(256, 256, 13, 13, 3, 3, 1, 1, 1, 1);

    // Task
    TRY(32, 1, 60, 61, 3, 3, 1, 1, 0, 0);
    TRY(32, 32, 20, 20, 3, 3, 1, 1, 0, 0);

    c->forward = exec_simple;
}

BcastConvolution::~BcastConvolution() {}

} // namespace implementation
} // namespace xnor_nn

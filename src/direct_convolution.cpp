#include "direct_convolution.hpp"

#include "utils.h"

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

#define USE(IC, IH, IW, KH, KW, SH, SW, PH, PW) \
    if (IC == c->ic && IH == c->ih && IW == c->iw && KH == c->kh \
            && KW == c->kw && SH == c->sh && SW == c->sw && PH == c->ph \
            && PW == c->pw) \
    { \
        c->forward = exec_template<IC, IH, IW, KH, KW, SH, SW, PH, PW>; \
        return; \
    }

#define U1 USE(64, 27, 27, 5, 5, 1, 1, 2, 2)
#define U2 USE(192, 13, 13, 3, 3, 1, 1, 1, 1)
#define U3 USE(384, 13, 13, 3, 3, 1, 1, 1, 1)
#define U4 USE(256, 13, 13, 3, 3, 1, 1, 1, 1)

namespace xnor_nn {
namespace implementation {

bool DirectConvolution::isApplicable(
        const xnor_nn_convolution_t *c) const {
    if (c->forward != nullptr) return false;
    if (c->algorithm != xnor_nn_algorithm_direct) return false;
    return true;
}

void DirectConvolution::setupConvolution(xnor_nn_convolution_t *c) {
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

    std::vector<Implementation*> *vec =
        (std::vector<Implementation*>*)c->state;
    vec->push_back(op);

    U1;
    U2;
    U3;
    U4;

    c->forward = exec_simple;
}

DirectConvolution::~DirectConvolution() {}

} // namespace implementation
} // namespace xnor_nn

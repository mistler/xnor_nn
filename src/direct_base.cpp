#include "direct_base.hpp"

#include "utils.h"

namespace xnor_nn {
namespace implementation {

bool DirectBase::isApplicable(const xnor_nn_convolution_t *c) const {
    bool ok = true
        && c->algorithm == xnor_nn_algorithm_direct;
    return ok;
}

void DirectBase::setupConvolution(xnor_nn_convolution_t *c) {
    BIC = ((c->ic + BITS - 1) / BITS) * BITS;
    ABIC = ((BIC + VLEN - 1) / VLEN) * VLEN;

    c->resource_size[xnor_nn_resource_bin_src] =
        c->mb * ABIC * c->ih * c->iw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_bin_weights] =
        c->oc * ABIC * c->kh * c->kw * ELEM_SIZE;
    c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
    c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);
}

DirectBase::~DirectBase() {}

} // namespace implementation
} // namespace xnor_nn

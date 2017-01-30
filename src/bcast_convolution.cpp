#include "bcast_convolution.hpp"

#include "cpuid.hpp"
#include "bcast_template_parameters.hpp"

namespace xnor_nn {
namespace implementation {

bool BcastConvolution::isApplicable(const xnor_nn_convolution_t *c) const {
    bool ok = this->BcastBase::isApplicable(c)
        && c->forward == nullptr;
    return ok;
}

void BcastConvolution::setupConvolution(xnor_nn_convolution_t *c) {
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

    using Cpuid = xnor_nn::utils::Cpuid;

#ifdef __x86_64__
    if (Cpuid::avx()) {
        BCAST_TEMPLATE_ASSIGN(c, avx);
    } else {
        BCAST_TEMPLATE_ASSIGN(c, default);
    }
#elif defined __arm__
    if (Cpuid::neon()) {
        BCAST_TEMPLATE_ASSIGN(c, neon);
    } else {
        BCAST_TEMPLATE_ASSIGN(c, default);
    }
#endif

    c->forward = exec_default_simple;
}

BcastConvolution::~BcastConvolution() {}

} // namespace implementation
} // namespace xnor_nn

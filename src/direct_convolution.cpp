#include "direct_convolution.hpp"

#include "cpuid.hpp"

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

    using Cpuid = xnor_nn::utils::Cpuid;

#ifdef __x86_64__
    if (Cpuid::avx()) {
        DIRECT_TEMPLATE_ASSIGN(c, avx);
        c->forward = exec_avx_simple;
    } else {
        DIRECT_TEMPLATE_ASSIGN(c, default);
        c->forward = exec_default_simple;
    }
#elif defined __arm__
    if (Cpuid::neon()) {
        DIRECT_TEMPLATE_ASSIGN(c, neon);
        c->forward = exec_neon_simple;
    } else {
        DIRECT_TEMPLATE_ASSIGN(c, default);
        c->forward = exec_default_simple;
    }
#endif

    c->forward = exec_default_simple;
}

DirectConvolution::~DirectConvolution() {}

} // namespace implementation
} // namespace xnor_nn

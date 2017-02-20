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

    using Cpuid = xnor_nn::utils::Cpuid;

#ifdef __x86_64__
    if (Cpuid::avx()) {
        BCAST_TEMPLATE_ASSIGN(c, avx);
        c->forward = exec_avx_simple;
    } else {
        BCAST_TEMPLATE_ASSIGN(c, default);
        c->forward = exec_default_simple;
    }
#elif defined __arm__
    if (Cpuid::neon()) {
        BCAST_TEMPLATE_ASSIGN(c, neon);
        c->forward = exec_neon_simple;
    } else {
        BCAST_TEMPLATE_ASSIGN(c, default);
        c->forward = exec_default_simple;
    }
#endif
}

BcastConvolution::~BcastConvolution() {}

} // namespace implementation
} // namespace xnor_nn

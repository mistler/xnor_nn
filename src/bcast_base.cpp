#include "bcast_base.hpp"

#include "cpuid.hpp"
#include "utils.hpp"

namespace xnor_nn {
namespace implementation {

bool BcastBase::isApplicable(const xnor_nn_convolution_t *c) const {
    bool ok = true
        && c->algorithm == xnor_nn_algorithm_bcast
        && c->oc % getOCI() == 0;
    return ok;
}

void BcastBase::setupConvolution(xnor_nn_convolution_t *c) {
    // TODO: unify
    BIC = utils::div_up(c->ic, BITS);
    ABIC = utils::div_up(BIC, BICI) * BICI;

    ICO = getICO(c->ic);
    OCO = getOCO(c->oc);
    OCI = getOCI();

    c->resource_size[xnor_nn_resource_bin_src] =
        c->mb * ABIC * c->ih * c->iw * sizeof(char);
    c->resource_size[xnor_nn_resource_bin_weights] =
        OCO * c->kh * c->kw * ICO * OCI * sizeof(int);
    c->resource_size[xnor_nn_resource_a] = c->ih * c->iw * sizeof(float);
    c->resource_size[xnor_nn_resource_k] = c->oh * c->ow * sizeof(float);
    c->resource_size[xnor_nn_resource_operations_count]
        = c->oh * c->ow * sizeof(int);
}

BcastBase::~BcastBase() {}

constexpr int BcastBase::SZ;
constexpr int BcastBase::BITS;
constexpr int BcastBase::BICI;

} // namespace implementation
} // namespace xnor_nn

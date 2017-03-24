#include "bcast_convolution.hpp"

#include <vector>

#include "cpuid.hpp"
#include "isa_traits.hpp"

namespace xnor_nn {
namespace implementation {

template<typename isa_traits, int N>
struct dispatch_helper{
    static xnor_nn_executor_t dispatch(const std::vector<int> &p) {
        bool b = true;
        for (int i = 0; i < tp_size; i++)
            b = b && (tp[N*tp_size+i] == p[i]);
        if (b) return BcastConvolution::exec<isa_traits,
            tp[N*tp_size+0], tp[N*tp_size+1], tp[N*tp_size+2], tp[N*tp_size+3],
            tp[N*tp_size+4], tp[N*tp_size+5], tp[N*tp_size+6], tp[N*tp_size+7],
            tp[N*tp_size+8], tp[N*tp_size+9]>;
        return dispatch_helper<isa_traits, N-1>::dispatch(p);
    }
};
template<typename isa_traits>
struct dispatch_helper<isa_traits, -1>{
    static xnor_nn_executor_t dispatch(const std::vector<int> &p) {
        (void)p;
        return nullptr;
    }
};

template<typename isa_traits>
struct dispatcher{
    static xnor_nn_executor_t dispatch(const xnor_nn_convolution_t *c) {
        return dispatch_helper<isa_traits, tp_elems-1>::dispatch({c->oc, c->ic,
                c->ih, c->iw, c->kh, c->kw, c->sh, c->sw, c->ph, c->pw});
    }
};

bool BcastConvolution::isApplicable(const xnor_nn_convolution_t *c) const {
    bool ok = true
        && c->algorithm == xnor_nn_algorithm_bcast
        && c->oc % getOCI() == 0
        && c->forward == nullptr;
    return ok;
}

void BcastConvolution::setupConvolution(xnor_nn_convolution_t *c) {
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

    BcastConvolution *op = new BcastConvolution(*this);
    setState(c, op);

    c->binarize_data = binarize_data;
    c->binarize_weights = binarize_weights;
    c->calculate_k = calculate_k;

    using Cpuid = xnor_nn::utils::Cpuid;
#ifdef __x86_64__
    if (Cpuid::avx()) {
        using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_avx>;
        c->forward = dispatcher<isa>::dispatch(c);
        if (!c->forward) c->forward = exec<isa>;
        return;
    }
    if (Cpuid::sse3()) {
        using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_sse3>;
        c->forward = dispatcher<isa>::dispatch(c);
        if (!c->forward) c->forward = exec<isa>;
        return;
    }
#elif defined __arm__
    if (Cpuid::neon()) {
        using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_neon>;
        c->forward = dispatcher<isa>::dispatch(c);
        if (!c->forward) c->forward = exec<isa>;
        return;
    }
#endif
    using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_default>;
    c->forward = dispatcher<isa>::dispatch(c);
    if (!c->forward) c->forward = exec<isa>;
}

BcastConvolution::~BcastConvolution() {}

constexpr int BcastConvolution::SZ;
constexpr int BcastConvolution::BITS;
constexpr int BcastConvolution::BICI;

} // namespace implementation
} // namespace xnor_nn

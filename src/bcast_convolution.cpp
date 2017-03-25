#include "bcast_convolution.hpp"

#include <vector>

#include "xnor_nn_types.h"
#include "cpuid.hpp"
#include "isa_traits.hpp"
#include "convolution_traits.hpp"

namespace xnor_nn {
namespace implementation {

template<typename ConvTraits, typename isa_traits, int N>
struct dispatch_helper{
    static xnor_nn_executor_t dispatch(const std::vector<int> &p) {
        bool b = true;
        for (int i = 0; i < tp_size; i++)
            b = b && (tp[N*tp_size+i] == p[i]);
        using conv = BcastConvolution<ConvTraits>;
        if (b) return conv::template exec<isa_traits,
            tp[N*tp_size+0], tp[N*tp_size+1], tp[N*tp_size+2], tp[N*tp_size+3],
            tp[N*tp_size+4], tp[N*tp_size+5], tp[N*tp_size+6], tp[N*tp_size+7],
            tp[N*tp_size+8], tp[N*tp_size+9]>;
        return dispatch_helper<ConvTraits, isa_traits, N-1>::dispatch(p);
    }
};
template<typename ConvTraits, typename isa_traits>
struct dispatch_helper<ConvTraits, isa_traits, -1>{
    static xnor_nn_executor_t dispatch(const std::vector<int> &p) {
        (void)p;
        return nullptr;
    }
};

template<typename ConvTraits, typename isa_traits>
struct dispatcher{
    static xnor_nn_executor_t dispatch(const xnor_nn_convolution_t *c) {
        return dispatch_helper<ConvTraits, isa_traits, tp_elems-1>::dispatch(
                {c->oc, c->ic, c->ih, c->iw, c->kh, c->kw,
                c->sh, c->sw, c->ph, c->pw});
    }
};

template<typename Traits>
bool BcastConvolution<Traits>::isApplicable(
        const xnor_nn_convolution_t *c) const {
    bool ok = true
        && c->algorithm == xnor_nn_algorithm_bcast
        && c->oc % getOCI() == 0
        && c->forward == nullptr;
    return ok;
}

template<typename Traits>
void BcastConvolution<Traits>::setupConvolution(xnor_nn_convolution_t *c) {
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
    c->resource_size[xnor_nn_resource_operations_count] =
        c->oh * c->ow * sizeof(int);

    using runtime_traits = ConvolutionTraits<RuntimeConvolutionTraits>;
    using runtime_conv = BcastConvolution<runtime_traits>;
    auto *op = new runtime_conv(*reinterpret_cast<runtime_conv*>(this));
    setState(c, op);

    c->binarize_data = binarize_data;
    c->binarize_weights = binarize_weights;
    c->calculate_k = calculate_k;

    using Cpuid = xnor_nn::utils::Cpuid;
    using CT = ConvolutionTraits<IntConvolutionTraits>;
#ifdef __x86_64__
    if (Cpuid::avx()) {
        using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_avx>;
        c->forward = dispatcher<CT, isa>::dispatch(c);
        if (!c->forward) c->forward = exec<isa>;
        return;
    }
    if (Cpuid::sse3()) {
        using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_sse3>;
        c->forward = dispatcher<CT, isa>::dispatch(c);
        if (!c->forward) c->forward = exec<isa>;
        return;
    }
#elif defined __arm__
    if (Cpuid::neon()) {
        using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_neon>;
        c->forward = dispatcher<CT, isa>::dispatch(c);
        if (!c->forward) c->forward = exec<isa>;
        return;
    }
#endif
    using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_default>;
    c->forward = dispatcher<CT, isa>::dispatch(c);
    if (!c->forward) c->forward = exec<isa>;
}

template<typename Traits>
BcastConvolution<Traits>::~BcastConvolution() {}

template<typename T> constexpr int BcastConvolution<T>::BICI;
template<typename T> constexpr int BcastConvolution<T>::BITS;
template<typename T> constexpr int BcastConvolution<T>::SZ;

template class BcastConvolution<ConvolutionTraits<RuntimeConvolutionTraits>>;
template class BcastConvolution<ConvolutionTraits<IntConvolutionTraits>>;

} // namespace implementation
} // namespace xnor_nn

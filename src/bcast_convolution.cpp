#include "bcast_convolution.hpp"

#include <vector>
#include <stdexcept>

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
    using Cpuid = xnor_nn::utils::Cpuid;

    bool ok = true
        && c->algorithm == xnor_nn_algorithm_bcast
        && c->oc % getOCI(Cpuid::vlen()) == 0
        && c->forward == nullptr
        && Traits::isApplicable(c);
    return ok;
}

template<typename Traits>
void BcastConvolution<Traits>::setupConvolution(xnor_nn_convolution_t *c) {
    using Cpuid = xnor_nn::utils::Cpuid;

    SZ = Traits::sz;
    BICI = Traits::bici;
    BITS = Traits::bits;

    BIC = getBIC(c->ic);
    ABIC = getABIC(c->ic);
    ICO = getICO(c->ic);
    OCI = getOCI(Cpuid::vlen());
    OCO = getOCO(c->oc, Cpuid::vlen());

    c->resource_size[xnor_nn_resource_bin_src] =
        c->mb * ABIC * c->ih * c->iw * sizeof(char);
    c->resource_size[xnor_nn_resource_bin_weights] =
        OCO * c->kh * c->kw * ICO * OCI * sizeof(int);
    c->resource_size[xnor_nn_resource_a] =
        c->mb * c->ih * c->iw * sizeof(float);
    c->resource_size[xnor_nn_resource_k] =
        c->mb * c->oh * c->ow * sizeof(float);
    c->resource_size[xnor_nn_resource_alpha] = c->oc * sizeof(float);
    c->resource_size[xnor_nn_resource_operations_count] =
        c->oh * c->ow * sizeof(typename Traits::data_t);

    using runtime_traits = ConvolutionTraits<RuntimeConvolutionTraits>;
    using runtime_conv = BcastConvolution<runtime_traits>;
    auto *op = new runtime_conv(*reinterpret_cast<runtime_conv*>(this));
    setState(c, op);

    c->binarize_data = binarize_data;
    c->binarize_weights = binarize_weights;
    c->calculate_k = calculate_k;

#ifdef __x86_64__
    if (Cpuid::avx()) {
        using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_avx>;
        c->forward = dispatcher<Traits, isa>::dispatch(c);
        if (!c->forward) c->forward = exec<isa>;
        return;
    }
    if (Cpuid::sse3()) {
        using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_sse3>;
        c->forward = dispatcher<Traits, isa>::dispatch(c);
        if (!c->forward) c->forward = exec<isa>;
        return;
    }
#elif defined __arm__
    if (Cpuid::neon()) {
        using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_neon>;
        c->forward = dispatcher<Traits, isa>::dispatch(c);
        if (!c->forward) c->forward = exec<isa>;
        return;
    }
#endif
    using isa = xnor_nn::isa::isa_traits<xnor_nn::isa::isa_default>;
    c->forward = dispatcher<Traits, isa>::dispatch(c);
    if (!c->forward) c->forward = exec<isa>;
}

template<> void BcastConvolution<ConvolutionTraits<
    RuntimeConvolutionTraits>>::setupConvolution(xnor_nn_convolution_t *c) {
    (void)c;
    throw std::runtime_error(
            "Runtime convolution traits has no setup method");
}

template<typename Traits>
BcastConvolution<Traits>::~BcastConvolution() {}

template class BcastConvolution<ConvolutionTraits<ShortConvolutionTraits>>;
template class BcastConvolution<ConvolutionTraits<IntConvolutionTraits>>;
template class BcastConvolution<ConvolutionTraits<RuntimeConvolutionTraits>>;

} // namespace implementation
} // namespace xnor_nn

#ifndef INSTANTIATOR_HPP
#define INSTANTIATOR_HPP

static volatile bool noinline = true;
template<typename Traits, int N>
struct instantiator {
    inline static void instantiate() {
        void (*ptr)() = nullptr;
        if (noinline) ptr = (void (*)()) BcastConvolution<Traits>::template
            exec<isa,
            tp[N*tp_size+0], tp[N*tp_size+1], tp[N*tp_size+2], tp[N*tp_size+3],
            tp[N*tp_size+4], tp[N*tp_size+5], tp[N*tp_size+6], tp[N*tp_size+7],
            tp[N*tp_size+8], tp[N*tp_size+9]>;
        ptr();
        instantiator<Traits, N-1>::instantiate();
    }
};
template<typename Traits>
struct instantiator<Traits, -1> {
    inline static void instantiate() {}
};

template
struct instantiator<ConvolutionTraits<IntConvolutionTraits>, tp_elems-1>;
template
struct instantiator<ConvolutionTraits<ShortConvolutionTraits>, tp_elems-1>;

template
xnor_nn_status_t BcastConvolution<ConvolutionTraits<
    IntConvolutionTraits>>::template
    exec<isa>(const xnor_nn_convolution_t *c, xnor_nn_resources_t res);
template
xnor_nn_status_t BcastConvolution<ConvolutionTraits<
    ShortConvolutionTraits>>::template
    exec<isa>(const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

template<> template<>
xnor_nn_status_t BcastConvolution<ConvolutionTraits<
    RuntimeConvolutionTraits>>::template exec<isa>(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res) {
        (void)c; (void)res; return xnor_nn_unimplemented;
    }

#endif // INSTANTIATOR_HPP

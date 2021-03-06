#ifndef BCAST_CONVOLUTION_HPP
#define BCAST_CONVOLUTION_HPP

#include "implementation.hpp"

#include "xnor_nn_types.h"
#include "cpuid.hpp"
#include "utils.hpp"

namespace xnor_nn {
namespace implementation {

constexpr static const int tp[] = {
    // AlexNet
    96, 3, 227, 227, 11, 11, 4, 4, 0, 0,
    256, 96, 27, 27, 5, 5, 1, 1, 2, 2,
    384, 256, 13, 13, 3, 3, 1, 1, 1, 1,
    384, 384, 13, 13, 3, 3, 1, 1, 1, 1,
    256, 384, 13, 13, 3, 3, 1, 1, 1, 1,
    // Task
    32, 1, 60, 61, 3, 3, 1, 1, 0, 0,
    32, 32, 20, 20, 3, 3, 1, 1, 0, 0,
    // Cifar10_tmp
    128, 128, 32, 32, 3, 3, 1, 1, 1, 1,
    256, 128, 16, 16, 3, 3, 1, 1, 1, 1,
    256, 256, 16, 16, 3, 3, 1, 1, 1, 1,
    512, 256, 8, 8, 3, 3, 1, 1, 1, 1,
    512, 512, 8, 8, 3, 3, 1, 1, 1, 1,
    // Cifar10
    32, 3, 32, 32, 5, 5, 1, 1, 2, 2,
    32, 32, 32, 32, 5, 5, 1, 1, 2, 2,
    64, 32, 32, 32, 5, 5, 1, 1, 2, 2,
    // Mnist
    24, 1, 28, 28, 5, 5, 1, 1, 0, 0,
    48, 24, 8, 8, 5, 5, 1, 1, 0, 0,
};
constexpr int tp_size = 10;
constexpr int tp_elems = sizeof(tp) / sizeof(tp[0]) / tp_size;

static_assert((sizeof(tp) / sizeof(tp[0])) % tp_size == 0,
        "Incorrect tp params");

template<typename ConvTraits>
class BcastConvolution : public Implementation{
public:
    ~BcastConvolution();
    bool isApplicable(const xnor_nn_convolution_t *c) const;
    void setupConvolution(xnor_nn_convolution_t *c);

protected:
    static constexpr int getICO(const int IC) {
        return utils::div_up(utils::div_up(IC, ConvTraits::bici),
                ConvTraits::sz);
    }

    static constexpr int getOCI(const int VLEN) {
        return VLEN / ConvTraits::sz / ConvTraits::bici;
    }

    static constexpr int getOCO(const int OC, const int VLEN) {
        return utils::div_up(OC, getOCI(VLEN));
    }

    static constexpr int getBIC(const int IC) {
        return utils::div_up(IC, ConvTraits::bits);
    }

    static constexpr int getABIC(const int IC) {
        return utils::div_up(getBIC(IC), ConvTraits::bici) * ConvTraits::bici;
    }

protected:
    int SZ, BICI, BITS;
    int BIC, ABIC, ICO, OCI, OCO;

private:
template<typename isa_traits, int OC, int IC, int IH, int IW, int KH, int KW,
        int SH, int SW, int PH, int PW>
static xnor_nn_status_t exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

template<typename isa_traits>
static xnor_nn_status_t exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

static xnor_nn_status_t binarize_data(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

static xnor_nn_status_t binarize_weights(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

static xnor_nn_status_t calculate_k(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

// I am friend of myself
template<typename T>
friend class BcastConvolution;

// template helpers
template<typename T, int N>
friend struct instantiator;

template<typename CT, typename isa_traits, int N>
friend struct dispatch_helper;
};

} // namespace implementation
} // namespace xnor_nn

#endif // BCAST_CONVOLUTION_HPP

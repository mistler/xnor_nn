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
};
constexpr int tp_size = 10;
constexpr int tp_elems = sizeof(tp) / sizeof(tp[0]) / tp_size;

static_assert((sizeof(tp) / sizeof(tp[0])) % tp_size == 0,
        "Incorrect tp params");

class BcastConvolution : public Implementation{
public:
    ~BcastConvolution();
    bool isApplicable(const xnor_nn_convolution_t *c) const;
    void setupConvolution(xnor_nn_convolution_t *c);

protected:
    static constexpr int constexpr_getICO(int IC) {
        return utils::div_up(utils::div_up(IC, BICI), SZ);
    }

    static constexpr int constexpr_getOCO(int OC, int VLEN) {
        return utils::div_up(OC, constexpr_getOCI(VLEN));
    }

    static constexpr int constexpr_getOCI(int VLEN) {
        return VLEN / SZ / BICI;
    }

    static constexpr int getICO(int IC) {
        return utils::div_up(utils::div_up(IC, BICI), SZ);
    }

    static int getOCO(int OC) {
        return utils::div_up(OC, getOCI());
    }

    static int getOCI() {
        return xnor_nn::utils::Cpuid::vlen() / SZ / BICI;
    }

protected:
    int BIC, ABIC, ICO, OCO, OCI;

    static constexpr int SZ = 8;
    static constexpr int BITS = sizeof(char) * SZ;
    static constexpr int BICI = sizeof(int);

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

// template helpers
template<int N>
friend struct instantiator;
template<typename isa_traits, int N>
friend struct dispatch_helper;
};

} // namespace implementation
} // namespace xnor_nn

#endif // BCAST_CONVOLUTION_HPP

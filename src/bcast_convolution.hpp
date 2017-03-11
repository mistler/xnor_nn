#ifndef BCAST_CONVOLUTION_HPP
#define BCAST_CONVOLUTION_HPP

#include "bcast_base.hpp"

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

class BcastConvolution : public BcastBase {
public:
    ~BcastConvolution();
    bool isApplicable(const xnor_nn_convolution_t *c) const;
    void setupConvolution(xnor_nn_convolution_t *c);

private:
template<typename isa_traits, int OC, int IC, int IH, int IW, int KH, int KW,
        int SH, int SW, int PH, int PW>
static xnor_nn_status_t exec(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

template<typename isa_traits>
static xnor_nn_status_t exec(
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

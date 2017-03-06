#ifndef BCAST_CONVOLUTION_HPP
#define BCAST_CONVOLUTION_HPP

#include "bcast_base.hpp"

namespace xnor_nn {
namespace implementation {

class BcastConvolution : public BcastBase {
public:
    ~BcastConvolution();
    bool isApplicable(const xnor_nn_convolution_t *c) const;
    void setupConvolution(xnor_nn_convolution_t *c);

private:
template<typename ISA, int OC, int IC, int IH, int IW, int KH, int KW,
        int SH, int SW, int PH, int PW>
static xnor_nn_status_t exec_template(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);

template<typename ISA>
static xnor_nn_status_t exec_simple(
        const xnor_nn_convolution_t *c, xnor_nn_resources_t res);
};

} // namespace implementation
} // namespace xnor_nn

#endif // BCAST_CONVOLUTION_HPP

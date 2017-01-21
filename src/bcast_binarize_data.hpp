#ifndef BCAST_BINARIZE_DATA_HPP
#define BCAST_BINARIZE_DATA_HPP

#include "bcast_base.hpp"

namespace xnor_nn {
namespace implementation {

class BcastBinarizeData : public BcastBase {
public:
    ~BcastBinarizeData();
    bool isApplicable(const xnor_nn_convolution_t *c) const;
    void setupConvolution(xnor_nn_convolution_t *c);
private:
    static xnor_nn_status_t exec(const xnor_nn_convolution_t *c,
            xnor_nn_resources_t res);
};

} // namespace implementation
} // namespace xnor_nn

#endif // BCAST_BINARIZE_DATA_HPP

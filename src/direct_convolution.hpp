#ifndef DIRECT_CONVOLUTION_HPP
#define DIRECT_CONVOLUTION_HPP

#include "implementation.hpp"

namespace xnor_nn {
namespace implementation {

class DirectConvolution : public Implementation {
public:
    ~DirectConvolution();
    bool isApplicable(const xnor_nn_convolution_t *c) const;
    void setupConvolution(xnor_nn_convolution_t *c);
private:
    template<int IC, int IH, int IW, int KH, int KW,
        int SH, int SW, int PH, int PW>
    static xnor_nn_status_t exec_template(const xnor_nn_convolution_t *c,
            xnor_nn_resources_t res);

    static xnor_nn_status_t exec_simple(const xnor_nn_convolution_t *c,
            xnor_nn_resources_t res);
};

} // namespace implementation
} // namespace xnor_nn

#endif // DIRECT_CONVOLUTION_HPP

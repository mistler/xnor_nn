#include "implementation.hpp"

#include <vector>

#include "bcast_convolution.hpp"
#include "direct_convolution.hpp"

#include "bcast_binarize_weights.hpp"
#include "direct_binarize_weights.hpp"
#include "bcast_binarize_data.hpp"
#include "direct_binarize_data.hpp"

#include "reference_binarize_data.hpp"
#include "reference_binarize_weights.hpp"
#include "reference_convolution.hpp"
#include "reference_calculate_k.hpp"

namespace xnor_nn {
namespace implementation {

std::vector<Implementation*> Implementations() {
    static std::vector<Implementation*> impls = {
        new BcastConvolution(),
        new DirectConvolution(),

        new BcastBinarizeWeights(),
        new DirectBinarizeWeights(),

        new BcastBinarizeData(),
        new DirectBinarizeData(),

        new ReferenceBinarizeWeights(),
        new ReferenceBinarizeData(),
        new ReferenceConvolution(),
        new ReferenceCalculateK(),
    };
    return impls;
};

void Implementation::setState(xnor_nn_convolution_t *c, Implementation *impl,
        const xnor_nn_operation_t operation) {
    std::vector<Implementation*> *vec =
        (std::vector<Implementation*>*)c->state;
    // TODO: use map
    impl->op = operation;
    vec->push_back(impl);
}

Implementation *Implementation::getState(const xnor_nn_convolution_t *c,
        const xnor_nn_operation_t operation) {
    std::vector<Implementation*> *vec =
        (std::vector<Implementation*>*)c->state;
    for (auto it = vec->begin(); it != vec->end(); ++it) {
        if ((*it)->op == operation) return *it;
    }

    return nullptr;
}

} // namespace implementation
} // namespace xnor_nn

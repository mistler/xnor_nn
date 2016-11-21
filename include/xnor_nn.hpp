#ifndef XNOR_NN_HPP
#define XNOR_NN_HPP

#include <memory>
#include <stdexcept>

#include "xnor_nn.h"

namespace xnor_nn {

class Convolution {
public:
    Convolution(const xnor_nn_algorithm_t algorithm,
            int MB, int OC, int IC, int IH, int IW,
            int KH, int KW, int SH, int SW, int PH, int PW,
            const void *weights) : res_{0} {
        check_status(xnor_nn_init_convolution(&convolution_, algorithm,
                    MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW));

        // Internal memory
        check_status(xnor_nn_allocate_resources(&convolution_, res_));

        // Prepare weights
        res_[xnor_nn_resource_user_weights] = (void*)weights;
        check_status(convolution_.binarize_weights(&convolution_, res_));
    }

    ~Convolution() {
        xnor_nn_free_resources(res_);
        xnor_nn_destroy_convolution(&convolution_);
    }


    void forward(const void *src, void *dst) {
        res_[xnor_nn_resource_user_src] = const_cast<void*>(src);
        res_[xnor_nn_resource_user_dst] = dst;
        check_status(convolution_.binarize_data(&convolution_, res_));
        check_status(convolution_.calculate_k(&convolution_, res_));
        check_status(convolution_.forward(&convolution_, res_));
        res_[xnor_nn_resource_user_src] = 0;
        res_[xnor_nn_resource_user_dst] = 0;
    }

private:
    typedef char data_t;

    xnor_nn_convolution_t convolution_;

    xnor_nn_resources_t res_;

    void check_status(xnor_nn_status_t status) {
        if (status == xnor_nn_success) return;
        char st_msg[16] = {0};
        xnor_nn_get_status_message(st_msg, status);
        throw new std::runtime_error{std::string(st_msg)};
    }
};

}
#endif

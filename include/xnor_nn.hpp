#ifndef XNOR_NN_HPP
#define XNOR_NN_HPP

#include <memory>
#include <stdexcept>

#include "xnor_nn.h"

namespace xnor_nn {

class Convolution {
public:
    Convolution(int MB, int OC, int IC, int IH, int IW,
            int KH, int KW, int SH, int SW, int PH, int PW,
            const void *weights)
                : mb_(MB), ic_(IC), ih_(IH), iw_(IW), oc_(OC)
                , kh_(KH), kw_(KW), sh_(SH), sw_(SW), ph_(PH), pw_(PW) {

        check_status(xnor_nn_init_convolution(&convolution_,
                    MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW));
        check_status(xnor_nn_init_data_binarizer(
                    &src_binarizer_, MB, IC, IH, IW));
        check_status(xnor_nn_init_weights_binarizer(
                    &weights_binarizer_, OC, IC, KH, KW));

        oh_ = convolution_.oh;
        ow_ = convolution_.ow;

        // Internal memory
        size_t sz_src_bin = src_binarizer_.size(&src_binarizer_);
        size_t sz_weights_bin =weights_binarizer_.size(&weights_binarizer_);
        void *src_bin = nullptr;
        void *weights_bin = nullptr;
        check_status(xnor_nn_memory_allocate(&src_bin, sz_src_bin));
        check_status(xnor_nn_memory_allocate(&weights_bin, sz_weights_bin));
        src_.reset((data_t*)src_bin);
        weights_.reset((data_t*)weights_bin);

        // Prepare weights
        check_status(weights_binarizer_.execute(&weights_binarizer_,
                weights, weights_bin));
    }

    void forward(const void *src, void *dst) {
        check_status(src_binarizer_.execute(&src_binarizer_, src, &src_[0]));
        check_status(convolution_.forward(
                    &convolution_, &src_[0], &weights_[0], dst));
    }

private:
    typedef char data_t;

    int mb_, ic_, ih_, iw_, oc_, oh_, ow_, kh_, kw_, sh_, sw_, ph_, pw_;
    xnor_nn_convolution_t convolution_;
    xnor_nn_data_binarizer_t src_binarizer_;
    xnor_nn_weights_binarizer_t weights_binarizer_;

    std::unique_ptr<data_t[]> src_;
    std::unique_ptr<const data_t[]> weights_;

    void check_status(xnor_nn_status_t status) {
        if (status == xnor_nn_success) return;
        char st_msg[16] = {0};
        xnor_nn_get_status_message(st_msg, status);
        throw new std::runtime_error{std::string(st_msg)};
    }
};

}
#endif

#ifndef _XNOR_NN_SRC_UTILS_CONVOLUTION_LOGGER_HPP_
#define _XNOR_NN_SRC_UTILS_CONVOLUTION_LOGGER_HPP_

#include <string>
#include <sstream>

#include "logger.hpp"
#include "xnor_nn_types.h"

namespace xnor_nn {
namespace utils {
namespace logger {

struct init {};
struct exec {};

struct data {};
struct weights {};
struct k {};
struct convolution {};

namespace {

std::string s(const xnor_nn_convolution_t *c) {
    std::stringstream ss;
    ss << "src:";
    switch (c->src_format) {
    case xnor_nn_data_format_nchw:
        ss << "(nchw)[" << c->mb << "][" << c->ic << "]["
            << c->ih << "][" << c->iw << "]"; break;
    case xnor_nn_data_format_nhwc:
        ss << "(nhwc)[" << c->mb << "]["
            << c->ih << "][" << c->iw << "][" << c->ic << "]"; break;
    default: break;
    }
    return ss.str();
}

std::string w(const xnor_nn_convolution_t *c) {
    std::stringstream ss;
    ss << "weights:";
    switch (c->weights_format) {
    case xnor_nn_weights_format_oihw:
        ss << "(oihw)[" << c->oc << "][" << c->ic << "]["
            << c->kh << "][" << c->kw << "]"; break;
    case xnor_nn_data_format_nhwc:
        ss << "(hwio)[" << c->kh << "][" << c->kw
            << "][" << c->ic << "][" << c->oc << "]"; break;
    default: break;
    }
    return ss.str();
}

std::string d(const xnor_nn_convolution_t *c) {
    std::stringstream ss;
    ss << "dst:";
    switch (c->dst_format) {
    case xnor_nn_data_format_nchw:
        ss << "(nchw)[" << c->mb << "][" << c->oc << "]["
            << c->ih << "][" << c->iw << "]"; break;
    case xnor_nn_data_format_nhwc:
        ss << "(nhwc)[" << c->mb << "]["
            << c->ih << "][" << c->iw << "][" << c->oc << "]"; break;
    default: break;
    }
    return ss.str();
}

std::string padding(const xnor_nn_convolution_t *c) {
    std::stringstream ss;
    ss << "padding:";
    ss << "(hw)[" << c->ph << "][" << c->pw << "]";
    return ss.str();
}

std::string stride(const xnor_nn_convolution_t *c) {
    std::stringstream ss;
    ss << "stride:";
    ss << "(hw)[" << c->sh << "][" << c->sw << "]";
    return ss.str();
}

std::string algorithm(const xnor_nn_convolution_t *c) {
    std::stringstream ss;
    ss << "algorithm:()[";
    switch (c->algorithm) {
    case xnor_nn_algorithm_bcast:
        ss << "bcast"; break;
    case xnor_nn_algorithm_reference:
        ss << "reference"; break;
    default: break;
    }
    ss << "]";
    return ss.str();
}

} // namespace

template<typename Status, typename Operation>
struct log_helper {
    template<typename ... T>
    static void info(const xnor_nn_convolution_t *c, T ... t) {}
};
template<>
struct log_helper<exec, data> {
    template<typename ... T>
    static void info(const xnor_nn_convolution_t *c, T ... t) {
        xnor_nn::utils::Logger::info("execute:", "data_binarization:",
                algorithm(c), s(c), t...);
    }
};
template<>
struct log_helper<exec, weights> {
    template<typename ... T>
    static void info(const xnor_nn_convolution_t *c, T ... t) {
        xnor_nn::utils::Logger::info("execute:", "weights_binarization:",
                algorithm(c), w(c), t...);
    }
};
template<>
struct log_helper<exec, k> {
    template<typename ... T>
    static void info(const xnor_nn_convolution_t *c, T ... t) {
        xnor_nn::utils::Logger::info("execute:", "calculate_k:",
                algorithm(c), s(c), w(c), t...);
    }
};
template<>
struct log_helper<init, convolution> {
    template<typename ... T>
    static void info(const xnor_nn_convolution_t *c, T ... t) {
        xnor_nn::utils::Logger::info("init:", "convolution:",
                algorithm(c), s(c), w(c), stride(c), padding(c), d(c), t...);
    }
};
template<>
struct log_helper<exec, convolution> {
    template<typename ... T>
    static void info(const xnor_nn_convolution_t *c, T ... t) {
        xnor_nn::utils::Logger::info("execute:", "convolution:",
                algorithm(c), s(c), w(c), stride(c), padding(c), d(c), t...);
    }
};

template<typename Status, typename Operation>
struct log {
    template<typename ... T>
    static void info(const xnor_nn_convolution_t *c, T ... t) {
        if (!xnor_nn::utils::Logger::enabled()) return;
        log_helper<Status, Operation>::info(c, t...);
    }
};


} // logger
} // utils
} // xnor_nn

#endif // _XNOR_NN_SRC_UTILS_CONVOLUTION_LOGGER_HPP_

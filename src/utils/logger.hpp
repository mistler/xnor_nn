#ifndef XNOR_NN_LOGGER_HPP
#define XNOR_NN_LOGGER_HPP

#include <iostream>
#include <string>
#include <cstdlib>

namespace xnor_nn {
namespace utils {

class Logger {
public:
    Logger(): info_stream_(std::cout), is_verbose_(false) {
        read_environment();
    }

    template<typename ... T>
    static void info(T ... t) {
        Logger &inst = instance();
        if (!inst.is_verbose_) return;
        inst.info_stream_ << "xnor_nn: info:";
        inst.info_variadic(t...);
    }

private:
    std::ostream &info_stream_;
    bool is_verbose_;

private:
    static Logger &instance() {
        static Logger instance;
        return instance;
    }
    Logger(Logger const&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger const&) = delete;
    Logger& operator=(Logger &&) = delete;

    void read_environment() {
        char *env = getenv("XNOR_NN_VERBOSE");
        if (env != NULL && env[0] == '1') is_verbose_ = true;
    }

    // Variadic template magic
    void info_variadic() {
        Logger &inst = instance();
        inst.info_stream_ << std::endl;
    }

    template<typename T0, typename ... T>
    void info_variadic(const T0& t0, T ... t) {
        Logger &inst = instance();
        inst.info_stream_ << " " << t0;
        inst.info_variadic(t...);
    }
};

}
}

#endif

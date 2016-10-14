#ifndef XNOR_NN_LOGGER_HPP
#define XNOR_NN_LOGGER_HPP

#include <iostream>
#include <string>

namespace xnor_nn {
namespace utils {

class Logger {
public:
    Logger(): info_stream(std::cout) {}

    template<typename ... T>
    static void info(T ... t) {
        Logger &inst = instance();
        inst.info_stream << "xnor_nn: info:";
        inst.info_variadic(t...);
    }

private:
    std::ostream &info_stream;

private:
    static Logger &instance() {
        static Logger instance;
        return instance;
    }
    Logger(Logger const&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger const&) = delete;
    Logger& operator=(Logger &&) = delete;

    // Variadic template magic
    void info_variadic() {
        Logger &inst = instance();
        inst.info_stream << std::endl;
    }

    template<typename T0, typename ... T>
    void info_variadic(const T0& t0, T ... t) {
        Logger &inst = instance();
        inst.info_stream << " " << t0;
        inst.info_variadic(t...);
    }
};

}
}

#endif

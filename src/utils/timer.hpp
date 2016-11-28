#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

#include "utils.h"

namespace xnor_nn{
namespace utils{

class Timer{
public:
    void start(){
        s = std::chrono::high_resolution_clock::now();
    }

    void stop(){
        e = std::chrono::high_resolution_clock::now();
    }

    double millis(){
        return std::chrono::duration_cast<
            std::chrono::nanoseconds>(e-s).count() / 1000.0 / 1000.0;
    }

    double micros(){
        return std::chrono::duration_cast<
            std::chrono::nanoseconds>(e-s).count() / 1000.0;
    }

    double nanos(){
        return std::chrono::duration_cast<
            std::chrono::nanoseconds>(e-s).count();
    }

#ifdef ARCH_X86
    static inline unsigned long long rdtsc() {
        unsigned int lo, hi;
        asm volatile("rdtsc\n" : "=a"(lo), "=d"(hi));
        return ((unsigned long long)hi << 32) | lo;
    }
#endif

private:
    std::chrono::high_resolution_clock::time_point s, e;
};

} // namespace implementation
} // namespace xnor_nn
#endif // TIMER_HPP

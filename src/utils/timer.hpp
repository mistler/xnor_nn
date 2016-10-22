#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

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

private:
    std::chrono::high_resolution_clock::time_point s, e;
};

}
}
#endif

#include <ctime>

namespace xnor_nn{
namespace utils{

class Timer{
public:
    Timer(const clockid_t clock_type = CLOCK_REALTIME) : type(clock_type){}

    void start(){
        clock_gettime(type, &s);
    }

    void stop(){
        clock_gettime(type, &e);
    }

    long long int millis(){
        return (long long int)(e.tv_sec - s.tv_sec) * 1000 +
            (e.tv_nsec - s.tv_nsec) / 1000000L;
    }

    long long int micros(){
        return (long long int)(e.tv_sec - s.tv_sec) * 10000 +
            (e.tv_nsec - s.tv_nsec) / 100000L;
    }

private:
    struct timespec s, e;
    clockid_t type;
};

}
}

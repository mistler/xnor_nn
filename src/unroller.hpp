#ifndef UNROLLER_HPP
#define UNROLLER_HPP

namespace xnor_nn{

inline constexpr int get_unroll_factor(const int total, const int max_unroll) {
    int unroll = 0;
    for (int u = 1; u < max_unroll; u++)
        if (total % u == 0) unroll = u;
    return unroll;
}

template<int Max, int N, typename F>
struct unroller_{
    inline static void unroll(const F &f) {
        f(Max - N);
        unroller_<Max, N-1, F>::unroll(f);
    }
};

template<int Max, typename F>
struct unroller_<Max, 0, F>{
    inline static void unroll(const F&) {}
};

template<int Max>
struct unroller{
    template<typename F>
    inline static void unroll(const F &f) {
        unroller_<Max, Max, F>::unroll(f);
    }
};

} // namespace xnor_nn

#endif // UNROLLER_HPP

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <cmath>

#include "xnor_nn.h"
#include "timer.hpp"

typedef struct {
    int MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW;

    void print(std::ostream &stream) const {
        stream
            << "MB: " << MB
            << ", OC: " << OC
            << ", IC: " << IC
            << ", IH: " << IH
            << ", IW: " << IW
            << ", KH: " << KH
            << ", KW: " << KW
            << ", SH: " << SH
            << ", SW: " << SW
            << ", PH: " << PH
            << ", PW: " << PW
            << std::endl;
    }
} convolution_params;

static inline void clean_cache(char *more_than_cache, int more_than_cache_size) {
#   pragma omp parallel for
    for (int i = 0; i < more_than_cache_size; i++)
        more_than_cache[i]++;
}

template<typename F>
static inline void measure_time(F &f, std::string msg) {
    const int N_EXECUTIONS = 3;

    const int more_than_cache_size = 1024*1024*64;
    static char more_than_cache[more_than_cache_size];

    xnor_nn::utils::Timer timer;

    double time = 0.0;
    std::cout << msg << "... ";
    for (int n = 0; n < N_EXECUTIONS; n++) {
        clean_cache(more_than_cache, more_than_cache_size);
        timer.start();

        xnor_nn_status_t st = f();
        if (st != xnor_nn_success) abort();

        timer.stop();
        time += timer.millis();
    }
    std::cout << "Time: " << time << " ms." << std::endl;
}

int main(){
    const std::vector<convolution_params> params =
    {
        // AlexNet
        { 1, 3, 64, 224, 224, 11, 11, 4, 4, 2, 2 },
        { 1, 64, 192, 27, 27, 5, 5, 1, 1, 2, 2 },
        { 1, 192, 384, 13, 13, 3, 3, 1, 1, 1, 1 },
        { 1, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1 },
        { 1, 256, 256, 13, 13, 3, 3, 1, 1, 1, 1 },
    };

    const int enough = 1024*1024*256;

    float *workspace = new float[enough/sizeof(float)];
    float *dst = new float[enough/sizeof(float)];

    std::generate(workspace, workspace + enough / sizeof(float),
            [&]() { static int i = 0; return std::sin(i++) * 10.f; });

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_data_binarizer_t src_binarizer;
    xnor_nn_weights_binarizer_t weights_binarizer;
    xnor_nn_convolution_t convolution;

    size_t sz_src_bin;
    size_t sz_weights_bin;

    void *src_bin = NULL, *weights_bin = NULL;

    // Main loop
    int index = 0;
    for (const convolution_params &p : params) {
        std::cout << ++index << " of " << params.size() << ": ";
        p.print(std::cout);
        st = xnor_nn_init_convolution(&convolution,
                p.MB, p.OC, p.IC, p.IH, p.IW,
                p.KH, p.KW, p.SH, p.SW, p.PH, p.PW);
        if (st != xnor_nn_success) goto label;

        st = xnor_nn_init_data_binarizer(&src_binarizer, &convolution);
        if (st != xnor_nn_success) goto label;

        st = xnor_nn_init_weights_binarizer(&weights_binarizer, &convolution);
        if (st != xnor_nn_success) goto label;

        // Internal data
        sz_src_bin = src_binarizer.size(&src_binarizer);
        sz_weights_bin = weights_binarizer.size(&weights_binarizer);

        st = xnor_nn_memory_allocate(&src_bin, sz_src_bin);
        if (st != xnor_nn_success) goto label;
        st = xnor_nn_memory_allocate(&weights_bin, sz_weights_bin);
        if (st != xnor_nn_success) goto label;

        // Warm up
#       pragma omp parallel for schedule(static)
        for (int s = 0; s < enough/(int)sizeof(float); s++)
            dst[s] += dst[s]*0.001f;

        // Execute
        auto f_weights_bin = std::bind(weights_binarizer.execute,
                &weights_binarizer, workspace, weights_bin);
        measure_time(f_weights_bin, "Weights binarizer");

        auto f_src_bin = std::bind(src_binarizer.binarize,
                &src_binarizer, workspace, src_bin);
        measure_time(f_src_bin, "Src binarizer");

        auto f_src_k = std::bind(src_binarizer.calculate_k,
                &src_binarizer, src_bin);
        measure_time(f_src_k, "Src binarizer calculate k");

        auto f_conv_exec = std::bind(convolution.forward,
                &convolution, src_bin, weights_bin, dst);
        measure_time(f_conv_exec, "Convolution");

        // Clean up
        xnor_nn_memory_free(src_bin);
        xnor_nn_memory_free(weights_bin);
        src_bin = NULL;
        weights_bin = NULL;

        std::cout << std::endl;
    }

label:
    xnor_nn_memory_free(src_bin);
    xnor_nn_memory_free(weights_bin);

    delete[] workspace;
    delete[] dst;

    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
    return st != xnor_nn_success;
}

#include <vector>
#include <algorithm>
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
            << std::endl << std::endl;
    }
} convolution_params;

void cache_clean(const char *more_than_cache, int more_than_cache_size) {
    (void)more_than_cache;
    (void)more_than_cache_size;
}

int main(){
    const std::vector<convolution_params> params =
    {
        { 8, 64, 128, 13, 13, 3, 3, 1, 1, 1, 1 },
    };

    const int N_EXECUTIONS = 5;

    const int enough = 1024*1024*256;
    const int more_than_cache_size = 1024*1024*64;

    float *workspace = new float[enough/sizeof(float)];
    float *dst = new float[enough/sizeof(float)];
    char *more_than_cache = new char[more_than_cache_size];

    std::generate(workspace, workspace + enough / sizeof(float),
            [&]() { static int i = 0; return std::sin(i++) * 10.f; });
    std::generate(more_than_cache, more_than_cache + more_than_cache_size,
            [&]() { static char i = 0; return i++; });

    xnor_nn::utils::Timer timer;
    double time;

    xnor_nn_status_t st;
    char st_msg[16];

    xnor_nn_data_binarizer_t src_binarizer;
    xnor_nn_weights_binarizer_t weights_binarizer;
    xnor_nn_convolution_t convolution;

    size_t sz_src_bin;
    size_t sz_weights_bin;

    void *src_bin = NULL, *weights_bin = NULL;

    // Main loop
    for (const convolution_params &p : params) {
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

        // Time weights
        time = 0.0;
        std::cout << "Binarizing weights..." << std::endl;
        for (int n = 0; n < N_EXECUTIONS; n++) {
            cache_clean(more_than_cache, more_than_cache_size);
            timer.start();

            st = weights_binarizer.execute(&weights_binarizer,
                    workspace, weights_bin);
            if (st != xnor_nn_success) goto label;

            timer.stop();
            time += timer.millis();
        }
        std::cout << "Time: " << time << " ms." << std::endl << std::endl;

        // Time data bin
        time = 0.0;
        std::cout << "Binarizing data..." << std::endl;
        for (int i = 0; i < N_EXECUTIONS; i++) {
            cache_clean(more_than_cache, more_than_cache_size);
            timer.start();

            st = src_binarizer.binarize(&src_binarizer, workspace, src_bin);
            if (st != xnor_nn_success) goto label;

            st = src_binarizer.calculate_k(&src_binarizer, src_bin);
            if (st != xnor_nn_success) goto label;

            timer.stop();
            time += timer.millis();
        }
        std::cout << "Time: " << time << " ms." << std::endl << std::endl;

        // Time convolution forward
        time = 0.0;
        std::cout << "Forward convolution..." << std::endl;
        for (int i = 0; i < N_EXECUTIONS; i++) {
            cache_clean(more_than_cache, more_than_cache_size);
            timer.start();

            st = convolution.forward(&convolution, src_bin, weights_bin, dst);
            if (st != xnor_nn_success) goto label;

            timer.stop();
            time += timer.millis();
        }
        std::cout << "Time: " << time << " ms." << std::endl << std::endl;

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
    delete[] more_than_cache;

    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
    return st != xnor_nn_success;
}

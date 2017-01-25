#include <vector>
#include <functional>
#include <iostream>
#include <cmath>

#include "xnor_nn.h"
#include "timer.hpp"

// TODO: swap ic and oc
typedef struct {
    xnor_nn_algorithm_t algorithm;
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
            << ", Alg: " << algorithm
            << std::endl;
    }
} convolution_params;

static inline void clean_cache(char *more_than_cache,
        int more_than_cache_size) {
#   pragma omp parallel for
    for (int i = 0; i < more_than_cache_size; i++)
        more_than_cache[i]++;
}

template<typename F>
static inline void measure_time(F &f, std::string msg, int N = 64) {
    const int more_than_cache_size = 1024*1024*16; // 16mb
    static char more_than_cache[more_than_cache_size];

    xnor_nn::utils::Timer timer;

    double time = 0.0;
    std::cout << msg << "... ";
    for (int n = 0; n < N; n++) {
        clean_cache(more_than_cache, more_than_cache_size);
        timer.start();

        xnor_nn_status_t st = f();
        if (st != xnor_nn_success) abort();

        timer.stop();
        time += timer.millis();
    }
    std::cout << "Time: " << time / N << " ms." << std::endl;
}

int main(){
    const xnor_nn_algorithm_t dir = xnor_nn_algorithm_direct;
    const xnor_nn_algorithm_t bcast = xnor_nn_algorithm_bcast;
    const std::vector<convolution_params> params =
    {
        // Task
        { bcast, 1, 32, 1, 60, 61, 3, 3, 3, 3, 0, 0 },
        { bcast, 1, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 },
        // { bcast, 256, 32, 1, 60, 61, 3, 3, 3, 3, 0, 0 },
        // { bcast, 256, 32, 32, 20, 20, 3, 3, 1, 1, 0, 0 },

        // AlexNet
        //{ alg, MB, 96, 3, 224, 224, 11, 11, 4, 4, 2, 2 }, // conv1
        { dir, 1, 256, 96, 27, 27, 5, 5, 1, 1, 2, 2, }, // conv2
        { bcast, 1, 256, 96, 27, 27, 5, 5, 1, 1, 2, 2, }, // conv2
        { dir, 1, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv3
        { bcast, 1, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv3
        { dir, 1, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv4
        { bcast, 1, 384, 384, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv4
        { dir, 1, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv5
        { bcast, 1, 256, 384, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv5
        { dir, 256, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv3
        { bcast, 256, 384, 256, 13, 13, 3, 3, 1, 1, 1, 1, }, // conv3
    };

    const int enough = 256*1024*384; // 384mb on float

    float *workspace = new float[enough];
    float *dst = new float[enough];

#   pragma omp parallel for schedule(static)
    for (int i = 0; i < enough; i++)
        workspace[i] = 37.f * (27 - (i % 53));

    xnor_nn_status_t st;
    char st_msg[16];

    // Main loop
    int index = 0;
    for (const convolution_params &p : params) {

        xnor_nn_resources_t res = {0};

        xnor_nn_convolution_t convolution;

        res[xnor_nn_resource_user_src] = workspace;
        res[xnor_nn_resource_user_weights] = workspace;
        res[xnor_nn_resource_user_dst] = dst;

        std::cout << ++index << " of " << params.size() << ": ";
        p.print(std::cout);

        st = xnor_nn_init_convolution(&convolution, p.algorithm,
                p.MB, p.OC, p.IC, p.IH, p.IW,
                p.KH, p.KW, p.SH, p.SW, p.PH, p.PW);
        if (st != xnor_nn_success) goto label;

        st = xnor_nn_allocate_resources(&convolution, res);
        if (st != xnor_nn_success) goto label;

        // Warm up
#       pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < 8; k++)
        for (int s = 1; s < enough; s++)
            workspace[s-1] += workspace[s]*0.001f;

        // Execute
        auto f_weights_bin = std::bind(convolution.binarize_weights,
                &convolution, res);
        measure_time(f_weights_bin, "Weights binarization");

        auto f_src_bin = std::bind(convolution.binarize_data,
                &convolution, res);
        measure_time(f_src_bin, "Src binarization");

        auto f_src_k = std::bind(convolution.calculate_k,
                &convolution, res);
        measure_time(f_src_k, "K calculation");

        auto f_conv_exec = std::bind(convolution.forward,
                &convolution, res);
        measure_time(f_conv_exec, "Convolution forward");

        std::cout << std::endl;

        // Clean up
        xnor_nn_free_resources(res);
        xnor_nn_destroy_convolution(&convolution);
    }

label:
    delete[] workspace;
    delete[] dst;

    xnor_nn_get_status_message(st_msg, st);
    printf("%s\n", st_msg);
    return st != xnor_nn_success;
}

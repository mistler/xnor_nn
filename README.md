# XNOR_NN: Deep Convolutional Neural Networks using XNOR and Binary Primitives

This is an open source performance library for acceleration of Machine Learning
frameworks on CPU. It includes highly vectorized and parallelized
implementations of common primitives with binary operations used in
convolutional neural networks with C interface.

### Citation
```bash
@inproceedings{rastegariECCV16,
    Author = {Mohammad Rastegari and Vicente Ordonez and
            Joseph Redmon and Ali Farhadi},
    Title = {XNOR-Net: ImageNet ClassificationUsing Binary Convolutional
            Neural Networks},
    Booktitle = {ECCV},
    Year = {2016}
}
```

### Build
```
    (mkdir -p build && rm -rf build/* && cd build && cmake ../ && make -j && make -j test)
```

Performance results for conv4 from AlexNet:
1x384x13x13->1x256x13x13
1 openmp thread

CPU: x86 Intel IvyBridge i7-3517u (AVX)
--------------------------------------------------------------------------------
Minibatch   |   Caffe(gemm-algorithm)   |   xnor(direct)
256         |   7013.8 ms               |   2887 ms (36+174+2677)
1           |   27.16 ms                |   11.5 ms

CPU: ARMv8 Cortex-A53 Raspberry PI 3 (NEON)
--------------------------------------------------------------------------------
Minibatch   |   Caffe(gemm-algorithm)   |   xnor(direct)
256         |   34225.5 ms              |   15756 ms (139+2139+13478)
1           |   134.4 ms                |   55.5 ms

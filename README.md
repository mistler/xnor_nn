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

### Performance
conv4 from Alexnet:
ICxIHxIW->OCxOHxOW
384x13x13->384x13x13

CPU: Intel IvyBridge i7-3517u (2 cores, AVX)
--------------------------------------------------------------------------------
Minibatch   |   Caffe(gemm-algorithm)   |   xnor(direct)
256         |   1944.55 ms              |   198 ms (240 ms)
1           |   7.0383 ms               |   1.5 ms (3.3 ms)

CPU: ARM Cortex-A53 (Raspberry PI 3)
--------------------------------------------------------------------------------
Minibatch   |   Caffe(gemm-algorithm)   |   xnor(direct)
256         |   20000 ms                |   9209 ms
1           |   70 ms                   |   55 ms

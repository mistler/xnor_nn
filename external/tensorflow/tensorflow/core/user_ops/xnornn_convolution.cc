/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "xnor_nn.hpp"

namespace tensorflow {

REGISTER_OP("XnornnConvolution")
    .Input("src: float")
    .Output("dst: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            // TODO: use correct shape
            // TODO: check that xnor_nn conv output equals with tf
            return Status::OK();
    });

class XnornnConvolution : public OpKernel {
 public:
  explicit XnornnConvolution(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                &output_tensor));
    auto output = output_tensor->flat<float>();

    const int N = input.size();
    for (int i = 0; i < N; i++) {
        output(i) = i;
    }


    const int MB = 2;
    const int IC = 64, OC = 128;
    const int IH = 13, IW = 13;
    const int OH = 13, OW = 13;
    const int KH = 3, KW = 3;
    const int SH = 1, SW = 1;
    const int PH = 1, PW = 1;

    // Usr data
    float *src = nullptr, *weights = nullptr;
    float *dst = nullptr;

    src = new float[MB*IC*IH*IW];
    weights = new float[OC*IC*KH*KW];
    dst = new float[MB*OC*OH*OW];

    // Convolution setup
    xnor_nn::Convolution convolution{xnor_nn_algorithm_best,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, weights};
    delete[] weights;

    convolution.forward(src, dst);

    delete[] src;
    delete[] dst;
  }
};

REGISTER_KERNEL_BUILDER(Name("XnornnConvolution").Device(DEVICE_CPU),
        XnornnConvolution);

}  // namespace tensorflow

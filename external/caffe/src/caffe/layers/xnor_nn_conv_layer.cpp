#include <vector>

#include "caffe/layers/xnor_nn_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void XnorNNConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void XnorNNConvolutionLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();

  if (!xnor_nn_conv) {
    const int MB = this->num_;

    const int OC = this->num_output_;
    const int IC = this->channels_;

    const int IH = this->conv_input_shape_.cpu_data()[1];
    const int IW = this->conv_input_shape_.cpu_data()[2];

    const int KH = this->kernel_shape_.cpu_data()[0];
    const int KW = this->kernel_shape_.cpu_data()[1];

    const int SH = this->stride_.cpu_data()[0];
    const int SW = this->stride_.cpu_data()[1];

    const int PH = this->pad_.cpu_data()[0];
    const int PW = this->pad_.cpu_data()[1];
    xnor_nn_conv.reset(new xnor_nn::Convolution{xnor_nn_algorithm_bcast,
            xnor_nn_data_format_nchw,
            xnor_nn_weights_format_oihw,
            xnor_nn_data_format_nchw,
            MB, OC, IC, IH, IW, KH, KW, SH, SW, PH, PW, weight});

    binWeights.reserve(OC*IC*KH*KW);
  }

  if (changeWeights) {
    changeWeights = false;
    xnor_nn_conv->change_weights(weight);
  }

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    // TODO: use allocated memory
    // TODO: check that aligned instructions generated on arm
    xnor_nn_conv->forward(bottom_data, top_data);
  }


  // Gemm implementation
#if 0
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
#endif
}

// Backward remains simple convolution on float via gemm using binWeights
template <typename Dtype>
void XnorNNConvolutionLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bweight = binWeights.data();

  // Binarize weights
  xnor_nn_conv->binarizeWeightsFloat(weight, bweight);

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          changeWeights = true;
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, bweight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(XnorNNConvolutionLayer);
#endif

INSTANTIATE_CLASS(XnorNNConvolutionLayer);

}  // namespace caffe

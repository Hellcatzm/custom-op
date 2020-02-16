#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <stdio.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "maximum_filter.h"


namespace tensorflow {
namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// CPU specialization of actual computation.
template <typename T>
struct MaximumFilterFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, 
                  const T* input_tensor, 
                  const int* footprint, 
                  T* output_tensor,
                  int batch_size,
                  int input_shape,
                  int footprint_shape) {
    // 优化思路：舍弃边界，利用offset，外面三层循环可以合并，里面两层循环可以合并
    for (int n=0; n<batch_size; ++n) {
      for (int x=0; x<input_shape; ++x) {
        for (int y=0; y<input_shape; ++y) {
          int input_idx = n*pow(input_shape, 2) + x*input_shape + y;
          int tmp = input_tensor[input_idx];
          for (int f_x=-footprint_shape/2; f_x<=footprint_shape/2; ++f_x) {
            int cur_x = x + f_x;
            if (cur_x<0 || cur_x>=input_shape) continue;
            int fp_x = f_x + footprint_shape/2;
            for (int f_y=-footprint_shape/2; f_y<=footprint_shape/2; ++f_y) {
              int cur_y = y + f_y;
              if (cur_y<0 || cur_y>=input_shape) continue;
              int fp_y = f_y + footprint_shape/2;
              int cur_input_idx = n*pow(input_shape, 2) + cur_x*input_shape + cur_y;
              if (input_tensor[cur_input_idx] > tmp \
                  && footprint[footprint_shape*fp_x + fp_y])
                tmp = input_tensor[cur_input_idx];
            }
          }
          output_tensor[input_idx] = tmp;
        }
      }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class MaximumFilterOp : public OpKernel {
 private:
  int crop_size;

 public:
  explicit MaximumFilterOp(OpKernelConstruction* context) : OpKernel(context) {};

  void Compute(OpKernelContext* context) override {
    // Grab input Tensors
    const Tensor& input_tensor = context->input(0);
    const Tensor& footprint = context->input(1);

    // Create output Tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    
    // Do the computation
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
      errors::InvalidArgument("Too many elements in tensor"));
    OP_REQUIRES(context, footprint.NumElements() <= tensorflow::kint32max,
      errors::InvalidArgument("Too many elements in tensor"));

    MaximumFilterFunctor<Device, T>()(
      context->eigen_device<Device>(),
      input_tensor.flat<T>().data(),         // [n, 256, 256]
      footprint.tensor<int, 2>().data(),     // [f_size, f_size]
      output_tensor->flat<T>().data(),       // [n, 256, 256]
      input_tensor.dim_size(0),
      input_tensor.dim_size(1),
      footprint.dim_size(0)
    );
  }
};
// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MaximumFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MaximumFilterOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct MaximumFilterFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MaximumFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      MaximumFilterOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}  // namespace functor
}  // namespace tensorflow

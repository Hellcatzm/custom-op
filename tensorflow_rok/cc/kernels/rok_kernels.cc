#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <stdio.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "rok.h"


namespace tensorflow {
namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// CPU specialization of actual computation.
template <typename T>
struct ROKFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, 
                  const float* images, 
                  const int* coords, 
                  float* regions,
                  int image_shape,
                  int region_num,
                  int region_shape) {
    // loop regions
    for (int reg_z=0; reg_z<region_num; ++reg_z) {
      int co_z = coords[reg_z];
      int co_x_lt = coords[region_num + reg_z] - region_shape/2;
      int co_y_lt = coords[region_num*2 + reg_z] - region_shape/2;
      // loop region position
      for (int reg_x=0; reg_x<region_shape; ++reg_x) {
        for (int reg_y=0; reg_y<region_shape; ++reg_y) {
          int co_x = co_x_lt + reg_x;
          int co_y = co_y_lt + reg_y;
          int idx_reg = pow(region_shape, 2) * reg_z + region_shape * reg_x + reg_y;
          if (co_x<0 || co_y<0)
            regions[idx_reg] = 0.;
          else {
            int idx_img = pow(image_shape, 2) * co_z + image_shape * co_x + co_y;
            regions[idx_reg] = images[idx_img];
          }
        }
      }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class ROKOp : public OpKernel {
 private:
  int crop_size;

 public:
  explicit ROKOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
            context->GetAttr("crop_size", &crop_size));
  }

  void Compute(OpKernelContext* context) override {
    // Grab input Tensors
    const Tensor& images = context->input(0);
    const Tensor& coords = context->input(1);

    // Create output Tensor
    Tensor* regions = NULL;
    OP_REQUIRES_OK(context, 
                   context->allocate_output(0, {coords.dim_size(1), crop_size, crop_size}, &regions));
    
    // Do the computation
    OP_REQUIRES(context, images.NumElements() <= tensorflow::kint32max,
        errors::InvalidArgument("Too many elements in tensor"));
    OP_REQUIRES(context, coords.NumElements() <= tensorflow::kint32max,
        errors::InvalidArgument("Too many elements in tensor"));

    ROKFunctor<Device, T>()(
        context->eigen_device<Device>(),
        images.flat<float>().data(),     // [n, 256, 256]
        coords.tensor<int, 2>().data(),  // [3, num_k]
        regions->flat<float>().data(),
        images.dim_size(1),
        coords.dim_size(1),
        crop_size
    );
  }
};
// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(  \
    Name("ROK").Device(DEVICE_CPU),  \
    ROKOp<CPUDevice, int32>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
extern template struct ROKFunctor<GPUDevice, int32>;
REGISTER_KERNEL_BUILDER(  \
    Name("ROK").Device(DEVICE_GPU),  \
    ROKOp<GPUDevice, int32>);
#endif  // GOOGLE_CUDA

}  // namespace functor
}  // namespace tensorflow

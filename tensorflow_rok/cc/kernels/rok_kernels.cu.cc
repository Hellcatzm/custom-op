#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "rok.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
// #include <math.h>
#include <stdio.h>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
__global__ void ROKCudaKernel(const float* images, const int* coords, float* regions,
                              int image_shape, int region_num, int region_shape) {
  int thr_x = blockIdx.x * blockDim.x + threadIdx.x;
  int thr_y = blockIdx.y * blockDim.y + threadIdx.y;
  int thr_z = blockIdx.z;
  
  // thread vaildate
  if (thr_x<1 && thr_y<1 && thr_z<region_num) {
    int co_z = coords[thr_z];
    int co_x_lt = coords[region_num + thr_z] - region_shape/2;    // left-top coord
    int co_y_lt = coords[region_num*2 + thr_z] - region_shape/2;  // left-top coord
    // loop region position
    for (int reg_x=0; reg_x<region_shape; ++reg_x) {
      for (int reg_y=0; reg_y<region_shape; ++reg_y) {
        int co_x = co_x_lt + reg_x;
        int co_y = co_y_lt + reg_y;
        int idx_reg = pow(region_shape, 2) * thr_z + region_shape * reg_x + reg_y;
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

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct ROKFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, 
                  const float* images, 
                  const int* coords, 
                  float* regions,
                  int image_shape,
                  int region_num,
                  int region_shape) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing kernel config.
    dim3 threads(32, 32);
    dim3 blocks((region_shape+threads.x-1)/threads.x, 
                (region_shape+threads.y-1)/threads.y, 
                region_num);

    ROKCudaKernel  \
        <<<blocks, threads, 0, d.stream()>>>(images, coords, regions,
                                             image_shape, region_num, region_shape);
  }
};

// Explicitly instantiate functors
template struct ROKFunctor<GPUDevice, int32>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
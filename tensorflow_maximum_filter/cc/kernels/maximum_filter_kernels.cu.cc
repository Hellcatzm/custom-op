#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "maximum_filter.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
// #include <math.h>
#include <stdio.h>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void MaximumFilterCudaKernel(const T* input_tensor, const int* footprint, 
                                        T* output_tensor, int batch_size, 
                                        int input_shape, int footprint_shape) {
  // get thread index
  int thr_x = blockIdx.x * blockDim.x + threadIdx.x;
  int thr_y = blockIdx.y * blockDim.y + threadIdx.y;
  int thr_z = blockIdx.z * blockDim.z + threadIdx.z;
  
  // thread vaildate
  if (thr_x<input_shape && thr_y<input_shape && thr_z<batch_size) {
    int input_idx = pow(input_shape, 2) * thr_z + input_shape * thr_x + thr_y;
    T tmp = input_tensor[input_idx];
    // loop footprint position
    for (int f_x=-footprint_shape/2; f_x<=footprint_shape/2; ++f_x) {
      int cur_x = thr_x + f_x;
      if (cur_x<0 || cur_x>=input_shape) continue;
      int fp_x = f_x + footprint_shape/2;
      for (int f_y=-footprint_shape/2; f_y<=footprint_shape/2; ++f_y) {
        int cur_y = thr_y + f_y;
        if (cur_y<0 || cur_y>=input_shape) continue;
        int fp_y = f_y + footprint_shape/2;
        int cur_input_idx = thr_z*pow(input_shape, 2) + cur_x*input_shape + cur_y;
        if (input_tensor[cur_input_idx]>tmp && footprint[footprint_shape*fp_x+fp_y])
          tmp = input_tensor[cur_input_idx];
      }
    }
    output_tensor[input_idx] = tmp;
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct MaximumFilterFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, 
                  const T* input_tensor, 
                  const int* footprint, 
                  T* output_tensor,
                  int batch_size,
                  int input_shape,
                  int footprint_shape) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing kernel config.
    dim3 threads(32, 32, 1);
    dim3 blocks((input_shape+threads.x-1)/threads.x, 
                (input_shape+threads.y-1)/threads.y, 
                batch_size);

    MaximumFilterCudaKernel<T>  \
        <<<blocks, threads, 0, d.stream()>>>(input_tensor, footprint, output_tensor,
                                             batch_size, input_shape, footprint_shape);
  }
};

// Explicitly instantiate functors
template struct MaximumFilterFunctor<GPUDevice, float>;
template struct MaximumFilterFunctor<GPUDevice, int32>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
#ifndef KERNEL_ROK_H_
#define KERNEL_ROK_H_

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct MaximumFilterFunctor {
  void operator()(const Device& d, 
                  const T* input_tensor, 
                  const int* footprint, 
                  T* output_tensor,
                  int batch_size,
                  int input_shape,
                  int footprint_shape);
};

}  // namespace functor
}  // namespace tensorflow

#endif //KERNEL_ROK_H_
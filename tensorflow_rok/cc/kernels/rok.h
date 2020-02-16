#ifndef KERNEL_ROK_H_
#define KERNEL_ROK_H_

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct ROKFunctor {
  void operator()(const Device& d, 
                  const float* images, 
                  const int* coords, 
                  float* regions,
                  int image_shape,
                  int region_num,
                  int region_shape);
};

}  // namespace functor
}  // namespace tensorflow

#endif //KERNEL_ROK_H_
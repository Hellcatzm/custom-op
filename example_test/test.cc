#include <stdio.h>
#include "tensorflow/core/framework/tensor.h"


int main(){
    int height=32, width=32, depth=3;
    tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height, width, depth}));
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    for (int y=0; y<height; ++y){
        for (int x=0; x<width; ++x){
            for (int c=0; c<depth; ++c){
                image_tensor_mapped(0, y, x, c) = 1;
            }
        }
    };
    std::cout<< image_tensor.NumElements() \
             << "\nend program" << std::endl;
    return 1;
};

// g++ -I/usr/local/lib/python2.7/dist-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -O2 -std=c++11 -o test test.cc -L/usr/local/lib/python2.7/dist-packages/tensorflow_core -l:libtensorflow_framework.so.2
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;

REGISTER_OP("MaximumFilter")
    .Attr("T: {float, int32}")
    .Input("input_tensor: T")
    .Input("footprint: int32")
    .Output("output_tensor: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

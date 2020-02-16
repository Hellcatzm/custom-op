#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;

REGISTER_OP("ROK")
    .Attr("crop_size: int = 5")
    .Input("images: float32")
    .Input("coords: int32")
    .Output("regions: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        int cs;
        c->GetAttr<int>("crop_size", &cs);
        std::vector<::tensorflow::shape_inference::DimensionHandle> output_dims;
        output_dims.push_back(c->Dim(c->input(1), 0));
        output_dims.push_back(c->MakeDim(cs));
        output_dims.push_back(c->MakeDim(cs));
        c->set_output(0, c->MakeShape(output_dims));
        return Status::OK();
    });

#ifndef SHAPE_INFERENCE_OP_INTERFACE
#define SHAPE_INFERENCE_OP_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to infer the returned shapes for operations.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for current op.", "void", "inferShapes">
  ];
}


#endif //SHAPE_INFERENCE_OP_INTERFACE

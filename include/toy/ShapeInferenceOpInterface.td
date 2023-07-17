#ifndef ShapeInferenceOpInterface
#define ShapeInferenceOpInterface

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let methods = [
    InterfaceMethod<"Infer output shape of current op.", "void", "inferShapes">
  ];
} 

#endif

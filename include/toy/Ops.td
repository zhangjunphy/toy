#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


//===----------------------------------------------------------------------===//
// Toy Dialect
//===----------------------------------------------------------------------===//

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}

class Toy_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Toy Ops
//===----------------------------------------------------------------------===//

def ConstantOp : Toy_Op<"Constant", [NoSideEffect]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value.
  }];

  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);

  let parser = [{ return ::parseConstantOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];

  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

  let verifier = [{ return ::verify(*this); }];
}

def AddOp : Toy_Op<"add", [NoSideEffect]> {
  let summary = "tensor addition";
  let description = [{
    Element-wise addtion of two compatible tensors.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let parser = [{ return ::parseBinaryOp(parser, result); }];
  let printer = [{ return ::printBinaryOp(p, *this); }];

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def MulOp : Toy_Op<"mul", [NoSideEffect]> {
  let summary = "tensor multiplication";
  let description = [{
    Element-wise multiplication.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let parser = [{ return ::parseBinaryOp(parser, result); }];
  let printer = [{ return ::printBinaryOp(p, *this); }];

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def TransposeOp : Toy_Op<"transpose", [NoSideEffect]> {
  let summary = "tensor transpose";
  let description = [{
    Transpose a tensor.
  }];

  let arguments = (ins F64Tensor:$lhs);
  let results = (outs F64Tensor);

  let hasCanonicalizer = 1;
}


def ReshapeOp : Toy_Op<"reshape", [NoSideEffect]> {
  let summary = "tensor reshape";
  let description = [{
    Transform the tensor into a different shape.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs StaticShapeTensorOf<[F64]>);
}

def ReturnOp : Toy_Op<"return", [NoSideEffect, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    Represent a return operation in a function.
  }];

  let arguments = (ins Variadic<F64Tensor>:$input);

  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let verifier = [{ return ::verify(*this); }];
}

def GenericCallOp : Toy_Op<"call"> {
  let summary = "generic function call";
  let description = [{
    Generic function call
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$input);
  let results = (outs F64Tensor);
}

def PrintOp : Toy_Op<"print", [HasParent<"FuncOp">]> {
  let summary = "print operation";
  let description = [{
    Print contents of a tensor.
  }];

  let arguments = (ins F64Tensor:$input);
}

#endif //TOY_OPS

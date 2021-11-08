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
      build($_builder, &_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

  let verifier = [{ return ::verifier(*this); }];
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

#endif //TOY_OPS

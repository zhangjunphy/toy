#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Toy_Dialect : Dialect {
    let name = "toy";
    let summary = "Dialect for the toy language";
    let description = [{Toy language.}];

    let cppNamespace = "toy";
}


class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

def ConstantOp : Toy_Op<"constant"> {
    let summary = "constant operation";
    let description = [{
        Constant operation turns a literal into an SSA value.
    }];

    let builders = [
        OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
          build(builder, result, value.getType(), value);
        }]>,

        OpBuilder<(ins "double":$value)>
    ];

    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor);
    let hasVerifier = 1;
}

#endif

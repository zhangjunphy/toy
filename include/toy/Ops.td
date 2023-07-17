#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

include "toy/ShapeInferenceOpInterface.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
  let summary = "Dialect for the toy language";
  let description = [{Toy language.}];
}


class Toy_Op<string mnemonic, list<Trait> traits = []> :
  Op<Toy_Dialect, mnemonic, traits>;

def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value.
  }];

  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def AddOp : Toy_Op<"add", [DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise addtion";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def FuncOp : Toy_Op<"func", [
  FunctionOpInterface, IsolatedFromAbove,
  DeclareOpInterfaceMethods<CallableOpInterface>
]> {
  let summary = "user defined function";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def GenericCallOp :Toy_Op<"generic_cal", [
  DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call";

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

def MulOp : Toy_Op<"mul", [DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "eltwise multiplication";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def PrintOp : Toy_Op<"print"> {
  let summary = "print";

  let arguments = (ins F64Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReshapeOp : Toy_Op<"reshape"> {
  let summary = "tensor reshape";

  let arguments = (ins F64Tensor:$input);
  let results = (outs StaticShapeTensorOf<[F64]>);
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  let hasCanonicalizer = 1;
}

def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "return";

  let arguments = (ins Variadic<F64Tensor>:$input);
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}

def TransposeOp : Toy_Op<"transpose", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def CastOp : Toy_Op<"cast", [
  DeclareOpInterfaceMethods<CastOpInterface>,
  Pure, SameOperandsAndResultShape,
  DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "shape cast";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
} 

#endif

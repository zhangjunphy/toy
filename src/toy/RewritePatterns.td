#ifndef OptimizationPatterns
#define OptimizationPatterns

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;

def ReshapeReshapeOptPattern : Pat<(ReshapeOp:$res (ReshapeOp $arg)), (ReshapeOp $arg), [(TypesAreIdentical $res, $arg)]>;

def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;

def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

#endif

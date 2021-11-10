#ifndef TOY_OPTIMIZE
#define TOP_OPTIMIZE

include "toy/Ops.td"


def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

// def ReshapeConstant :
//   NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
// def FoldConstantReshapeOptPattern : Pat<
//   (ReshapeOp:$res (ConstantOp $arg)),
//   (ConstantOp (ReshapeConstant $arg, $res))>;

#endif //TOP_OPTIMIZE

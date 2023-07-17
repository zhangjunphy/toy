#pragma once

#include "mlir/Transforms/InliningUtils.h"
#include "toy/Dialect.h"

namespace mlir {
namespace toy {

struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToReplace) const final {
    auto returnOp = cast<ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToReplace.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToReplace[it.index()].replaceAllUsesWith(it.value());
    }
  }

  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};

} // namespace toy
} // namespace mlir

#include "toy/Dialect.h"

#include "mlir/Pass/Pass.h"

#include "toy/ShapeInterfaceOpInterface.cc.inc"

namespace mlir {
namespace toy {

void AddOp::inferShapes() { getResult().setType(getOperand(0).getType()); }
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
void TransposeOp::inferShapes() {
  auto inType = getOperand().getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims(llvm::reverse(inType.getShape()));
  getResult().setType(RankedTensorType::get(dims, inType.getElementType()));
}
void CastOp::inferShapes() { getResult().setType(getOperand().getType()); }

class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    auto f = getOperation();
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;

    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    while (!opWorklist.empty()) {
      auto next = llvm::find_if(opWorklist, allOperandsInfered);
      if (next == opWorklist.end()) {
        break;
      }

      Operation *op = *next;
      opWorklist.erase(op);

      if (auto shape = dyn_cast<ShapeInference>(op)) {
        shape.inferShapes();
      } else {
        op->emitError("Shape inference not implemented for op " +
                      op->getName().getStringRef());
        return signalPassFailure();
      }
    }

    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed.");
      signalPassFailure();
    }
  }

  static bool allOperandsInfered(Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
      return operandType.isa<RankedTensorType>();
    });
  }

  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      return !resultType.isa<RankedTensorType>();
    });
  }
};

std::unique_ptr<mlir::Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

} // namespace toy
} // namespace mlir

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Support/LLVM.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#include <memory>

#define DEBUG_TYPE "shape-inference"

namespace mlir {
namespace toy {
#include "toy/ShapeInferenceOpInterface.cpp.inc"
}
} // namespace mlir

namespace {

class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, mlir::FunctionPass> {

  void runOnFunction() override {
    mlir::FuncOp f = getFunction();

    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    while (!opWorklist.empty()) {
      auto nextop = llvm::find_if(opWorklist, allOperandsInfered);
      if (nextop == opWorklist.end())
        break;

      mlir::Operation *op = *nextop;
      opWorklist.erase(op);

      LLVM_DEBUG(llvm::dbgs() << "Inferring shapes for: " << *op << "\n");
      if (auto shapeOp = mlir::dyn_cast<mlir::toy::ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation");
        return signalPassFailure();
      }
    }
  }

  static bool allOperandsInfered(mlir::Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](mlir::Type operandType) {
      return operandType.isa<mlir::RankedTensorType>();
    });
  }

  static bool returnsDynamicShape(mlir::Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](mlir::Type resultType) {
      return !resultType.isa<mlir::RankedTensorType>();
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

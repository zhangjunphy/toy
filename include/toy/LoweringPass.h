#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace toy {
std::unique_ptr<mlir::Pass> createLoweringPass();
std::unique_ptr<mlir::Pass> createLoweringToLLVMPass();
} // namespace toy
} // namespace mlir

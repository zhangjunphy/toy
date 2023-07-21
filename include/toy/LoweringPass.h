#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace toy {
std::unique_ptr<mlir::Pass> createLoweringToyToAffinePass();
std::unique_ptr<mlir::Pass> createLoweringAffineToLLVMPass();
} // namespace toy
} // namespace mlir

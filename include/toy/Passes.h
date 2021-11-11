#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>
namespace mlir {
namespace toy {

std::unique_ptr<Pass> createShapeInferencePass();

} // namespace toy
} // namespace mlir

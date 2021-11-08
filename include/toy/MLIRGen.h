#pragma once

#include <memory>

namespace mlir {
  class MLIRContext;
  class OwningModuleRef;
}

namespace toy {
  class ModuleAST;

  mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
}

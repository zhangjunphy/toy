add_mlir_dialect_library(MLIRToy
  Dialect.cpp

  DEPENDS
  ToyOpsIncGen

  LINK_COMPONENTS
  Core)

add_mlir_dialect_library(MLIRToy
  Dialect.cpp

  DEPENDS
  OpsIncGen

  LINK_COMPONENTS
  Core)

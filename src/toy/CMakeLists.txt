add_mlir_dialect_library(MLIRToy
  Dialect.cc

  DEPENDS
  OpsIncGen

  LINK_COMPONENTS
  Core)

set(LLVM_TARGET_DEFINITIONS RewritePatterns.td)
mlir_tablegen(RewritePatterns.cc.inc -gen-rewriters)
add_public_tablegen_target(RewritePatternsGen)

add_mlir_dialect_library(MLIRToy
  Dialect.cc
  RewritePatterns.cc
  ShapeInferencePass.cc
  LoweringPass.cc

  DEPENDS
  OpsIncGen
  RewritePatternsGen

  LINK_COMPONENTS
  Core)

set(LLVM_TARGET_DEFINITIONS mlir/Optimize.td)
mlir_tablegen(Optimize.inc -gen-rewriters)
add_public_tablegen_target(ToyOptimizeIncGen)

add_library(libtoy
  parser/AST.cpp
  mlir/Dialect.cpp
  mlir/MLIRGen.cpp
  mlir/Optimize.cpp
  mlir/ShapeInferencePass.cpp)
llvm_update_compile_flags(libtoy)
add_dependencies(libtoy ToyOpsIncGen ToyOptimizeIncGen ToyShapeInferenceOpInterfaceGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceOpInterface.td)
mlir_tablegen(ShapeInterfaceOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInterfaceOpInterface.cc.inc -gen-op-interface-defs)
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cc.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cc.inc -gen-dialect-defs)
add_public_tablegen_target(OpsIncGen)

# add_mlir_dialect(ToyOps toy)
# add_mlir_doc(ToyDialect ToyDialect toy/ -gen-dialect-doc)
# add_mlir_doc(ToyOps ToyOps toy/ -gen-op-doc)

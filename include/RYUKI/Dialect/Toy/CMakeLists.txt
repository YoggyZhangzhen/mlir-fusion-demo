set(LLVM_TARGET_DEFINITIONS ToyOps.td)

mlir_tablegen(ToyOps.h.inc      -gen-op-decls)
mlir_tablegen(ToyOps.cpp.inc    -gen-op-defs)
mlir_tablegen(ToyDialect.h.inc  -gen-dialect-decls  -dialect=toy)
mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs  -dialect=toy)
mlir_tablegen(ToyTypes.h.inc    -gen-typedef-decls  -dialect=toy)

add_public_tablegen_target(RYUKIToyOpsIncGen)

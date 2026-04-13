get_property(dialect_libs    GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(ryuki-opt ryuki-opt.cpp)
llvm_update_compile_flags(ryuki-opt)

target_link_libraries(ryuki-opt PRIVATE
  RYUKIToyDialect
  RYUKITransforms

  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms

  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRMemRefDialect

  ${dialect_libs}
  ${conversion_libs}
)

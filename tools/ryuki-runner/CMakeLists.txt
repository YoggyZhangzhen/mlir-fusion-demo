get_property(dialect_libs    GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs  GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(ryuki-runner ryuki-runner.cpp)
llvm_update_compile_flags(ryuki-runner)

target_link_libraries(ryuki-runner PRIVATE
  # ── 自定义库 ──────────────────────────────────────────────────────────
  RYUKIToyDialect
  RYUKITransforms

  # ── MLIR 核心 ─────────────────────────────────────────────────────────
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms

  # ── Dialect 库 ────────────────────────────────────────────────────────
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationPipelines
  MLIRBufferizationTransforms

  # ── Bufferization 外部模型（OneShotBufferize 需要）────────────────────
  MLIRArithTransforms        # arith bufferizable op interface
  MLIRLinalgTransforms       # linalg bufferizable op interface
  MLIRSCFTransforms          # scf bufferizable op interface
  MLIRTensorTransforms       # tensor bufferizable op interface

  # ── Conversion Pass 库 ────────────────────────────────────────────────
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRLinalgToStandard
  MLIRReconcileUnrealizedCasts

  # ── LLVM IR 导出 ──────────────────────────────────────────────────────
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration

  # ── LLVM Core ─────────────────────────────────────────────────────────
  LLVMCore
  LLVMSupport

  # ── 通配兜底 ──────────────────────────────────────────────────────────
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
)

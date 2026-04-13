//===- ryuki-runner.cpp - 完整 Lowering Driver ────────────────────────────===//
//
// 精简版 mlir-cpu-runner 风格 Driver，完成以下完整流水线：
//
//  .mlir 输入
//    │
//    ▼ [Stage 1] Tensor 级优化（Fusion）
//    │  ├─ ryuki-matmul-bias-fusion     自定义融合 Pass
//    │  ├─ linalg-generalize-named-ops  named op → linalg.generic
//    │  └─ canonicalize + cse
//    │
//    ▼ [Stage 2] Bufferization（Tensor → MemRef）
//    │  └─ one-shot-bufferize + expand-strided-metadata + canonicalize
//    │
//    ▼ [Stage 3] Loop 层降级（Linalg → Affine → SCF）
//    │  ├─ convert-linalg-to-affine-loops
//    │  └─ lower-affine
//    │
//    ▼ [Stage 4] 降级到 LLVM Dialect
//    │  ├─ convert-scf-to-cf
//    │  ├─ convert-arith-to-llvm
//    │  ├─ convert-func-to-llvm
//    │  ├─ convert-cf-to-llvm
//    │  ├─ finalize-memref-to-llvm
//    │  └─ reconcile-unrealized-casts
//    │
//    ▼ [Stage 5（可选）] 导出 LLVM IR (.ll 文件)
//       translateModuleToLLVMIR
//
//===----------------------------------------------------------------------===//

// ── MLIR 核心 ────────────────────────────────────────────────────────────────
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

// ── Dialects（注册用）────────────────────────────────────────────────────────
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// ── BufferizableOpInterface 外部模型 ──────────────────────────────────────────
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

// ── 所有 conversion/dialect passes（按名注册，供 parsePassPipeline 使用）────
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"

// ── LLVM IR 导出 ──────────────────────────────────────────────────────────────
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// ── LLVM Support ──────────────────────────────────────────────────────────────
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

// ── 自定义 Pass ───────────────────────────────────────────────────────────────
#include "RYUKI/Transforms/Passes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// 命令行参数
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> kInputFile(
    llvm::cl::Positional,
    llvm::cl::desc("<input .mlir file>"),
    llvm::cl::Required
);

static llvm::cl::opt<std::string> kOutputFile(
    "o",
    llvm::cl::desc("Output file ('-' for stdout)"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init("-")
);

static llvm::cl::opt<bool> kEmitLLVM(
    "emit-llvm",
    llvm::cl::desc("Emit LLVM IR (.ll) instead of MLIR text"),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> kNoFusion(
    "no-fusion",
    llvm::cl::desc("Skip the custom matmul+bias fusion pass"),
    llvm::cl::init(false)
);

enum class StopAfter {
  Fusion,
  Bufferize,
  Affine,
  LLVMDialect,
};

static llvm::cl::opt<StopAfter> kStopAfter(
    "stop-after",
    llvm::cl::desc("Stop the pipeline after a given stage"),
    llvm::cl::values(
        clEnumValN(StopAfter::Fusion,      "fusion",       "After fusion passes"),
        clEnumValN(StopAfter::Bufferize,   "bufferize",    "After bufferization"),
        clEnumValN(StopAfter::Affine,      "affine",       "After affine→SCF"),
        clEnumValN(StopAfter::LLVMDialect, "llvm-dialect", "After full LLVM lowering")
    ),
    llvm::cl::init(StopAfter::LLVMDialect)
);

static llvm::cl::opt<bool> kVerifyEach(
    "verify-each",
    llvm::cl::desc("Verify IR after each pass"),
    llvm::cl::init(false)
);

//===----------------------------------------------------------------------===//
// Dialect 注册
//===----------------------------------------------------------------------===//

static void registerAllDialects(DialectRegistry &registry) {
  registry.insert<
      func::FuncDialect,
      arith::ArithDialect,
      linalg::LinalgDialect,
      tensor::TensorDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      affine::AffineDialect,
      mlir::cf::ControlFlowDialect,
      bufferization::BufferizationDialect,
      LLVM::LLVMDialect
  >();

  // OneShotBufferize 所需的外部接口注册
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::
      registerBufferizableOpInterfaceExternalModels(registry);

  // LLVM IR 翻译接口
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
}

//===----------------------------------------------------------------------===//
// Pass Pipeline（用字符串管道，避免版本特定的 C++ 工厂函数）
//===----------------------------------------------------------------------===//

/// 用 parsePassPipeline 按名字添加 pass，兼容 LLVM 22+。
static LogicalResult addPipelineFromString(PassManager &pm,
                                           llvm::StringRef pipeline) {
  if (failed(mlir::parsePassPipeline(pipeline, pm))) {
    llvm::errs() << "Error: failed to parse pass pipeline:\n"
                 << "  " << pipeline << "\n";
    return failure();
  }
  return success();
}

static LogicalResult buildPipeline(PassManager &pm, StopAfter stopAt) {
  // ── Stage 1: Fusion ───────────────────────────────────────────────────────
  {
    std::string fusionPipeline = "func.func(";
    if (!kNoFusion)
      fusionPipeline += "ryuki-matmul-bias-fusion,";
    fusionPipeline +=
        "linalg-generalize-named-ops,"
        "canonicalize,"
        "cse"
        ")";
    if (failed(addPipelineFromString(pm, fusionPipeline)))
      return failure();
  }
  if (stopAt == StopAfter::Fusion) return success();

  // ── Stage 2: Bufferization ────────────────────────────────────────────────
  // one-shot-bufferize 参数说明：
  //   bufferize-function-boundaries=1  → 对函数边界的 tensor 参数也 bufferize
  if (failed(addPipelineFromString(pm,
      "one-shot-bufferize{bufferize-function-boundaries=1},"
      "func.func(expand-strided-metadata),"
      "canonicalize")))
    return failure();
  if (stopAt == StopAfter::Bufferize) return success();

  // ── Stage 3: Linalg → Affine → SCF ──────────────────────────────────────
  if (failed(addPipelineFromString(pm,
      "func.func("
      "  convert-linalg-to-affine-loops,"
      "  lower-affine"
      ")")))
    return failure();
  if (stopAt == StopAfter::Affine) return success();

  // ── Stage 4: SCF → CF → LLVM Dialect ────────────────────────────────────
  if (failed(addPipelineFromString(pm,
      "func.func(convert-scf-to-cf),"
      "convert-func-to-llvm,"
      "convert-cf-to-llvm,"
      "convert-arith-to-llvm,"
      "finalize-memref-to-llvm,"
      "reconcile-unrealized-casts")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// LLVM IR 导出
//===----------------------------------------------------------------------===//

static LogicalResult emitLLVMIR(ModuleOp module, llvm::raw_ostream &os) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    module.emitError(
        "Failed to translate to LLVM IR. "
        "Make sure all ops are lowered (try --stop-after=llvm-dialect first).");
    return failure();
  }
  llvmModule->print(os, nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  // ── 注册所有 Pass（包括内置 conversion pass，供 parsePassPipeline 使用）───
  DialectRegistry registry;
  registerAllDialects(registry);

  // 注册内置 passes（canonicalize / cse / conversion passes 等）
  mlir::registerTransformsPasses();
  mlir::registerConversionPasses();
  registerLinalgPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::memref::registerMemRefPasses();
  registerSCFPasses();
  mlir::affine::registerAffinePasses();

  // 注册自定义 Pass
  ryuki::registerRYUKITransformsPasses();

  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "RYUKI Lowering Driver\n"
      "  Reads .mlir → fusion + lowering → MLIR or LLVM IR\n\n"
      "Examples:\n"
      "  ryuki-runner input.mlir --stop-after=fusion -o -\n"
      "  ryuki-runner input.mlir --emit-llvm -o output.ll\n"
  );

  // ── MLIRContext ───────────────────────────────────────────────────────────
  MLIRContext context(registry);
  if (kVerifyEach)
    context.disableMultithreading();

  context.getDiagEngine().registerHandler([](Diagnostic &diag) {
    diag.print(llvm::errs());
    llvm::errs() << "\n";
    return success();
  });

  // ── 解析输入 .mlir 文件 ───────────────────────────────────────────────────
  llvm::SourceMgr sourceMgr;
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(kInputFile);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Error: cannot open '" << kInputFile
                 << "': " << ec.message() << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module =
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error: failed to parse '" << kInputFile << "'\n";
    return 1;
  }

  // ── 构建并运行 Pass Pipeline ──────────────────────────────────────────────
  PassManager pm(&context);
  pm.enableVerifier(kVerifyEach);

  StopAfter stopAt = kStopAfter;
  if (kEmitLLVM && stopAt != StopAfter::LLVMDialect) {
    llvm::errs() << "Note: --emit-llvm requires full lowering\n";
    stopAt = StopAfter::LLVMDialect;
  }

  if (failed(buildPipeline(pm, stopAt))) return 1;

  llvm::errs() << "Running pipeline on '" << kInputFile << "'...\n";
  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: pipeline failed\n";
    return 1;
  }
  llvm::errs() << "Pipeline succeeded.\n";

  // ── 写出结果 ──────────────────────────────────────────────────────────────
  std::string errMsg;
  auto outputFile = mlir::openOutputFile(kOutputFile, &errMsg);
  if (!outputFile) {
    llvm::errs() << "Error: " << errMsg << "\n";
    return 1;
  }

  if (kEmitLLVM) {
    if (failed(emitLLVMIR(*module, outputFile->os()))) return 1;
  } else {
    module->print(outputFile->os());
    outputFile->os() << "\n";
  }

  outputFile->keep();
  llvm::errs() << "Output: "
               << (kOutputFile.getValue() == "-" ? "stdout" : kOutputFile.getValue())
               << "\n";
  return 0;
}

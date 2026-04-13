//===- ryuki-opt.cpp - mlir-opt 风格的 Pass 调试工具 ─────────────────────===//
//
// 用途：手动组合 Pass，逐步观察 IR 变化。
// 适合开发期调试，不做完整 Lowering。
//
// 示例:
//   ryuki-opt --ryuki-matmul-bias-fusion input.mlir
//   ryuki-opt --ryuki-matmul-bias-fusion --linalg-generalize-named-ops input.mlir
//
//===----------------------------------------------------------------------===//

#include "RYUKI/Dialect/Toy/ToyDialect.h"
#include "RYUKI/Dialect/Toy/ToyOps.h"
#include "RYUKI/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // 注册所有要支持的 Dialect
  registry.insert<
      ryuki::toy::ToyDialect,
      mlir::func::FuncDialect,
      mlir::arith::ArithDialect,
      mlir::linalg::LinalgDialect,
      mlir::tensor::TensorDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect,
      mlir::affine::AffineDialect
  >();

  // 注册内置 Pass
  mlir::registerTransformsPasses();
  mlir::registerConversionPasses();
  mlir::registerLinalgPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::memref::registerMemRefPasses();
  mlir::registerSCFPasses();
  mlir::affine::registerAffinePasses();

  // 注册自定义 Pass
  ryuki::registerRYUKITransformsPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "RYUKI optimizer\n", registry));
}

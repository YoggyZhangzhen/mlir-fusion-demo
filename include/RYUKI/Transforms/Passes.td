#ifndef RYUKI_TRANSFORMS_PASSES
#define RYUKI_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Pass A: Matmul + Bias 融合（Named Op 方式，推荐）
//===----------------------------------------------------------------------===//
// 核心思路：linalg.matmul 语义是 C[i,j] += A[i,k]*B[k,j]
// 将 bias 作为 matmul 的 outs 初始值，天然得到 bias + A@B
//
// 匹配模式:
//   %C   = linalg.matmul ins(%A, %B) outs(%zeros_init)
//   %out = linalg.add   ins(%C, %bias) outs(%empty)
//
// 替换结果:
//   %out = linalg.matmul ins(%A, %B) outs(%bias)
def RYUKIMatmulBiasFusion
    : Pass<"ryuki-matmul-bias-fusion", "::mlir::func::FuncOp"> {
  let summary = "将 linalg.matmul + linalg.add(bias) 融合为单一 matmul";
  let constructor = "ryuki::createMatmulBiasFusionPass()";
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::func::FuncDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Pass B: Matmul + Bias 融合（显式 linalg.generic 方式，演示用）
//===----------------------------------------------------------------------===//
// 与 Pass A 相同的匹配逻辑，输出为手动构建的 linalg.generic，
// 展示 indexing_maps / iterator_types / region body 的完整构造流程
def RYUKIMatmulBiasGenericFusion
    : Pass<"ryuki-matmul-bias-generic-fusion", "::mlir::func::FuncOp"> {
  let summary = "将 linalg.matmul + linalg.add 融合为显式 linalg.generic";
  let constructor = "ryuki::createMatmulBiasGenericFusionPass()";
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::func::FuncDialect",
  ];
}

#endif // RYUKI_TRANSFORMS_PASSES

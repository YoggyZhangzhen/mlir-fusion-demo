//===- Passes.h - RYUKI Transform Pass 声明 ─────────────────────────────===//
#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace ryuki {

/// Pass A：使用 bias-as-init 技巧融合
/// 匹配 matmul(A,B,zeros) + add(result,bias) → matmul(A,B,bias)
std::unique_ptr<mlir::Pass> createMatmulBiasFusionPass();

/// Pass B：将 matmul + add 融合为显式 linalg.generic
/// 展示如何编程式构造 indexing_maps 和 region body
std::unique_ptr<mlir::Pass> createMatmulBiasGenericFusionPass();

/// 注册本项目所有 Transformation Passes
void registerRYUKITransformsPasses();

} // namespace ryuki

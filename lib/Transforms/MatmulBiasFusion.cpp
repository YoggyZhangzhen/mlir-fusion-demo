//===- MatmulBiasFusion.cpp - Matmul + Bias 算子融合 Pass ─────────────────===//
//
// 实现两种等价融合策略，将：
//   linalg.matmul(A, B, zeros) + linalg.add(result, bias)
// 变换为单一操作。
//
// 【策略 A】MatmulBiasAsInitPattern
//   利用 linalg.matmul 的累加语义：C[i,j] += A[i,k] * B[k,j]
//   将 zero-fill 的 outs 替换为 bias，得到 bias + A@B。
//   输出仍是 linalg.matmul，简洁高效。
//
// 【策略 B】MatmulBiasGenericPattern
//   手动构建 linalg.generic，展示 indexing_maps / iterator_types /
//   region body 的完整编程式构造流程。
//
// 匹配的 IR 模式:
//
//   %empty1 = tensor.empty() : tensor<MxNxf32>
//   %c0     = arith.constant 0.0 : f32
//   %zeros  = linalg.fill ins(%c0) outs(%empty1) -> tensor<MxNxf32>
//   %C      = linalg.matmul ins(%A, %B) outs(%zeros) -> tensor<MxNxf32>
//   %empty2 = tensor.empty() : tensor<MxNxf32>
//   %out    = linalg.add ins(%C, %bias) outs(%empty2) -> tensor<MxNxf32>
//
//===----------------------------------------------------------------------===//

#include "RYUKI/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ryuki-matmul-bias-fusion"

using namespace mlir;
using namespace mlir::linalg;

namespace ryuki {

//===----------------------------------------------------------------------===//
// 公共辅助函数
//===----------------------------------------------------------------------===//

/// 判断 tensor value 是否来自 linalg.fill(0)。
///
/// 要求调用链：
///   %c0     = arith.constant 0.0
///   %result = linalg.fill ins(%c0) outs(%empty) → %result  ← 传入此值
///
/// 只有原始 outs 为零时，替换为 bias 才等价于 A@B + bias。
static bool isZeroFill(Value initTensor) {
  auto fillOp = initTensor.getDefiningOp<linalg::FillOp>();
  if (!fillOp)
    return false;

  // 新版 MLIR 中 FillOp 遵循 ins/outs 接口，ins[0] 是填充标量
  Value fillScalar = fillOp.getInputs()[0];
  auto constOp = fillScalar.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;

  if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue()))
    return floatAttr.getValue().isZero();

  if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getValue().isZero();

  return false;
}

/// 在 linalg.add 的两个输入中找到 matmul 结果和 bias。
///
/// 加法具有交换律，matmul 结果可能是 inputs[0] 或 inputs[1]。
/// 约束：matmul 结果只能有一个使用者（即这个 add），
/// 否则中间张量 %C 不能被安全消除。
static std::pair<linalg::MatmulOp, Value>
findMatmulAndBias(linalg::AddOp addOp) {
  auto inputs = addOp.getInputs();
  if (inputs.size() != 2)
    return {nullptr, {}};

  for (auto [idx, input] : llvm::enumerate(inputs)) {
    auto matmulOp = input.getDefiningOp<linalg::MatmulOp>();
    if (!matmulOp)
      continue;
    if (!input.hasOneUse()) // matmul 结果必须只被这一个 add 消费
      continue;
    Value bias = inputs[1 - idx];
    return {matmulOp, bias};
  }
  return {nullptr, {}};
}

//===----------------------------------------------------------------------===//
// 策略 A：MatmulBiasAsInitPattern
//===----------------------------------------------------------------------===//
//
// 变换逻辑：
//   linalg.matmul 的语义：for i,j: C[i,j] = C_init[i,j]; for k: C[i,j] += A*B
//   当 C_init = bias 时：C[i,j] = bias[i,j] + sum_k(A[i,k]*B[k,j])
//
// 消除中间张量 %C（MxN 大小），减少一次完整内存往返。
//
struct MatmulBiasAsInitPattern : public OpRewritePattern<linalg::AddOp> {
  using OpRewritePattern<linalg::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();

    // ── 步骤 1：在 add 输入中寻找 matmul + bias ──────────────────────────
    auto [matmulOp, bias] = findMatmulAndBias(addOp);
    if (!matmulOp) {
      LLVM_DEBUG(dbgs() << "[AsInit] No eligible matmul found\n");
      return failure();
    }

    // ── 步骤 2：验证 matmul 的 outs 初始化为零 ────────────────────────────
    if (!isZeroFill(matmulOp.getOutputs()[0])) {
      LLVM_DEBUG(dbgs() << "[AsInit] Matmul outs is not zero-fill, skip\n");
      return failure();
    }

    // ── 步骤 3：类型检查 ──────────────────────────────────────────────────
    auto matmulResultType =
        cast<RankedTensorType>(matmulOp.getResult(0).getType());
    auto biasType = cast<RankedTensorType>(bias.getType());
    if (matmulResultType != biasType) {
      LLVM_DEBUG(dbgs() << "[AsInit] Type mismatch, skip\n");
      return failure();
    }

    // ── 步骤 4：构建融合后的 matmul（bias 作为 outs init）────────────────
    //
    // Tensor 在 MLIR 中是不可变的！linalg.matmul 不会修改 bias，
    // 而是产生一个新的 tensor，值 = bias + A@B。
    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];

    auto fusedMatmul = rewriter.create<linalg::MatmulOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{matmulResultType},
        /*inputs=*/ValueRange{A, B},
        /*outputs=*/ValueRange{bias} // ← 核心：用 bias 替换零初始化
    );

    LLVM_DEBUG(dbgs() << "[AsInit] Fusion succeeded: " << fusedMatmul << "\n");

    // ── 步骤 5：替换 add 的所有结果引用并清理旧 Op ───────────────────────
    rewriter.replaceOp(addOp, fusedMatmul.getResults());
    rewriter.eraseOp(matmulOp);
    // linalg.fill 和 tensor.empty 由 DCE / canonicalize 清理

    return success();
  }
};

//===----------------------------------------------------------------------===//
// 策略 B：MatmulBiasGenericPattern
//===----------------------------------------------------------------------===//
//
// 生成的 linalg.generic：
//
//   %out = linalg.generic {
//     indexing_maps = [
//       affine_map<(i, j, k) -> (i, k)>,  // A
//       affine_map<(i, j, k) -> (k, j)>,  // B
//       affine_map<(i, j, k) -> (i, j)>,  // out（bias 作为累加器 init）
//     ],
//     iterator_types = ["parallel", "parallel", "reduction"]
//   } ins(%A, %B) outs(%bias) {
//   ^bb0(%a: f32, %b: f32, %acc: f32):
//     %mul     = arith.mulf %a, %b     : f32
//     %new_acc = arith.addf %acc, %mul : f32
//     linalg.yield %new_acc            : f32
//   } -> tensor<MxNxf32>
//
struct MatmulBiasGenericPattern : public OpRewritePattern<linalg::AddOp> {
  using OpRewritePattern<linalg::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // ── 步骤 1 & 2：与策略 A 相同的匹配 + 验证 ──────────────────────────
    auto [matmulOp, bias] = findMatmulAndBias(addOp);
    if (!matmulOp || !isZeroFill(matmulOp.getOutputs()[0]))
      return failure();

    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];

    auto typeA   = cast<RankedTensorType>(A.getType());
    auto typeOut = cast<RankedTensorType>(bias.getType());
    Type elemType = typeA.getElementType();

    // ── 步骤 3：构建 Indexing Maps ────────────────────────────────────────
    //
    // 循环结构：
    //   d0 = i（M 维，parallel）
    //   d1 = j（N 维，parallel）
    //   d2 = k（K 归约维，reduction）
    //
    // Operand 访问模式：
    //   A[i, k]   → affine_map<(d0,d1,d2) -> (d0,d2)>
    //   B[k, j]   → affine_map<(d0,d1,d2) -> (d2,d1)>
    //   out[i, j] → affine_map<(d0,d1,d2) -> (d0,d1)>（归约后写回）
    constexpr unsigned kNumLoops = 3;
    AffineExpr d0 = rewriter.getAffineDimExpr(0); // i
    AffineExpr d1 = rewriter.getAffineDimExpr(1); // j
    AffineExpr d2 = rewriter.getAffineDimExpr(2); // k

    SmallVector<AffineMap, 3> indexingMaps = {
        AffineMap::get(kNumLoops, 0, {d0, d2}, ctx), // A[i, k]
        AffineMap::get(kNumLoops, 0, {d2, d1}, ctx), // B[k, j]
        AffineMap::get(kNumLoops, 0, {d0, d1}, ctx), // out[i, j]
    };

    // ── 步骤 4：构建 Iterator Types ───────────────────────────────────────
    // parallel：出现在所有 output map 中 → 可并行
    // reduction：不在 output map 中 → 需归约
    SmallVector<utils::IteratorType, 3> iteratorTypes = {
        utils::IteratorType::parallel,  // i
        utils::IteratorType::parallel,  // j
        utils::IteratorType::reduction, // k
    };

    // ── 步骤 5：创建 linalg.generic Op ───────────────────────────────────
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{typeOut},
        /*inputs=*/ValueRange{A, B},
        /*outputs=*/ValueRange{bias}, // bias 作为累加器 init
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*doc=*/"ryuki_matmul_bias_fused",
        /*libraryCall=*/StringRef{}
    );

    // ── 步骤 6：填充 Region Body ─────────────────────────────────────────
    //
    // block 参数对应每个 operand 的标量元素，顺序：[inputs..., outputs...]
    //   arg0: f32 ← A[i, k]
    //   arg1: f32 ← B[k, j]
    //   arg2: f32 ← out[i, j]（累加器，初值来自 bias）
    //
    // 计算：out[i,j] = bias[i,j] + sum_k(A[i,k] * B[k,j])
    Region &region = genericOp.getRegion();
    Block *body = rewriter.createBlock(
        &region, region.begin(),
        TypeRange{elemType, elemType, elemType},
        {loc, loc, loc}
    );

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(body);

    Value a   = body->getArgument(0); // A 的标量元素
    Value b   = body->getArgument(1); // B 的标量元素
    Value acc = body->getArgument(2); // 累加器（初值 = bias[i,j]）

    Value mul    = rewriter.create<arith::MulFOp>(loc, a, b);
    Value newAcc = rewriter.create<arith::AddFOp>(loc, acc, mul);
    rewriter.create<linalg::YieldOp>(loc, ValueRange{newAcc});

    // ── 步骤 7：替换旧 Op ─────────────────────────────────────────────────
    rewriter.replaceOp(addOp, genericOp.getResults());
    rewriter.eraseOp(matmulOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass 定义
//===----------------------------------------------------------------------===//

namespace {

struct MatmulBiasFusionPass
    : public PassWrapper<MatmulBiasFusionPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulBiasFusionPass)

  StringRef getArgument()    const final { return "ryuki-matmul-bias-fusion"; }
  StringRef getDescription() const final {
    return "Fuse linalg.matmul + linalg.add (bias) via bias-as-init trick";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    tensor::TensorDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<MatmulBiasAsInitPattern>(funcOp.getContext());

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("MatmulBiasFusionPass: rewrite failed");
      signalPassFailure();
    }
  }
};

struct MatmulBiasGenericFusionPass
    : public PassWrapper<MatmulBiasGenericFusionPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulBiasGenericFusionPass)

  StringRef getArgument()    const final { return "ryuki-matmul-bias-generic-fusion"; }
  StringRef getDescription() const final {
    return "Fuse linalg.matmul + linalg.add into an explicit linalg.generic";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    tensor::TensorDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<MatmulBiasGenericPattern>(funcOp.getContext());

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("MatmulBiasGenericFusionPass: rewrite failed");
      signalPassFailure();
    }
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// 工厂函数 & 注册
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createMatmulBiasFusionPass() {
  return std::make_unique<MatmulBiasFusionPass>();
}

std::unique_ptr<mlir::Pass> createMatmulBiasGenericFusionPass() {
  return std::make_unique<MatmulBiasGenericFusionPass>();
}

void registerRYUKITransformsPasses() {
  PassRegistration<MatmulBiasFusionPass>();
  PassRegistration<MatmulBiasGenericFusionPass>();
}

} // namespace ryuki

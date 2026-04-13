#!/usr/bin/env bash
# =============================================================================
# build_ryuki.sh - 构建 RYUKI 项目
#
# 用法:
#   chmod +x scripts/build_ryuki.sh
#   ./scripts/build_ryuki.sh
#
# 前提: 已运行 build_llvm.sh 或手动安装了 LLVM/MLIR
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

LLVM_INSTALL="${LLVM_INSTALL:-$HOME/llvm-install}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_JOBS="${BUILD_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}"

echo "================================================="
echo " RYUKI 项目构建脚本"
echo " 项目根目录: $PROJECT_ROOT"
echo " 构建目录:   $BUILD_DIR"
echo " LLVM 安装:  $LLVM_INSTALL"
echo " 构建类型:   $BUILD_TYPE"
echo "================================================="

# ── 检查 LLVM 是否存在 ─────────────────────────────────────────────────────
if [ ! -f "$LLVM_INSTALL/lib/cmake/mlir/MLIRConfig.cmake" ]; then
  echo "ERROR: 找不到 MLIR CMake 配置：$LLVM_INSTALL/lib/cmake/mlir/"
  echo ""
  echo "请先运行: ./scripts/build_llvm.sh"
  echo "或者设置: LLVM_INSTALL=/your/llvm/install ./scripts/build_ryuki.sh"
  exit 1
fi

# ── 创建构建目录 ───────────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR"

# ── CMake 配置 ────────────────────────────────────────────────────────────
echo ">>> 配置 CMake..."
cmake -G Ninja -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DLLVM_DIR="$LLVM_INSTALL/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_INSTALL/lib/cmake/mlir" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  # 给 clangd / VSCode 用

# ── 编译 ──────────────────────────────────────────────────────────────────
echo ">>> 编译..."
ninja -C "$BUILD_DIR" -j"$BUILD_JOBS"

# ── 验证二进制 ─────────────────────────────────────────────────────────────
echo ">>> 验证构建产物..."
for bin in ryuki-opt ryuki-runner; do
  if [ -f "$BUILD_DIR/bin/$bin" ]; then
    echo "  ✓ $bin"
  else
    echo "  ✗ $bin (构建失败)"
    exit 1
  fi
done

echo ""
echo "================================================="
echo " 构建成功！现在可以运行："
echo ""
echo "  # 查看融合后的 IR"
echo "  $BUILD_DIR/bin/ryuki-runner \\"
echo "    $PROJECT_ROOT/test/Toy/matmul_bias.mlir \\"
echo "    --stop-after=fusion -o -"
echo ""
echo "  # 导出 LLVM IR"
echo "  $BUILD_DIR/bin/ryuki-runner \\"
echo "    $PROJECT_ROOT/test/Toy/matmul_bias.mlir \\"
echo "    --emit-llvm -o /tmp/output.ll"
echo ""
echo "  # 查看汇编"
echo "  $LLVM_INSTALL/bin/llc -O2 /tmp/output.ll -o /tmp/output.s"
echo "  cat /tmp/output.s"
echo "================================================="

#!/usr/bin/env bash
# =============================================================================
# build_llvm.sh - 一键编译 LLVM + MLIR（macOS / Linux）
#
# 用法:
#   chmod +x scripts/build_llvm.sh
#   ./scripts/build_llvm.sh
#
# 环境变量（可覆盖默认值）:
#   LLVM_SRC_DIR   LLVM 源码路径（默认克隆到 ~/llvm-project）
#   LLVM_INSTALL   安装目标路径（默认 ~/llvm-install）
#   BUILD_JOBS     并行编译线程数（默认 CPU 核心数）
# =============================================================================

set -euo pipefail

LLVM_SRC_DIR="${LLVM_SRC_DIR:-$HOME/llvm-project}"
LLVM_INSTALL="${LLVM_INSTALL:-$HOME/llvm-install}"
BUILD_JOBS="${BUILD_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}"
LLVM_BUILD="$LLVM_SRC_DIR/build"

echo "================================================="
echo " LLVM + MLIR 构建脚本"
echo " 源码目录: $LLVM_SRC_DIR"
echo " 安装目录: $LLVM_INSTALL"
echo " 并行线程: $BUILD_JOBS"
echo "================================================="

# ── 1. 检查依赖 ────────────────────────────────────────────────────────────
for cmd in cmake ninja clang clang++; do
  if ! command -v $cmd &>/dev/null; then
    echo "ERROR: $cmd not found. 请先安装:"
    echo "  macOS: brew install cmake ninja llvm"
    echo "  Linux: apt install cmake ninja-build clang"
    exit 1
  fi
done

# ── 2. 克隆 LLVM（如果不存在）─────────────────────────────────────────────
if [ ! -d "$LLVM_SRC_DIR" ]; then
  echo ">>> 克隆 llvm-project（仅需一次，约 2GB）..."
  git clone --depth=1 https://github.com/llvm/llvm-project.git "$LLVM_SRC_DIR"
else
  echo ">>> LLVM 源码已存在，跳过克隆"
fi

# ── 3. 创建构建目录 ────────────────────────────────────────────────────────
mkdir -p "$LLVM_BUILD"

# ── 4. CMake 配置 ─────────────────────────────────────────────────────────
echo ">>> 配置 CMake..."
cmake -G Ninja -S "$LLVM_SRC_DIR/llvm" -B "$LLVM_BUILD" \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_ENABLE_RTTI=OFF \
  -DLLVM_ENABLE_EH=OFF \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# ── 5. 编译 ───────────────────────────────────────────────────────────────
echo ">>> 开始编译（约 30-60 分钟，使用 $BUILD_JOBS 线程）..."
ninja -C "$LLVM_BUILD" -j"$BUILD_JOBS"

# ── 6. 安装 ───────────────────────────────────────────────────────────────
echo ">>> 安装到 $LLVM_INSTALL ..."
ninja -C "$LLVM_BUILD" install

# ── 7. 验证 ───────────────────────────────────────────────────────────────
echo ">>> 验证安装..."
"$LLVM_INSTALL/bin/mlir-opt" --version

echo ""
echo "================================================="
echo " 构建完成！"
echo ""
echo " 接下来构建 RYUKI 项目："
echo ""
echo "   cd $(dirname "$0")/.."
echo "   mkdir build && cd build"
echo "   cmake .. -G Ninja \\"
echo "     -DLLVM_DIR=$LLVM_INSTALL/lib/cmake/llvm \\"
echo "     -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir \\"
echo "     -DCMAKE_BUILD_TYPE=Release \\"
echo "     -DCMAKE_C_COMPILER=clang \\"
echo "     -DCMAKE_CXX_COMPILER=clang++ \\"
echo "     -DLLVM_ENABLE_ASSERTIONS=ON"
echo "   ninja -j$BUILD_JOBS"
echo "================================================="

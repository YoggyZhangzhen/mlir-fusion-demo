# RYUKI 项目构建指南

## 前提条件

已编译 LLVM + MLIR 并安装到某个路径，例如 `$HOME/llvm-install`。

如果还没编译，参考以下最小命令：

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && mkdir build && cd build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install

ninja -j$(nproc) install
```

## 编译本项目

```bash
cd /path/to/RYUKI
mkdir build && cd build

cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=$HOME/llvm-install/lib/cmake/llvm \
  -DMLIR_DIR=$HOME/llvm-install/lib/cmake/mlir \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

ninja -j$(nproc)
```

调试构建（开发阶段推荐）：

```bash
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_DIR=$HOME/llvm-install/lib/cmake/llvm \
  -DMLIR_DIR=$HOME/llvm-install/lib/cmake/mlir \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## 使用工具

### ryuki-opt（调试用，单独测试 Pass）

```bash
# 仅运行融合 Pass
./bin/ryuki-opt --ryuki-matmul-bias-fusion ../test/Toy/matmul_bias.mlir

# 融合 + 泛化
./bin/ryuki-opt \
  --ryuki-matmul-bias-fusion \
  --linalg-generalize-named-ops \
  --canonicalize \
  ../test/Toy/matmul_bias.mlir
```

### ryuki-runner（完整 Lowering Driver）

```bash
# 完整降级，输出 MLIR LLVM Dialect
./bin/ryuki-runner ../test/Toy/matmul_bias.mlir \
  -o /tmp/output_llvm_dialect.mlir

# 在 fusion 阶段停止（观察融合效果）
./bin/ryuki-runner ../test/Toy/matmul_bias.mlir \
  --stop-after=fusion \
  -o /tmp/after_fusion.mlir

# 在 bufferize 阶段停止
./bin/ryuki-runner ../test/Toy/matmul_bias.mlir \
  --stop-after=bufferize \
  -o /tmp/after_bufferize.mlir

# 在 affine 阶段停止（看循环结构）
./bin/ryuki-runner ../test/Toy/matmul_bias.mlir \
  --stop-after=affine \
  -o /tmp/after_affine.mlir

# 导出 LLVM IR（.ll 文件）
./bin/ryuki-runner ../test/Toy/matmul_bias.mlir \
  --emit-llvm \
  -o /tmp/output.ll

# 调试模式：每个 Pass 后验证 + 打印耗时
./bin/ryuki-runner ../test/Toy/matmul_bias.mlir \
  --verify-each \
  --enable-timing \
  --stop-after=affine \
  -o /tmp/debug.mlir

# 跳过融合（对比用）
./bin/ryuki-runner ../test/Toy/matmul_bias.mlir \
  --no-fusion \
  --emit-llvm \
  -o /tmp/no_fusion.ll
```

### 编译 LLVM IR 为汇编

```bash
# 用 llc 编译为 x86 汇编
llc -O2 -march=x86-64 /tmp/output.ll -o /tmp/output.s

# 或直接编译为可执行文件
clang /tmp/output.ll -o /tmp/output
```

### Python 脚本生成测试 MLIR

```bash
# 需要 mlir python bindings
export PYTHONPATH=$HOME/llvm-install/python_packages/mlir_core

cd tools
python gen_matmul_bias.py
# 生成 matmul_bias.mlir，可直接用 ryuki-runner 处理
```

## 项目结构

```
RYUKI/
├── CMakeLists.txt                        顶层 CMake
├── BUILD.md                              本文件
├── cmake/modules/RYUKIConfig.cmake       CMake 辅助模块
├── include/RYUKI/
│   ├── Dialect/Toy/
│   │   ├── ToyOps.td                     TableGen 定义
│   │   ├── ToyDialect.h                  Dialect 头文件
│   │   └── ToyOps.h                      Op 头文件
│   └── Transforms/
│       ├── Passes.td                     Pass TableGen 定义
│       └── Passes.h                      Pass 声明
├── lib/
│   ├── Dialect/Toy/
│   │   ├── ToyDialect.cpp                Dialect 注册
│   │   └── ToyOps.cpp                    Op 实现
│   └── Transforms/
│       └── MatmulBiasFusion.cpp          融合 Pass 实现
├── tools/
│   ├── ryuki-opt/ryuki-opt.cpp           调试工具（类 mlir-opt）
│   ├── ryuki-runner/ryuki-runner.cpp     完整 Lowering Driver
│   └── gen_matmul_bias.py               Python 脚本生成测试 MLIR
└── test/Toy/
    └── matmul_bias.mlir                  测试用 MLIR 文件
```

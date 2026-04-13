//===- ToyDialect.cpp - Toy Dialect 注册实现 ────────────────────────────===//

#include "RYUKI/Dialect/Toy/ToyOps.h"

using namespace mlir;
using namespace ryuki::toy;

//===----------------------------------------------------------------------===//
// Toy Dialect
//===----------------------------------------------------------------------===//

// TableGen 生成的 Dialect 定义（initialize() 等）
#include "RYUKI/Dialect/Toy/ToyDialect.cpp.inc"

void ToyDialect::initialize() {
  // 注册所有在 ToyOps.td 中定义的 Op
  addOperations<
#define GET_OP_LIST
#include "RYUKI/Dialect/Toy/ToyOps.cpp.inc"
  >();
}

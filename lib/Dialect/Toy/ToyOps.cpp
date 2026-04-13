//===- ToyOps.cpp - Toy Dialect Op 实现 ─────────────────────────────────===//

#include "RYUKI/Dialect/Toy/ToyOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace ryuki::toy;

// TableGen 生成的 Op 定义（verify、build、print/parse 等）
#define GET_OP_CLASSES
#include "RYUKI/Dialect/Toy/ToyOps.cpp.inc"

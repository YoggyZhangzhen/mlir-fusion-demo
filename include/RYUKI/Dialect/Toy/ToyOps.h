//===- ToyOps.h - Toy Dialect Op 声明 ───────────────────────────────────===//
#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// 先包含 Dialect 声明
#include "RYUKI/Dialect/Toy/ToyDialect.h"

// TableGen 生成的 Op 声明（来自 ToyOps.td → ToyOps.h.inc）
#define GET_OP_CLASSES
#include "RYUKI/Dialect/Toy/ToyOps.h.inc"

#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Toy Dialect
//===----------------------------------------------------------------------===//

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "RYUKI Toy dialect for learning purposes";
  let cppNamespace = "::ryuki::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []>
    : Op<Toy_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

// 常量 Op：持有一个 f64 tensor 常量值
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant operation";
  let arguments = (ins F64ElementsAttr:$value);
  let results   = (outs F64Tensor:$result);
  // 使用默认格式，避免 LLVM 22 对 type 可推导性的严格校验
  let hasCustomAssemblyFormat = 0;
}

// 加法 Op：两个 f64 tensor 逐元素相加
def AddOp : Toy_Op<"add", [Pure]> {
  let summary = "element-wise addition";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results   = (outs F64Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

#endif // TOY_OPS

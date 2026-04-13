"""
使用 MLIR Python Bindings 构建 Matmul + Bias Add 计算图。

依赖：
  使用 LLVM 编译出的 Python 包（保证版本一致）：
    pip install mlir-python-bindings
  或者设置 PYTHONPATH 指向 LLVM build 目录：
    export PYTHONPATH=$LLVM_BUILD/python_packages/mlir_core

运行：
  python gen_matmul_bias.py
  python gen_matmul_bias.py > matmul_bias.mlir
"""

from mlir import ir
from mlir.dialects import (
    builtin,
    func,
    linalg,
    tensor,
    arith,
)


def build_matmul_bias(
    M: int = 4,
    N: int = 8,
    K: int = 16,
) -> ir.Module:
    """
    构建如下语义的 MLIR 模块：

        result = matmul(A, B) + bias

    其中：
        A    : tensor<MxK x f32>   输入矩阵
        B    : tensor<KxN x f32>   权重矩阵
        bias : tensor<MxN x f32>   偏置（逐元素加）
        out  : tensor<MxN x f32>   输出

    计算分两步：
        step1 = linalg.matmul(A, B)       shape [M, N]
        step2 = linalg.add(step1, bias)   shape [M, N]（elementwise）
    """
    ctx = ir.Context()

    with ctx, ir.Location.unknown():
        module = ir.Module.create()

        dtype = ir.F32Type.get()

        # ── 定义 Tensor 类型 ───────────────────────────────────────────
        type_A  = ir.RankedTensorType.get([M, K], dtype)   # tensor<MxKxf32>
        type_B  = ir.RankedTensorType.get([K, N], dtype)   # tensor<KxNxf32>
        type_MN = ir.RankedTensorType.get([M, N], dtype)   # tensor<MxNxf32>

        # ── 函数签名：(A, B, bias) -> result ─────────────────────────
        func_type = ir.FunctionType.get(
            inputs=[type_A, type_B, type_MN],
            results=[type_MN],
        )

        with ir.InsertionPoint(module.body):
            fn = func.FuncOp(name="matmul_bias", type=func_type)
            fn.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

            with ir.InsertionPoint(fn.add_entry_block()):
                A    = fn.arguments[0]   # tensor<MxKxf32>
                B    = fn.arguments[1]   # tensor<KxNxf32>
                bias = fn.arguments[2]   # tensor<MxNxf32>

                # ── Step 1：为 matmul 准备零初始化输出 tensor ─────────
                empty_mn = tensor.EmptyOp(
                    sizes=[],
                    result_type=type_MN,
                    static_sizes=[M, N],
                ).result

                zero = arith.ConstantOp(
                    result=dtype,
                    value=ir.FloatAttr.get(dtype, 0.0),
                ).result

                init_matmul = linalg.FillOp(
                    output=empty_mn,
                    value=zero,
                ).result[0]

                # ── Step 2：linalg.matmul —— C = A × B ─────────────
                # C[i,j] += A[i,k] * B[k,j]
                matmul_result = linalg.matmul(
                    inputs=[A, B],
                    outputs=[init_matmul],
                    result_tensors=[type_MN],
                ).results[0]

                # ── Step 3：linalg.add —— out = matmul_result + bias
                # out[i,j] = matmul_result[i,j] + bias[i,j]
                empty_out = tensor.EmptyOp(
                    sizes=[],
                    result_type=type_MN,
                    static_sizes=[M, N],
                ).result

                add_result = linalg.add(
                    inputs=[matmul_result, bias],
                    outputs=[empty_out],
                    result_tensors=[type_MN],
                ).results[0]

                func.ReturnOp([add_result])

    return module


def main():
    module = build_matmul_bias(M=4, N=8, K=16)

    mlir_text = str(module)
    print(mlir_text)

    with open("matmul_bias.mlir", "w") as f:
        f.write(mlir_text)
        f.write("\n")

    import sys
    print("\n// Written to matmul_bias.mlir", file=sys.stderr)


if __name__ == "__main__":
    main()

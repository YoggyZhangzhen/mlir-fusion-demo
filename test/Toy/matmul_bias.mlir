// RUN: ryuki-opt --ryuki-matmul-bias-fusion %s | FileCheck %s --check-prefix=CHECK-FUSED
// RUN: ryuki-runner %s --stop-after=fusion -o - | FileCheck %s --check-prefix=CHECK-FUSED
// RUN: ryuki-runner %s --emit-llvm -o /dev/null

// CHECK-FUSED-LABEL: func.func @matmul_bias
// CHECK-FUSED-NOT:   linalg.fill
// CHECK-FUSED-NOT:   linalg.add
// CHECK-FUSED:       linalg.matmul

// ============================================================
// 测试输入：matmul + bias add 的标准模式
// 语义：result[M,N] = (A[M,K] × B[K,N]) + bias[M,N]
// ============================================================

module {
  func.func @matmul_bias(
      %A    : tensor<4x16xf32>,
      %B    : tensor<16x8xf32>,
      %bias : tensor<4x8xf32>
  ) -> tensor<4x8xf32> {

    // Step 1: 为 matmul 准备零初始化 tensor
    %empty1 = tensor.empty() : tensor<4x8xf32>
    %c0     = arith.constant 0.000000e+00 : f32
    %zeros  = linalg.fill ins(%c0 : f32)
                          outs(%empty1 : tensor<4x8xf32>)
            -> tensor<4x8xf32>

    // Step 2: 矩阵乘法 C = A × B（从零开始累加）
    %C = linalg.matmul
           ins(%A, %B     : tensor<4x16xf32>, tensor<16x8xf32>)
           outs(%zeros    : tensor<4x8xf32>)
         -> tensor<4x8xf32>

    // Step 3: 加 bias
    %empty2 = tensor.empty() : tensor<4x8xf32>
    %out = linalg.add
             ins(%C, %bias : tensor<4x8xf32>, tensor<4x8xf32>)
             outs(%empty2  : tensor<4x8xf32>)
           -> tensor<4x8xf32>

    func.return %out : tensor<4x8xf32>
  }
}

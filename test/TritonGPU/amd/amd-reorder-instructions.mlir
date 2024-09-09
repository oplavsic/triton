// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
// CHECK-LABEL: order_load_alloc_local_load_local_store
//       CHECK:   %[[LOAD:.+]] = tt.load
//       CHECK:   %[[ALLOC:.+]] = triton_gpu.local_alloc
//       CHECK:   triton_gpu.local_store %[[LOAD]], %[[ALLOC]]
//       CHECK:   triton_gpu.local_load %[[ALLOC]]
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @order_load_alloc_local_load_local_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %10 = triton_gpu.local_alloc : () -> !tt.memdesc<32x32xf32, #shared, mutable>
    triton_gpu.local_store %9, %10 : tensor<32x32xf32, #blocked> -> !tt.memdesc<32x32xf32, #shared, mutable>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %11 = triton_gpu.local_load %10 : !tt.memdesc<32x32xf32, #shared, mutable> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %12 = tt.dot %11, %cst_0, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %13 = triton_gpu.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


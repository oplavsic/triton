
"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import pytest
import torch
import sys

import triton
import triton.language as tl


# POPRAVI POKAZIVACE U EPILOGU!!!
# ir = """
# #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #mfma = #triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
# #shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared2 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
#   tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
#     %c128_i64 = arith.constant 128 : i64
#     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
#     %c0_i64 = arith.constant 0 : i64
#     %c128_i32 = arith.constant 128 : i32
#     %c0_i32 = arith.constant 0 : i32
#     %cst_2 = arith.constant 1.44269502 : f32
#     %0 = arith.mulf %arg3, %cst_2 : f32
#     %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
#     %2 = triton_gpu.view_slice %1[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %3 = triton_gpu.view_slice %1[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %4 = triton_gpu.view_slice %1[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %5 = triton_gpu.view_slice %1[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %6 = tt.get_program_id x : i32
#     %7 = arith.muli %6, %c128_i32 : i32
#     %8 = arith.extsi %7 : i32 to i64
#     %9 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %10 = tt.get_program_id y : i32
#     %11 = arith.muli %10, %arg7 : i32
#     %12 = tt.addptr %arg0, %11 : !tt.ptr<f16, 1>, i32
#     %13 = tt.splat %12 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
#     %14 = arith.extsi %arg8 : i32 to i64
#     %15 = tt.splat %14 : (i64) -> tensor<128x1xi64, #blocked>
#     %16 = tt.addptr %arg1, %11 : !tt.ptr<f16, 1>, i32
#     %17 = arith.extsi %arg11 : i32 to i64
#     %18 = tt.addptr %arg2, %11 : !tt.ptr<f16, 1>, i32
#     %19 = arith.extsi %arg14 : i32 to i64
#     %20 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %22 = arith.extsi %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %23 = arith.addi %9, %22 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %24 = tt.expand_dims %23 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
#     %25 = arith.muli %24, %15 : tensor<128x1xi64, #blocked>
#     %26 = tt.addptr %13, %25 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
#     %27 = tt.broadcast %26 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
#     %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %30 = arith.extsi %29 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %31 = tt.expand_dims %30 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
#     %32 = tt.broadcast %31 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
#     %33 = tt.addptr %27, %32 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
#     %34 = triton_gpu.view_slice %33[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %35 = triton_gpu.view_slice %33[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %36 = triton_gpu.view_slice %33[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %37 = triton_gpu.view_slice %33[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %38 = tt.load %34 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %39 = tt.load %35 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %40 = tt.load %36 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %41 = tt.load %37 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %42 = arith.extf %38 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %43 = arith.extf %39 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %44 = arith.extf %40 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %45 = arith.extf %41 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %46 = arith.mulf %42, %2 : tensor<128x32xf32, #blocked>
#     %47 = arith.mulf %43, %3 : tensor<128x32xf32, #blocked>
#     %48 = arith.mulf %44, %4 : tensor<128x32xf32, #blocked>
#     %49 = arith.mulf %45, %5 : tensor<128x32xf32, #blocked>
#     %50 = arith.truncf %46 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %51 = arith.truncf %47 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %52 = arith.truncf %48 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %53 = arith.truncf %49 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %54 = triton_gpu.convert_layout %50 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %55 = triton_gpu.convert_layout %51 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %56 = triton_gpu.convert_layout %52 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %57 = triton_gpu.convert_layout %53 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     gpu.barrier
#     %58 = triton_gpu.convert_layout %54 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %59 = triton_gpu.convert_layout %55 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %60 = triton_gpu.convert_layout %56 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %61 = triton_gpu.convert_layout %57 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %63 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %64 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %65 = arith.extsi %28 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %66 = arith.extsi %62 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %67 = arith.extsi %63 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %68 = arith.extsi %64 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %69 = arith.addi %20, %65 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %70 = tt.expand_dims %69 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
#     %71 = tt.expand_dims %66 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
#     %72 = tt.expand_dims %67 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#     %73 = tt.broadcast %71 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
#     %74 = tt.expand_dims %68 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#     %75 = tt.splat %16 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %76 = tt.addptr %75, %74 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#     %77 = tt.broadcast %76 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#     %78 = tt.splat %17 : (i64) -> tensor<1x128xi64, #blocked1>
#     %79 = tt.splat %18 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %80 = tt.splat %19 : (i64) -> tensor<1x128xi64, #blocked1>
#     %81 = arith.muli %72, %80 : tensor<1x128xi64, #blocked1>
#     %82 = tt.broadcast %81 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %83 = arith.muli %72, %78 : tensor<1x128xi64, #blocked1>
#     %84 = tt.broadcast %83 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %85 = tt.addptr %77, %84 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#     %86 = triton_gpu.view_slice %85[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %88 = triton_gpu.convert_layout %87 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#     %89 = arith.subi %arg20, %c128_i32 : i32
#     %90:6 = scf.for %arg21 = %c0_i32 to %89 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %88, %arg27 = %c128_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, tensor<32x128xf16, #shared1>, i64)  : i32 {
#       %169 = tt.splat %arg27 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %170 = arith.addi %169, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %171 = tt.expand_dims %170 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#       %172 = arith.muli %171, %78 : tensor<1x128xi64, #blocked1>
#       %173 = tt.broadcast %172 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#       %174 = tt.addptr %77, %173 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %175 = triton_gpu.view_slice %174[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %176 = tt.load %175 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>

#       %177 = arith.addi %arg27, %c128_i64 : i64
#       %569 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %570 = arith.addi %569, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %571 = tt.expand_dims %570 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#       %572 = arith.muli %571, %78 : tensor<1x128xi64, #blocked1>
#       %573 = tt.broadcast %572 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#       %574 = tt.addptr %77, %573 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>

#       %178 = triton_gpu.view_slice %574[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %179 = triton_gpu.view_slice %574[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %180 = triton_gpu.view_slice %574[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %181 = tt.load %180 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %182 = triton_gpu.convert_layout %arg26 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %183 = tt.dot %61, %182, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %184 = triton_gpu.convert_layout %181 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %185 = tt.load %179 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %186 = triton_gpu.convert_layout %184 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %187 = tt.dot %60, %186, %183 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %188 = triton_gpu.convert_layout %185 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %189 = tt.load %178 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %190 = triton_gpu.convert_layout %188 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %191 = tt.dot %59, %190, %187 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %192 = triton_gpu.convert_layout %189 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       gpu.barrier
#       %193 = triton_gpu.convert_layout %192 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %194 = tt.dot %58, %193, %191 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>



#       %195 = "tt.reduce"(%194) <{axis = 1 : i32}> ({
#       ^bb0(%arg29: f32, %arg30: f32):
#         %246 = arith.maximumf %arg29, %arg30 : f32
#         tt.reduce.return %246 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %196 = arith.maximumf %arg24, %195 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %197 = tt.expand_dims %196 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %198 = tt.broadcast %197 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %199 = arith.subf %194, %198 : tensor<128x128xf32, #mfma>
#       %200 = tt.extern_elementwise %199 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %212 = arith.truncf %200 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>



#       %201 = arith.subf %arg24, %196 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %202 = tt.extern_elementwise %201 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %203 = tt.expand_dims %202 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %204 = tt.broadcast %203 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %205 = arith.mulf %arg22, %204 : tensor<128x128xf32, #mfma>



#       %206 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %207 = arith.addi %206, %68 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %208 = tt.expand_dims %207 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#       %209 = tt.addptr %79, %208 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#       %210 = tt.broadcast %209 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#       %211 = tt.addptr %210, %82 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %221 = triton_gpu.view_slice %211[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %222 = triton_gpu.view_slice %211[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %223 = triton_gpu.view_slice %211[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %224 = triton_gpu.view_slice %211[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>

#       %216 = triton_gpu.view_slice %212[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %220 = triton_gpu.convert_layout %216 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %213 = triton_gpu.view_slice %212[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %217 = triton_gpu.convert_layout %213 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %214 = triton_gpu.view_slice %212[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %218 = triton_gpu.convert_layout %214 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %215 = triton_gpu.view_slice %212[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %219 = triton_gpu.convert_layout %215 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>

#       %228 = tt.load %224 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %225 = tt.load %221 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %226 = tt.load %222 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %227 = tt.load %223 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>

#       gpu.barrier

#       %232 = triton_gpu.convert_layout %228 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %229 = triton_gpu.convert_layout %225 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %230 = triton_gpu.convert_layout %226 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %231 = triton_gpu.convert_layout %227 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>

#       gpu.barrier
#       %236 = triton_gpu.convert_layout %232 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %233 = triton_gpu.convert_layout %229 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %234 = triton_gpu.convert_layout %230 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %235 = triton_gpu.convert_layout %231 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>


#       %237 = tt.dot %220, %236, %205 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %238 = tt.dot %217, %233, %237 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %239 = tt.dot %218, %234, %238 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %240 = tt.dot %219, %235, %239 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

#       %241 = "tt.reduce"(%200) <{axis = 1 : i32}> ({
#       ^bb0(%arg29: f32, %arg30: f32):
#         %246 = arith.addf %arg29, %arg30 : f32
#         tt.reduce.return %246 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>


#       %242 = arith.mulf %arg23, %202 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %243 = arith.addf %242, %241 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %244 = arith.addi %arg25, %c128_i64 : i64
#       gpu.barrier
#       %245 = triton_gpu.convert_layout %176 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       scf.yield %240, %243, %196, %244, %245, %177 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, tensor<32x128xf16, #shared1>, i64
#     }

    






#     %869 = tt.splat %c128_i64 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %870 = arith.addi %869, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %871 = tt.expand_dims %870 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#     %872 = arith.muli %871, %78 : tensor<1x128xi64, #blocked1>
#     %873 = tt.broadcast %872 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %874 = tt.addptr %77, %873 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>

#     %91 = triton_gpu.view_slice %874[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %92 = triton_gpu.view_slice %874[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %93 = triton_gpu.view_slice %874[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %94 = tt.load %93 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     gpu.barrier
#     %95 = triton_gpu.convert_layout %90#4 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %96 = tt.dot %61, %95, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#     gpu.barrier
#     %97 = triton_gpu.convert_layout %94 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#     %98 = tt.load %92 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     gpu.barrier
#     %99 = triton_gpu.convert_layout %97 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %100 = tt.dot %60, %99, %96 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#     gpu.barrier
#     %101 = triton_gpu.convert_layout %98 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#     %102 = tt.load %91 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     gpu.barrier
#     %103 = triton_gpu.convert_layout %101 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %104 = tt.dot %59, %103, %100 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#     gpu.barrier
#     %105 = triton_gpu.convert_layout %102 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#     gpu.barrier
#     %106 = triton_gpu.convert_layout %105 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %107 = tt.dot %58, %106, %104 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#     %108 = "tt.reduce"(%107) <{axis = 1 : i32}> ({
#     ^bb0(%arg21: f32, %arg22: f32):
#       %169 = arith.maximumf %arg21, %arg22 : f32
#       tt.reduce.return %169 : f32
#     }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %109 = arith.maximumf %90#2, %108 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %110 = tt.expand_dims %109 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %111 = tt.broadcast %110 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %112 = arith.subf %107, %111 : tensor<128x128xf32, #mfma>
#     %113 = tt.extern_elementwise %112 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %114 = arith.subf %90#2, %109 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %115 = tt.extern_elementwise %114 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %116 = tt.expand_dims %115 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %117 = tt.broadcast %116 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %118 = arith.mulf %90#0, %117 : tensor<128x128xf32, #mfma>
#     %119 = tt.splat %c0_i64 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %120 = arith.addi %119, %68 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %121 = tt.expand_dims %120 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#     %122 = tt.addptr %79, %121 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#     %123 = tt.broadcast %122 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#     %124 = tt.addptr %123, %82 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#     %125 = arith.truncf %113 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     %126 = triton_gpu.view_slice %125[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %127 = triton_gpu.view_slice %125[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %128 = triton_gpu.view_slice %125[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %129 = triton_gpu.view_slice %125[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %130 = triton_gpu.convert_layout %126 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %131 = triton_gpu.convert_layout %127 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %132 = triton_gpu.convert_layout %128 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %133 = triton_gpu.convert_layout %129 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %134 = triton_gpu.view_slice %124[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %135 = triton_gpu.view_slice %124[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %136 = triton_gpu.view_slice %124[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %137 = triton_gpu.view_slice %124[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>

#     %141 = tt.load %137 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %138 = tt.load %134 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %139 = tt.load %135 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %140 = tt.load %136 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     gpu.barrier

#     %145 = triton_gpu.convert_layout %141 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#     %142 = triton_gpu.convert_layout %138 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#     %143 = triton_gpu.convert_layout %139 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#     %144 = triton_gpu.convert_layout %140 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#     gpu.barrier

#     %149 = triton_gpu.convert_layout %145 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#     %146 = triton_gpu.convert_layout %142 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#     %147 = triton_gpu.convert_layout %143 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#     %148 = triton_gpu.convert_layout %144 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>

#     %150 = tt.dot %133, %149, %118 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#     %151 = tt.dot %130, %146, %150 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#     %152 = tt.dot %131, %147, %151 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#     %153 = tt.dot %132, %148, %152 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

#     %154 = "tt.reduce"(%113) <{axis = 1 : i32}> ({
#     ^bb0(%arg21: f32, %arg22: f32):
#       %169 = arith.addf %arg21, %arg22 : f32
#       tt.reduce.return %169 : f32
#     }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %155 = arith.mulf %90#1, %115 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %156 = arith.addf %155, %154 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %157 = tt.expand_dims %156 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %158 = tt.broadcast %157 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %159 = arith.divf %153, %158 : tensor<128x128xf32, #mfma>
#     %160 = tt.addptr %arg5, %11 : !tt.ptr<f16, 1>, i32
#     %161 = arith.extsi %arg17 : i32 to i64
#     %162 = arith.truncf %159 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     %163 = tt.splat %161 : (i64) -> tensor<128x1xi64, #mfma>
#     %164 = arith.muli %70, %163 : tensor<128x1xi64, #mfma>
#     %165 = tt.splat %160 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
#     %166 = tt.addptr %165, %164 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
#     %167 = tt.broadcast %166 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
#     %168 = tt.addptr %167, %73 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
#     tt.store %168, %162 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
#     tt.return
#   }
# }

#     """


# ir = """
# #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #mfma = #triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
# #shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared2 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
#   tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
#     %c128_i64 = arith.constant 128 : i64
#     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
#     %c0_i64 = arith.constant 0 : i64
#     %c128_i32 = arith.constant 128 : i32
#     %c0_i32 = arith.constant 0 : i32
#     %cst_2 = arith.constant 1.44269502 : f32
#     %0 = arith.mulf %arg3, %cst_2 : f32
#     %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
#     %2 = triton_gpu.view_slice %1[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %3 = triton_gpu.view_slice %1[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %4 = triton_gpu.view_slice %1[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %5 = triton_gpu.view_slice %1[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %6 = tt.get_program_id x : i32
#     %7 = arith.muli %6, %c128_i32 : i32
#     %8 = arith.extsi %7 : i32 to i64
#     %9 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %10 = tt.get_program_id y : i32
#     %11 = arith.muli %10, %arg7 : i32
#     %12 = tt.addptr %arg0, %11 : !tt.ptr<f16, 1>, i32
#     %13 = tt.splat %12 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
#     %14 = arith.extsi %arg8 : i32 to i64
#     %15 = tt.splat %14 : (i64) -> tensor<128x1xi64, #blocked>
#     %16 = tt.addptr %arg1, %11 : !tt.ptr<f16, 1>, i32
#     %17 = arith.extsi %arg11 : i32 to i64
#     %18 = tt.addptr %arg2, %11 : !tt.ptr<f16, 1>, i32
#     %19 = arith.extsi %arg14 : i32 to i64
#     %20 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %22 = arith.extsi %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %23 = arith.addi %9, %22 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %24 = tt.expand_dims %23 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
#     %25 = arith.muli %24, %15 : tensor<128x1xi64, #blocked>
#     %26 = tt.addptr %13, %25 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
#     %27 = tt.broadcast %26 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
#     %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %30 = arith.extsi %29 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %31 = tt.expand_dims %30 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
#     %32 = tt.broadcast %31 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
#     %33 = tt.addptr %27, %32 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
#     %34 = triton_gpu.view_slice %33[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %35 = triton_gpu.view_slice %33[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %36 = triton_gpu.view_slice %33[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %37 = triton_gpu.view_slice %33[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %38 = tt.load %34 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %39 = tt.load %35 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %40 = tt.load %36 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %41 = tt.load %37 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %42 = arith.extf %38 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %43 = arith.extf %39 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %44 = arith.extf %40 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %45 = arith.extf %41 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %46 = arith.mulf %42, %2 : tensor<128x32xf32, #blocked>
#     %47 = arith.mulf %43, %3 : tensor<128x32xf32, #blocked>
#     %48 = arith.mulf %44, %4 : tensor<128x32xf32, #blocked>
#     %49 = arith.mulf %45, %5 : tensor<128x32xf32, #blocked>
#     %50 = arith.truncf %46 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %51 = arith.truncf %47 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %52 = arith.truncf %48 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %53 = arith.truncf %49 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %54 = triton_gpu.convert_layout %50 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %55 = triton_gpu.convert_layout %51 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %56 = triton_gpu.convert_layout %52 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %57 = triton_gpu.convert_layout %53 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %63 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %64 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %65 = arith.extsi %28 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %66 = arith.extsi %62 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %67 = arith.extsi %63 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %68 = arith.extsi %64 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %69 = arith.addi %20, %65 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %70 = tt.expand_dims %69 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
#     %71 = tt.expand_dims %66 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
#     %72 = tt.expand_dims %67 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#     %73 = tt.broadcast %71 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
#     %74 = tt.expand_dims %68 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#     %75 = tt.splat %16 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %76 = tt.addptr %75, %74 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#     %77 = tt.broadcast %76 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#     %78 = tt.splat %17 : (i64) -> tensor<1x128xi64, #blocked1>
#     %79 = tt.splat %18 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %80 = tt.splat %19 : (i64) -> tensor<1x128xi64, #blocked1>
#     %81 = arith.muli %72, %80 : tensor<1x128xi64, #blocked1>
#     %82 = tt.broadcast %81 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %83 = arith.muli %72, %78 : tensor<1x128xi64, #blocked1>
#     %84 = tt.broadcast %83 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %85 = tt.addptr %77, %84 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#     %86 = triton_gpu.view_slice %85[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %88 = triton_gpu.convert_layout %87 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#     %89 = arith.subi %arg20, %c128_i32 : i32
#     %90:6 = scf.for %arg21 = %c0_i32 to %89 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %88, %arg27 = %c128_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, tensor<32x128xf16, #shared1>, i64)  : i32 {
#       %169 = tt.splat %arg27 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %170 = arith.addi %169, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %171 = tt.expand_dims %170 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#       %172 = arith.muli %171, %78 : tensor<1x128xi64, #blocked1>
#       %173 = tt.broadcast %172 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#       %174 = tt.addptr %77, %173 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %175 = triton_gpu.view_slice %174[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %176 = tt.load %175 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %177 = arith.addi %arg27, %c128_i64 : i64
#       %569 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %570 = arith.addi %569, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %571 = tt.expand_dims %570 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#       %572 = arith.muli %571, %78 : tensor<1x128xi64, #blocked1>
#       %573 = tt.broadcast %572 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#       %574 = tt.addptr %77, %573 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>

#       %178 = triton_gpu.view_slice %574[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %179 = triton_gpu.view_slice %574[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %180 = triton_gpu.view_slice %574[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %181 = tt.load %180 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %182 = triton_gpu.convert_layout %arg26 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %61 = triton_gpu.convert_layout %57 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#       %183 = tt.dot %61, %182, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       %184 = triton_gpu.convert_layout %181 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %185 = tt.load %179 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %186 = triton_gpu.convert_layout %184 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %60 = triton_gpu.convert_layout %56 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#       %187 = tt.dot %60, %186, %183 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       %188 = triton_gpu.convert_layout %185 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %189 = tt.load %178 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %190 = triton_gpu.convert_layout %188 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %59 = triton_gpu.convert_layout %55 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#       %191 = tt.dot %59, %190, %187 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       %192 = triton_gpu.convert_layout %189 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %193 = triton_gpu.convert_layout %192 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %58 = triton_gpu.convert_layout %54 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#       %194 = tt.dot %58, %193, %191 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       %195 = "tt.reduce"(%194) <{axis = 1 : i32}> ({
#       ^bb0(%arg29: f32, %arg30: f32):
#         %246 = arith.maximumf %arg29, %arg30 : f32
#         tt.reduce.return %246 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %196 = arith.maximumf %arg24, %195 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %197 = tt.expand_dims %196 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %198 = tt.broadcast %197 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %199 = arith.subf %194, %198 : tensor<128x128xf32, #mfma>
#       %200 = tt.extern_elementwise %199 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %201 = arith.subf %arg24, %196 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %202 = tt.extern_elementwise %201 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %203 = tt.expand_dims %202 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %204 = tt.broadcast %203 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %205 = arith.mulf %arg22, %204 : tensor<128x128xf32, #mfma>
#       %206 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %207 = arith.addi %206, %68 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %208 = tt.expand_dims %207 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#       %209 = tt.addptr %79, %208 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#       %210 = tt.broadcast %209 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#       %211 = tt.addptr %210, %82 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %212 = arith.truncf %200 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#       %213 = triton_gpu.view_slice %212[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %214 = triton_gpu.view_slice %212[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %215 = triton_gpu.view_slice %212[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %216 = triton_gpu.view_slice %212[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %217 = triton_gpu.convert_layout %213 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %218 = triton_gpu.convert_layout %214 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %219 = triton_gpu.convert_layout %215 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %220 = triton_gpu.convert_layout %216 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %221 = triton_gpu.view_slice %211[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %222 = triton_gpu.view_slice %211[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %223 = triton_gpu.view_slice %211[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %224 = triton_gpu.view_slice %211[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>

#       %228 = tt.load %224 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %226 = tt.load %222 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %225 = tt.load %221 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %227 = tt.load %223 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>

#       %232 = triton_gpu.convert_layout %228 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %229 = triton_gpu.convert_layout %225 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %230 = triton_gpu.convert_layout %226 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %231 = triton_gpu.convert_layout %227 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>

#       %236 = triton_gpu.convert_layout %232 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %233 = triton_gpu.convert_layout %229 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %234 = triton_gpu.convert_layout %230 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %235 = triton_gpu.convert_layout %231 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>

#       %237 = tt.dot %220, %236, %205 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %238 = tt.dot %217, %233, %237 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %239 = tt.dot %218, %234, %238 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %240 = tt.dot %219, %235, %239 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

#       %241 = "tt.reduce"(%200) <{axis = 1 : i32}> ({
#       ^bb0(%arg29: f32, %arg30: f32):
#         %246 = arith.addf %arg29, %arg30 : f32
#         tt.reduce.return %246 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %242 = arith.mulf %arg23, %202 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %243 = arith.addf %242, %241 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %244 = arith.addi %arg25, %c128_i64 : i64
#       %245 = triton_gpu.convert_layout %176 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       scf.yield %240, %243, %196, %244, %245, %177 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, tensor<32x128xf16, #shared1>, i64
#     }
#     %869 = tt.splat %90#3 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %870 = arith.addi %869, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %871 = tt.expand_dims %870 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#     %872 = arith.muli %871, %78 : tensor<1x128xi64, #blocked1>
#     %873 = tt.broadcast %872 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %874 = tt.addptr %77, %873 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>

#     %91 = triton_gpu.view_slice %874[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %92 = triton_gpu.view_slice %874[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %93 = triton_gpu.view_slice %874[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %94 = tt.load %93 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %95 = triton_gpu.convert_layout %90#4 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %611 = triton_gpu.convert_layout %57 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %96 = tt.dot %611, %95, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#     %97 = triton_gpu.convert_layout %94 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#     %98 = tt.load %92 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %99 = triton_gpu.convert_layout %97 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %601 = triton_gpu.convert_layout %56 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %100 = tt.dot %601, %99, %96 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#     %101 = triton_gpu.convert_layout %98 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#     %102 = tt.load %91 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %103 = triton_gpu.convert_layout %101 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %591 = triton_gpu.convert_layout %55 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %104 = tt.dot %591, %103, %100 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#     %105 = triton_gpu.convert_layout %102 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#     %106 = triton_gpu.convert_layout %105 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %581 = triton_gpu.convert_layout %54 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %107 = tt.dot %581, %106, %104 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#     %108 = "tt.reduce"(%107) <{axis = 1 : i32}> ({
#     ^bb0(%arg21: f32, %arg22: f32):
#       %169 = arith.maximumf %arg21, %arg22 : f32
#       tt.reduce.return %169 : f32
#     }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %109 = arith.maximumf %90#2, %108 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %110 = tt.expand_dims %109 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %111 = tt.broadcast %110 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %112 = arith.subf %107, %111 : tensor<128x128xf32, #mfma>
#     %113 = tt.extern_elementwise %112 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %114 = arith.subf %90#2, %109 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %115 = tt.extern_elementwise %114 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %116 = tt.expand_dims %115 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %117 = tt.broadcast %116 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %118 = arith.mulf %90#0, %117 : tensor<128x128xf32, #mfma>
#     %119 = tt.splat %90#3 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %120 = arith.addi %119, %68 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %121 = tt.expand_dims %120 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#     %122 = tt.addptr %79, %121 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#     %123 = tt.broadcast %122 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#     %124 = tt.addptr %123, %82 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#     %125 = arith.truncf %113 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     %126 = triton_gpu.view_slice %125[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %127 = triton_gpu.view_slice %125[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %128 = triton_gpu.view_slice %125[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %129 = triton_gpu.view_slice %125[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %130 = triton_gpu.convert_layout %126 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %131 = triton_gpu.convert_layout %127 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %132 = triton_gpu.convert_layout %128 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %133 = triton_gpu.convert_layout %129 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %134 = triton_gpu.view_slice %124[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %135 = triton_gpu.view_slice %124[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %136 = triton_gpu.view_slice %124[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %137 = triton_gpu.view_slice %124[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>

#     %141 = tt.load %137 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %138 = tt.load %134 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %139 = tt.load %135 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %140 = tt.load %136 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>

#     %145 = triton_gpu.convert_layout %141 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#     %142 = triton_gpu.convert_layout %138 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#     %143 = triton_gpu.convert_layout %139 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#     %144 = triton_gpu.convert_layout %140 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>

#     %149 = triton_gpu.convert_layout %145 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#     %146 = triton_gpu.convert_layout %142 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#     %147 = triton_gpu.convert_layout %143 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#     %148 = triton_gpu.convert_layout %144 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>

#     %150 = tt.dot %133, %149, %118 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#     %151 = tt.dot %130, %146, %150 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#     %152 = tt.dot %131, %147, %151 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#     %153 = tt.dot %132, %148, %152 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

#     %154 = "tt.reduce"(%113) <{axis = 1 : i32}> ({
#     ^bb0(%arg21: f32, %arg22: f32):
#       %169 = arith.addf %arg21, %arg22 : f32
#       tt.reduce.return %169 : f32
#     }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %155 = arith.mulf %90#1, %115 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %156 = arith.addf %155, %154 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %157 = tt.expand_dims %156 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %158 = tt.broadcast %157 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %159 = arith.divf %153, %158 : tensor<128x128xf32, #mfma>
#     %160 = tt.addptr %arg5, %11 : !tt.ptr<f16, 1>, i32
#     %161 = arith.extsi %arg17 : i32 to i64
#     %162 = arith.truncf %159 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     %163 = tt.splat %161 : (i64) -> tensor<128x1xi64, #mfma>
#     %164 = arith.muli %70, %163 : tensor<128x1xi64, #mfma>
#     %165 = tt.splat %160 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
#     %166 = tt.addptr %165, %164 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
#     %167 = tt.broadcast %166 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
#     %168 = tt.addptr %167, %73 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
#     tt.store %168, %162 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
#     tt.return
#   }
# }

#     """

# ir = """
# #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #mfma = #triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
# #shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared2 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
#   tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
#     %c64_i64 = arith.constant 64 : i64
#     %c128_i32 = arith.constant 128 : i32
#     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
#     %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mfma>
#     %c0_i64 = arith.constant 0 : i64
#     %c64_i32 = arith.constant 64 : i32
#     %c0_i32 = arith.constant 0 : i32
#     %cst_3 = arith.constant 1.44269502 : f32
#     %0 = arith.mulf %arg3, %cst_3 : f32
#     %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
#     %2 = triton_gpu.view_slice %1[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %3 = triton_gpu.view_slice %1[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %4 = triton_gpu.view_slice %1[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %5 = triton_gpu.view_slice %1[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %6 = tt.get_program_id x : i32
#     %7 = arith.muli %6, %c128_i32 : i32
#     %8 = arith.extsi %7 : i32 to i64
#     %9 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %10 = tt.get_program_id y : i32
#     %11 = arith.muli %10, %arg7 : i32
#     %12 = tt.addptr %arg0, %11 : !tt.ptr<f16, 1>, i32
#     %13 = tt.splat %12 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
#     %14 = arith.extsi %arg8 : i32 to i64
#     %15 = tt.splat %14 : (i64) -> tensor<128x1xi64, #blocked>
#     %16 = tt.addptr %arg1, %11 : !tt.ptr<f16, 1>, i32
#     %17 = arith.extsi %arg11 : i32 to i64
#     %18 = tt.addptr %arg2, %11 : !tt.ptr<f16, 1>, i32
#     %19 = arith.extsi %arg14 : i32 to i64
#     %20 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %22 = arith.extsi %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %23 = arith.addi %9, %22 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %24 = tt.expand_dims %23 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
#     %25 = arith.muli %24, %15 : tensor<128x1xi64, #blocked>
#     %26 = tt.addptr %13, %25 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
#     %27 = tt.broadcast %26 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
#     %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %30 = arith.extsi %29 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %31 = tt.expand_dims %30 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
#     %32 = tt.broadcast %31 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
#     %33 = tt.addptr %27, %32 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
#     %34 = triton_gpu.view_slice %33[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %35 = triton_gpu.view_slice %33[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %36 = triton_gpu.view_slice %33[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %37 = triton_gpu.view_slice %33[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %38 = tt.load %34 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %39 = tt.load %35 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %40 = tt.load %36 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %41 = tt.load %37 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %42 = arith.extf %38 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %43 = arith.extf %39 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %44 = arith.extf %40 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %45 = arith.extf %41 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %46 = arith.mulf %42, %2 : tensor<128x32xf32, #blocked>
#     %47 = arith.mulf %43, %3 : tensor<128x32xf32, #blocked>
#     %48 = arith.mulf %44, %4 : tensor<128x32xf32, #blocked>
#     %49 = arith.mulf %45, %5 : tensor<128x32xf32, #blocked>
#     %50 = arith.truncf %46 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %51 = arith.truncf %47 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %52 = arith.truncf %48 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %53 = arith.truncf %49 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %54 = triton_gpu.convert_layout %50 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %55 = triton_gpu.convert_layout %51 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %56 = triton_gpu.convert_layout %52 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %57 = triton_gpu.convert_layout %53 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     gpu.barrier
#     %58 = triton_gpu.convert_layout %54 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %59 = triton_gpu.convert_layout %55 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %60 = triton_gpu.convert_layout %56 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %61 = triton_gpu.convert_layout %57 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %63 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %64 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %65 = arith.extsi %28 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %66 = arith.extsi %62 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %67 = arith.extsi %63 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %68 = arith.extsi %64 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %69 = arith.addi %20, %65 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %70 = tt.expand_dims %69 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
#     %71 = tt.expand_dims %66 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
#     %72 = tt.expand_dims %67 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#     %73 = tt.broadcast %71 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
#     %74 = tt.expand_dims %68 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#     %75 = tt.splat %16 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %76 = tt.addptr %75, %74 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#     %77 = tt.broadcast %76 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x64x!tt.ptr<f16, 1>, #blocked1>
#     %78 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %79 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %80 = arith.extsi %78 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %81 = arith.extsi %79 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %82 = tt.splat %17 : (i64) -> tensor<1x64xi64, #blocked1>
#     %83 = tt.splat %18 : (!tt.ptr<f16, 1>) -> tensor<64x1x!tt.ptr<f16, 1>, #blocked1>
#     %84 = tt.splat %19 : (i64) -> tensor<1x128xi64, #blocked1>
#     %85 = arith.muli %72, %84 : tensor<1x128xi64, #blocked1>
#     %86 = tt.broadcast %85 : (tensor<1x128xi64, #blocked1>) -> tensor<64x128xi64, #blocked1>
#     %87 = tt.expand_dims %80 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi64, #blocked1>
#     %88 = arith.muli %87, %82 : tensor<1x64xi64, #blocked1>
#     %89 = tt.broadcast %88 : (tensor<1x64xi64, #blocked1>) -> tensor<128x64xi64, #blocked1>
#     %90 = tt.addptr %77, %89 : tensor<128x64x!tt.ptr<f16, 1>, #blocked1>, tensor<128x64xi64, #blocked1>
#     %91 = triton_gpu.view_slice %90[0, 0] [32, 64] [1, 1] : tensor<128x64x!tt.ptr<f16, 1>, #blocked1> to tensor<32x64x!tt.ptr<f16, 1>, #blocked1>
#     %92 = tt.load %91 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
#     gpu.barrier
#     %93 = triton_gpu.convert_layout %92 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
#     %94 = arith.subi %arg20, %c64_i32 : i32
#     %95:6 = scf.for %arg21 = %c0_i32 to %94 step %c64_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %93, %arg27 = %c64_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, tensor<32x64xf16, #shared1>, i64)  : i32 {
#       %160 = tt.splat %arg27 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %161 = arith.addi %160, %80 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %162 = tt.expand_dims %161 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi64, #blocked1>
#       %163 = arith.muli %162, %82 : tensor<1x64xi64, #blocked1>
#       %164 = tt.broadcast %163 : (tensor<1x64xi64, #blocked1>) -> tensor<128x64xi64, #blocked1>
#       %165 = tt.addptr %77, %164 : tensor<128x64x!tt.ptr<f16, 1>, #blocked1>, tensor<128x64xi64, #blocked1>
#       %166 = triton_gpu.view_slice %165[0, 0] [32, 64] [1, 1] : tensor<128x64x!tt.ptr<f16, 1>, #blocked1> to tensor<32x64x!tt.ptr<f16, 1>, #blocked1>
#       %167 = tt.load %166 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
#       %168 = arith.addi %arg27, %c64_i64 : i64

#       %569 = tt.splat %arg25 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %570 = arith.addi %569, %80 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %571 = tt.expand_dims %570 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi64, #blocked1>
#       %572 = arith.muli %571, %82 : tensor<1x64xi64, #blocked1>
#       %573 = tt.broadcast %572 : (tensor<1x64xi64, #blocked1>) -> tensor<128x64xi64, #blocked1>
#       %574 = tt.addptr %77, %573 : tensor<128x64x!tt.ptr<f16, 1>, #blocked1>, tensor<128x64xi64, #blocked1>

#       %169 = triton_gpu.view_slice %574[96, 0] [32, 64] [1, 1] : tensor<128x64x!tt.ptr<f16, 1>, #blocked1> to tensor<32x64x!tt.ptr<f16, 1>, #blocked1>
#       %170 = triton_gpu.view_slice %574[64, 0] [32, 64] [1, 1] : tensor<128x64x!tt.ptr<f16, 1>, #blocked1> to tensor<32x64x!tt.ptr<f16, 1>, #blocked1>
#       %171 = triton_gpu.view_slice %574[32, 0] [32, 64] [1, 1] : tensor<128x64x!tt.ptr<f16, 1>, #blocked1> to tensor<32x64x!tt.ptr<f16, 1>, #blocked1>
#       %172 = tt.load %171 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
#       gpu.barrier
#       %173 = triton_gpu.convert_layout %arg26 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %174 = tt.dot %61, %173, %cst_2 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x64xf32, #mfma>
#       gpu.barrier
#       %175 = triton_gpu.convert_layout %172 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
#       %176 = tt.load %170 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
#       gpu.barrier
#       %177 = triton_gpu.convert_layout %175 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %178 = tt.dot %60, %177, %174 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x64xf32, #mfma>
#       gpu.barrier
#       %179 = triton_gpu.convert_layout %176 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
#       %180 = tt.load %169 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
#       gpu.barrier
#       %181 = triton_gpu.convert_layout %179 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %182 = tt.dot %59, %181, %178 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x64xf32, #mfma>
#       gpu.barrier
#       %183 = triton_gpu.convert_layout %180 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
#       gpu.barrier
#       %184 = triton_gpu.convert_layout %183 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %185 = tt.dot %58, %184, %182 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x64xf32, #mfma>


#       %186 = "tt.reduce"(%185) <{axis = 1 : i32}> ({
#       ^bb0(%arg28: f32, %arg29: f32):
#         %223 = arith.maximumf %arg28, %arg29 : f32
#         tt.reduce.return %223 : f32
#       }) : (tensor<128x64xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

#       %187 = arith.maximumf %arg24, %186 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %188 = tt.expand_dims %187 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %189 = tt.broadcast %188 : (tensor<128x1xf32, #mfma>) -> tensor<128x64xf32, #mfma>
#       %190 = arith.subf %185, %189 : tensor<128x64xf32, #mfma>
#       %191 = tt.extern_elementwise %190 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x64xf32, #mfma>) -> tensor<128x64xf32, #mfma>
#       %192 = arith.subf %arg24, %187 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %193 = tt.extern_elementwise %192 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %194 = tt.expand_dims %193 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %195 = tt.broadcast %194 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %196 = arith.mulf %arg22, %195 : tensor<128x128xf32, #mfma>

#       %197 = tt.splat %arg25 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %198 = arith.addi %197, %81 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %199 = tt.expand_dims %198 {axis = 1 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi64, #blocked1>
#       %200 = tt.addptr %83, %199 : tensor<64x1x!tt.ptr<f16, 1>, #blocked1>, tensor<64x1xi64, #blocked1>
#       %201 = tt.broadcast %200 : (tensor<64x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<64x128x!tt.ptr<f16, 1>, #blocked1>
#       %202 = tt.addptr %201, %86 : tensor<64x128x!tt.ptr<f16, 1>, #blocked1>, tensor<64x128xi64, #blocked1>

#       %203 = arith.truncf %191 : tensor<128x64xf32, #mfma> to tensor<128x64xf16, #mfma>
#       %204 = triton_gpu.view_slice %203[0, 32] [128, 32] [1, 1] : tensor<128x64xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %205 = triton_gpu.view_slice %203[0, 0] [128, 32] [1, 1] : tensor<128x64xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %206 = triton_gpu.convert_layout %204 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %207 = triton_gpu.convert_layout %205 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>

#       %208 = triton_gpu.view_slice %202[32, 0] [32, 128] [1, 1] : tensor<64x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %209 = triton_gpu.view_slice %202[0, 0] [32, 128] [1, 1] : tensor<64x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>

#       %210 = tt.load %208 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %211 = tt.load %209 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>

#       gpu.barrier
#       %212 = triton_gpu.convert_layout %210 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %213 = triton_gpu.convert_layout %211 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       gpu.barrier
#       %214 = triton_gpu.convert_layout %212 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %215 = triton_gpu.convert_layout %213 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %216 = tt.dot %207, %215, %196 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %217 = tt.dot %206, %214, %216 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %218 = "tt.reduce"(%191) <{axis = 1 : i32}> ({
#       ^bb0(%arg28: f32, %arg29: f32):
#         %223 = arith.addf %arg28, %arg29 : f32
#         tt.reduce.return %223 : f32
#       }) : (tensor<128x64xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %219 = arith.mulf %arg23, %193 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %220 = arith.addf %219, %218 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %221 = arith.addi %arg25, %c64_i64 : i64
#       gpu.barrier
#       %222 = triton_gpu.convert_layout %167 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
#       scf.yield %217, %220, %187, %221, %222, %168 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, tensor<32x64xf16, #shared1>, i64
#     }


#     %869 = tt.splat %c64_i64 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %870 = arith.addi %869, %80 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %871 = tt.expand_dims %870 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi64, #blocked1>
#     %872 = arith.muli %871, %82 : tensor<1x64xi64, #blocked1>
#     %873 = tt.broadcast %872 : (tensor<1x64xi64, #blocked1>) -> tensor<128x64xi64, #blocked1>
#     %874 = tt.addptr %77, %873 : tensor<128x64x!tt.ptr<f16, 1>, #blocked1>, tensor<128x64xi64, #blocked1>

#     %96 = triton_gpu.view_slice %874[96, 0] [32, 64] [1, 1] : tensor<128x64x!tt.ptr<f16, 1>, #blocked1> to tensor<32x64x!tt.ptr<f16, 1>, #blocked1>
#     %97 = triton_gpu.view_slice %874[64, 0] [32, 64] [1, 1] : tensor<128x64x!tt.ptr<f16, 1>, #blocked1> to tensor<32x64x!tt.ptr<f16, 1>, #blocked1>
#     %98 = triton_gpu.view_slice %874[32, 0] [32, 64] [1, 1] : tensor<128x64x!tt.ptr<f16, 1>, #blocked1> to tensor<32x64x!tt.ptr<f16, 1>, #blocked1>
#     %99 = tt.load %98 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
#     gpu.barrier
#     %100 = triton_gpu.convert_layout %95#4 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %101 = tt.dot %61, %100, %cst_2 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x64xf32, #mfma>
#     gpu.barrier
#     %102 = triton_gpu.convert_layout %99 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
#     %103 = tt.load %97 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
#     gpu.barrier
#     %104 = triton_gpu.convert_layout %102 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %105 = tt.dot %60, %104, %101 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x64xf32, #mfma>
#     gpu.barrier
#     %106 = triton_gpu.convert_layout %103 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
#     %107 = tt.load %96 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
#     gpu.barrier
#     %108 = triton_gpu.convert_layout %106 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %109 = tt.dot %59, %108, %105 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x64xf32, #mfma>
#     gpu.barrier
#     %110 = triton_gpu.convert_layout %107 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
#     gpu.barrier
#     %111 = triton_gpu.convert_layout %110 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#     %112 = tt.dot %58, %111, %109 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x64xf32, #mfma>
#     %113 = "tt.reduce"(%112) <{axis = 1 : i32}> ({
#     ^bb0(%arg21: f32, %arg22: f32):
#       %160 = arith.maximumf %arg21, %arg22 : f32
#       tt.reduce.return %160 : f32
#     }) : (tensor<128x64xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %114 = arith.maximumf %95#2, %113 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %115 = tt.expand_dims %114 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %116 = tt.broadcast %115 : (tensor<128x1xf32, #mfma>) -> tensor<128x64xf32, #mfma>
#     %117 = arith.subf %112, %116 : tensor<128x64xf32, #mfma>
#     %118 = tt.extern_elementwise %117 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x64xf32, #mfma>) -> tensor<128x64xf32, #mfma>
#     %119 = arith.subf %95#2, %114 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %120 = tt.extern_elementwise %119 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %121 = tt.expand_dims %120 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %122 = tt.broadcast %121 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %123 = arith.mulf %95#0, %122 : tensor<128x128xf32, #mfma>
#     %124 = tt.splat %c0_i64 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %125 = arith.addi %124, %81 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %126 = tt.expand_dims %125 {axis = 1 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi64, #blocked1>
#     %127 = tt.addptr %83, %126 : tensor<64x1x!tt.ptr<f16, 1>, #blocked1>, tensor<64x1xi64, #blocked1>
#     %128 = tt.broadcast %127 : (tensor<64x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<64x128x!tt.ptr<f16, 1>, #blocked1>
#     %129 = tt.addptr %128, %86 : tensor<64x128x!tt.ptr<f16, 1>, #blocked1>, tensor<64x128xi64, #blocked1>
#     %130 = arith.truncf %118 : tensor<128x64xf32, #mfma> to tensor<128x64xf16, #mfma>
#     %131 = triton_gpu.view_slice %130[0, 32] [128, 32] [1, 1] : tensor<128x64xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %132 = triton_gpu.view_slice %130[0, 0] [128, 32] [1, 1] : tensor<128x64xf16, #mfma> to tensor<128x32xf16, #mfma>
#     %133 = triton_gpu.convert_layout %131 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %134 = triton_gpu.convert_layout %132 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#     %135 = triton_gpu.view_slice %129[32, 0] [32, 128] [1, 1] : tensor<64x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %136 = triton_gpu.view_slice %129[0, 0] [32, 128] [1, 1] : tensor<64x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#     %137 = tt.load %135 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     %138 = tt.load %136 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#     gpu.barrier
#     %139 = triton_gpu.convert_layout %137 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#     %140 = triton_gpu.convert_layout %138 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#     gpu.barrier
#     %141 = triton_gpu.convert_layout %139 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#     %142 = triton_gpu.convert_layout %140 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#     %143 = tt.dot %134, %142, %123 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#     %144 = tt.dot %133, %141, %143 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#     %145 = "tt.reduce"(%118) <{axis = 1 : i32}> ({
#     ^bb0(%arg21: f32, %arg22: f32):
#       %160 = arith.addf %arg21, %arg22 : f32
#       tt.reduce.return %160 : f32
#     }) : (tensor<128x64xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %146 = arith.mulf %95#1, %120 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %147 = arith.addf %146, %145 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %148 = tt.expand_dims %147 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %149 = tt.broadcast %148 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %150 = arith.divf %144, %149 : tensor<128x128xf32, #mfma>
#     %151 = tt.addptr %arg5, %11 : !tt.ptr<f16, 1>, i32
#     %152 = arith.extsi %arg17 : i32 to i64
#     %153 = arith.truncf %150 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     %154 = tt.splat %152 : (i64) -> tensor<128x1xi64, #mfma>
#     %155 = arith.muli %70, %154 : tensor<128x1xi64, #mfma>
#     %156 = tt.splat %151 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
#     %157 = tt.addptr %156, %155 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
#     %158 = tt.broadcast %157 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
#     %159 = tt.addptr %158, %73 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
#     tt.store %159, %153 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
#     tt.return
#   }
# }


#     """

# OVERLAPS WITH REDUCE
# ir = """
# #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #mfma = #triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
# #shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared2 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
#   tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
#     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
#     %c128_i64 = arith.constant 128 : i64
#     %c0_i64 = arith.constant 0 : i64
#     %c128_i32 = arith.constant 128 : i32
#     %c0_i32 = arith.constant 0 : i32
#     %cst_2 = arith.constant 1.44269502 : f32
#     %0 = arith.mulf %arg3, %cst_2 : f32
#     %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
#     %2 = triton_gpu.view_slice %1[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %3 = triton_gpu.view_slice %1[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %4 = triton_gpu.view_slice %1[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %5 = triton_gpu.view_slice %1[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %6 = tt.get_program_id x : i32
#     %7 = arith.muli %6, %c128_i32 : i32
#     %8 = arith.extsi %7 : i32 to i64
#     %9 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %10 = tt.get_program_id y : i32
#     %11 = arith.muli %10, %arg7 : i32
#     %12 = tt.addptr %arg0, %11 : !tt.ptr<f16, 1>, i32
#     %13 = tt.splat %12 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
#     %14 = arith.extsi %arg8 : i32 to i64
#     %15 = tt.splat %14 : (i64) -> tensor<128x1xi64, #blocked>
#     %16 = tt.addptr %arg1, %11 : !tt.ptr<f16, 1>, i32
#     %17 = arith.extsi %arg11 : i32 to i64
#     %18 = tt.addptr %arg2, %11 : !tt.ptr<f16, 1>, i32
#     %19 = arith.extsi %arg14 : i32 to i64
#     %20 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %22 = arith.extsi %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %23 = arith.addi %9, %22 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %24 = tt.expand_dims %23 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
#     %25 = arith.muli %24, %15 : tensor<128x1xi64, #blocked>
#     %26 = tt.addptr %13, %25 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
#     %27 = tt.broadcast %26 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
#     %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %30 = arith.extsi %29 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %31 = tt.expand_dims %30 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
#     %32 = tt.broadcast %31 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
#     %33 = tt.addptr %27, %32 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
#     %34 = triton_gpu.view_slice %33[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %35 = triton_gpu.view_slice %33[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %36 = triton_gpu.view_slice %33[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %37 = triton_gpu.view_slice %33[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %38 = tt.load %34 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %39 = tt.load %35 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %40 = tt.load %36 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %41 = tt.load %37 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %42 = arith.extf %38 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %43 = arith.extf %39 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %44 = arith.extf %40 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %45 = arith.extf %41 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %46 = arith.mulf %42, %2 : tensor<128x32xf32, #blocked>
#     %47 = arith.mulf %43, %3 : tensor<128x32xf32, #blocked>
#     %48 = arith.mulf %44, %4 : tensor<128x32xf32, #blocked>
#     %49 = arith.mulf %45, %5 : tensor<128x32xf32, #blocked>
#     %50 = arith.truncf %46 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %51 = arith.truncf %47 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %52 = arith.truncf %48 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %53 = arith.truncf %49 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %54 = triton_gpu.convert_layout %50 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %55 = triton_gpu.convert_layout %51 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %56 = triton_gpu.convert_layout %52 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %57 = triton_gpu.convert_layout %53 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     gpu.barrier
#     %58 = triton_gpu.convert_layout %54 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %59 = triton_gpu.convert_layout %55 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %60 = triton_gpu.convert_layout %56 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %61 = triton_gpu.convert_layout %57 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %63 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %64 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %65 = arith.extsi %28 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %66 = arith.extsi %62 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %67 = arith.extsi %63 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %68 = arith.extsi %64 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %69 = arith.addi %20, %65 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %70 = tt.expand_dims %69 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
#     %71 = tt.expand_dims %66 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
#     %72 = tt.expand_dims %67 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#     %73 = tt.broadcast %71 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
#     %74 = tt.expand_dims %68 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#     %75 = tt.splat %16 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %76 = tt.addptr %75, %74 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#     %77 = tt.broadcast %76 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#     %78 = tt.splat %17 : (i64) -> tensor<1x128xi64, #blocked1>
#     %79 = tt.splat %18 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %80 = tt.splat %19 : (i64) -> tensor<1x128xi64, #blocked1>
#     %81 = arith.muli %72, %80 : tensor<1x128xi64, #blocked1>
#     %82 = tt.broadcast %81 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %83:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
#       %96 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %97 = arith.addi %96, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %98 = tt.expand_dims %97 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#       %99 = arith.muli %98, %78 : tensor<1x128xi64, #blocked1>
#       %100 = tt.broadcast %99 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#       %101 = tt.addptr %77, %100 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %102 = triton_gpu.view_slice %101[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %103 = triton_gpu.view_slice %101[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %104 = triton_gpu.view_slice %101[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %105 = triton_gpu.view_slice %101[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %106 = tt.load %105 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %107 = triton_gpu.convert_layout %106 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %108 = tt.load %104 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %109 = triton_gpu.convert_layout %107 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %110 = tt.dot %61, %109, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %111 = triton_gpu.convert_layout %108 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %112 = tt.load %103 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %113 = triton_gpu.convert_layout %111 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %114 = tt.dot %60, %113, %110 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %115 = triton_gpu.convert_layout %112 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %116 = tt.load %102 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %117 = triton_gpu.convert_layout %115 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %118 = tt.dot %59, %117, %114 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %119 = triton_gpu.convert_layout %116 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       gpu.barrier
#       %120 = triton_gpu.convert_layout %119 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %121 = tt.dot %58, %120, %118 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       %122 = "tt.reduce"(%121) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %173 = arith.maximumf %arg27, %arg28 : f32
#         tt.reduce.return %173 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %123 = arith.maximumf %arg24, %122 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %124 = tt.expand_dims %123 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %125 = tt.broadcast %124 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %126 = arith.subf %121, %125 : tensor<128x128xf32, #mfma>
#       %127 = tt.extern_elementwise %126 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x128xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %168 = "tt.reduce"(%127) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %173 = arith.addf %arg27, %arg28 : f32
#         tt.reduce.return %173 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

#       %128 = arith.subf %arg24, %123 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %129 = tt.extern_elementwise %128 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %130 = tt.expand_dims %129 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %131 = tt.broadcast %130 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %132 = arith.mulf %arg22, %131 : tensor<128x128xf32, #mfma>
#       %133 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %134 = arith.addi %133, %68 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %135 = tt.expand_dims %134 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#       %136 = tt.addptr %79, %135 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#       %137 = tt.broadcast %136 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#       %138 = tt.addptr %137, %82 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %139 = arith.truncf %127 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#       %140 = triton_gpu.view_slice %139[0, 32] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %141 = triton_gpu.view_slice %139[0, 64] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %142 = triton_gpu.view_slice %139[0, 96] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %143 = triton_gpu.view_slice %139[0, 0] [128, 32] [1, 1] : tensor<128x128xf16, #mfma> to tensor<128x32xf16, #mfma>
#       %144 = triton_gpu.convert_layout %140 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %145 = triton_gpu.convert_layout %141 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %146 = triton_gpu.convert_layout %142 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %147 = triton_gpu.convert_layout %143 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %148 = triton_gpu.view_slice %138[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %149 = triton_gpu.view_slice %138[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %150 = triton_gpu.view_slice %138[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %151 = triton_gpu.view_slice %138[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>


#       %152 = tt.load %151 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %154 = tt.load %150 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %158 = tt.load %149 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %162 = tt.load %148 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>

#       gpu.barrier

#       %153 = triton_gpu.convert_layout %152 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %157 = triton_gpu.convert_layout %154 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %161 = triton_gpu.convert_layout %158 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %165 = triton_gpu.convert_layout %162 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>



#       gpu.barrier
#       %155 = triton_gpu.convert_layout %153 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %159 = triton_gpu.convert_layout %157 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %163 = triton_gpu.convert_layout %161 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %166 = triton_gpu.convert_layout %165 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>

#       %156 = tt.dot %147, %155, %132 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %160 = tt.dot %146, %159, %156 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>


#       %164 = tt.dot %145, %163, %160 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       %167 = tt.dot %144, %166, %164 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

#       %169 = arith.mulf %arg23, %129 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %170 = arith.addf %169, %168 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %171 = arith.addi %arg25, %c128_i64 : i64
#       %172 = arith.addi %arg26, %c128_i64 : i64
#       scf.yield %167, %170, %123, %171, %172 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
#     }
#     %84 = tt.expand_dims %83#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %85 = tt.broadcast %84 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %86 = arith.divf %83#0, %85 : tensor<128x128xf32, #mfma>
#     %87 = tt.addptr %arg5, %11 : !tt.ptr<f16, 1>, i32
#     %88 = arith.extsi %arg17 : i32 to i64
#     %89 = arith.truncf %86 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     %90 = tt.splat %88 : (i64) -> tensor<128x1xi64, #mfma>
#     %91 = arith.muli %70, %90 : tensor<128x1xi64, #mfma>
#     %92 = tt.splat %87 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
#     %93 = tt.addptr %92, %91 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
#     %94 = tt.broadcast %93 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
#     %95 = tt.addptr %94, %73 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
#     tt.store %95, %89 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
#     tt.return
#   }
# }

# """


# ir = """
# #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
# #mfma = #triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
# #shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# #shared2 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
# module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
#   tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
#     %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
#     %c128_i64 = arith.constant 128 : i64
#     %c0_i64 = arith.constant 0 : i64
#     %c128_i32 = arith.constant 128 : i32
#     %c0_i32 = arith.constant 0 : i32
#     %cst_2 = arith.constant 1.44269502 : f32
#     %0 = arith.mulf %arg3, %cst_2 : f32
#     %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
#     %2 = triton_gpu.view_slice %1[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %3 = triton_gpu.view_slice %1[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %4 = triton_gpu.view_slice %1[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %5 = triton_gpu.view_slice %1[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
#     %6 = tt.get_program_id x : i32
#     %7 = arith.muli %6, %c128_i32 : i32
#     %8 = arith.extsi %7 : i32 to i64
#     %9 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %10 = tt.get_program_id y : i32
#     %11 = arith.muli %10, %arg7 : i32
#     %12 = tt.addptr %arg0, %11 : !tt.ptr<f16, 1>, i32
#     %13 = tt.splat %12 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
#     %14 = arith.extsi %arg8 : i32 to i64
#     %15 = tt.splat %14 : (i64) -> tensor<128x1xi64, #blocked>
#     %16 = tt.addptr %arg1, %11 : !tt.ptr<f16, 1>, i32
#     %17 = arith.extsi %arg11 : i32 to i64
#     %18 = tt.addptr %arg2, %11 : !tt.ptr<f16, 1>, i32
#     %19 = arith.extsi %arg14 : i32 to i64
#     %20 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %22 = arith.extsi %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %23 = arith.addi %9, %22 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
#     %24 = tt.expand_dims %23 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
#     %25 = arith.muli %24, %15 : tensor<128x1xi64, #blocked>
#     %26 = tt.addptr %13, %25 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
#     %27 = tt.broadcast %26 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
#     %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %30 = arith.extsi %29 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
#     %31 = tt.expand_dims %30 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
#     %32 = tt.broadcast %31 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
#     %33 = tt.addptr %27, %32 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
#     %34 = triton_gpu.view_slice %33[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %35 = triton_gpu.view_slice %33[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %36 = triton_gpu.view_slice %33[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %37 = triton_gpu.view_slice %33[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
#     %38 = tt.load %34 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %39 = tt.load %35 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %40 = tt.load %36 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %41 = tt.load %37 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
#     %42 = arith.extf %38 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %43 = arith.extf %39 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %44 = arith.extf %40 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %45 = arith.extf %41 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
#     %46 = arith.mulf %42, %2 : tensor<128x32xf32, #blocked>
#     %47 = arith.mulf %43, %3 : tensor<128x32xf32, #blocked>
#     %48 = arith.mulf %44, %4 : tensor<128x32xf32, #blocked>
#     %49 = arith.mulf %45, %5 : tensor<128x32xf32, #blocked>
#     %50 = arith.truncf %46 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %51 = arith.truncf %47 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %52 = arith.truncf %48 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %53 = arith.truncf %49 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
#     %54 = triton_gpu.convert_layout %50 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %55 = triton_gpu.convert_layout %51 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %56 = triton_gpu.convert_layout %52 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     %57 = triton_gpu.convert_layout %53 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
#     gpu.barrier
#     %58 = triton_gpu.convert_layout %54 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %59 = triton_gpu.convert_layout %55 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %60 = triton_gpu.convert_layout %56 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %61 = triton_gpu.convert_layout %57 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
#     %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %63 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %64 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %65 = arith.extsi %28 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %66 = arith.extsi %62 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
#     %67 = arith.extsi %63 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#     %68 = arith.extsi %64 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#     %69 = arith.addi %20, %65 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#     %70 = tt.expand_dims %69 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
#     %71 = tt.expand_dims %66 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
#     %72 = tt.expand_dims %67 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#     %73 = tt.broadcast %71 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
#     %74 = tt.expand_dims %68 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#     %75 = tt.splat %16 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %76 = tt.addptr %75, %74 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#     %77 = tt.broadcast %76 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#     %78 = tt.splat %17 : (i64) -> tensor<1x128xi64, #blocked1>
#     %79 = tt.splat %18 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
#     %80 = tt.splat %19 : (i64) -> tensor<1x128xi64, #blocked1>
#     %81 = arith.muli %72, %80 : tensor<1x128xi64, #blocked1>
#     %82 = tt.broadcast %81 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#     %83:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
#       %96 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %97 = arith.addi %96, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
#       %98 = tt.expand_dims %97 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
#       %99 = arith.muli %98, %78 : tensor<1x128xi64, #blocked1>
#       %100 = tt.broadcast %99 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
#       %101 = tt.addptr %77, %100 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %102 = triton_gpu.view_slice %101[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %103 = triton_gpu.view_slice %101[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %104 = triton_gpu.view_slice %101[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %105 = triton_gpu.view_slice %101[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %106 = tt.load %105 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %107 = triton_gpu.convert_layout %106 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %108 = tt.load %104 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %109 = triton_gpu.convert_layout %107 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %110 = tt.dot %61, %109, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %111 = triton_gpu.convert_layout %108 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %112 = tt.load %103 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %113 = triton_gpu.convert_layout %111 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %114 = tt.dot %60, %113, %110 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %115 = triton_gpu.convert_layout %112 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       %116 = tt.load %102 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       gpu.barrier
#       %117 = triton_gpu.convert_layout %115 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %118 = tt.dot %59, %117, %114 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
#       gpu.barrier
#       %119 = triton_gpu.convert_layout %116 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
#       gpu.barrier
#       %120 = triton_gpu.convert_layout %119 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
#       %121 = tt.dot %58, %120, %118 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>

#       %133 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %134 = arith.addi %133, %68 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
#       %135 = tt.expand_dims %134 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
#       %136 = tt.addptr %79, %135 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
#       %137 = tt.broadcast %136 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
#       %138 = tt.addptr %137, %82 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
#       %151 = triton_gpu.view_slice %138[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %602 = triton_gpu.view_slice %138[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %603 = triton_gpu.view_slice %138[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
#       %150 = triton_gpu.view_slice %138[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>

#       %152 = tt.load %151 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>


#       rocdl.sched.barrier 0
#       %122 = "tt.reduce"(%121) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %173 = arith.maximumf %arg27, %arg28 : f32
#         tt.reduce.return %173 : f32
#       }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

#       %123 = arith.maximumf %arg24, %122 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %128 = arith.subf %arg24, %123 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %124 = tt.expand_dims %123 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %125 = tt.broadcast %124 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>

#       rocdl.sched.barrier 0
#       gpu.barrier
#       %154 = tt.load %602 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       %153 = triton_gpu.convert_layout %152 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>

#       %126 = arith.subf %121, %125 : tensor<128x128xf32, #mfma>
#       %143 = triton_gpu.view_slice %126[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #mfma> to tensor<128x32xf32, #mfma>
#       %140 = triton_gpu.view_slice %126[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #mfma> to tensor<128x32xf32, #mfma>
#       %141 = triton_gpu.view_slice %126[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #mfma> to tensor<128x32xf32, #mfma>
#       %142 = triton_gpu.view_slice %126[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #mfma> to tensor<128x32xf32, #mfma>

#       rocdl.sched.barrier 0
#       gpu.barrier
#       %200 = tt.extern_elementwise %143 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x32xf32, #mfma>) -> tensor<128x32xf32, #mfma>
#       %204 = arith.truncf %200 : tensor<128x32xf32, #mfma> to tensor<128x32xf16, #mfma>
#       %300 = "tt.reduce"(%200) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %401 = arith.addf %arg27, %arg28 : f32
#         tt.reduce.return %401 : f32
#       }) : (tensor<128x32xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

#       %208 = triton_gpu.convert_layout %204 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %129 = tt.extern_elementwise %128 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %155 = triton_gpu.convert_layout %153 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>

#       %130 = tt.expand_dims %129 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#       %131 = tt.broadcast %130 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#       %132 = arith.mulf %arg22, %131 : tensor<128x128xf32, #mfma>

#       %156 = tt.dot %208, %155, %132 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       rocdl.sched.barrier 0
#       gpu.barrier

#       %157 = triton_gpu.convert_layout %154 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %158 = tt.load %603 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
#       rocdl.sched.barrier 0
#       gpu.barrier

#       %201 = tt.extern_elementwise %140 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x32xf32, #mfma>) -> tensor<128x32xf32, #mfma>
#       %205 = arith.truncf %201 : tensor<128x32xf32, #mfma> to tensor<128x32xf16, #mfma>
#       %301 = "tt.reduce"(%201) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %402 = arith.addf %arg27, %arg28 : f32
#         tt.reduce.return %402 : f32
#       }) : (tensor<128x32xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %304 = arith.addf %300, %301 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

#       %209 = triton_gpu.convert_layout %205 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %159 = triton_gpu.convert_layout %157 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %160 = tt.dot %209, %159, %156 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>

#       rocdl.sched.barrier 0
#       gpu.barrier
#       %161 = triton_gpu.convert_layout %158 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       %162 = tt.load %150 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>

#       rocdl.sched.barrier 0
#       gpu.barrier
#       %202 = tt.extern_elementwise %141 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x32xf32, #mfma>) -> tensor<128x32xf32, #mfma>
#       %206 = arith.truncf %202 : tensor<128x32xf32, #mfma> to tensor<128x32xf16, #mfma>
#       %302 = "tt.reduce"(%202) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %403 = arith.addf %arg27, %arg28 : f32
#         tt.reduce.return %403 : f32
#       }) : (tensor<128x32xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

#       %210 = triton_gpu.convert_layout %206 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %163 = triton_gpu.convert_layout %161 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %164 = tt.dot %210, %163, %160 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
#       rocdl.sched.barrier 0
#       gpu.barrier

#       %165 = triton_gpu.convert_layout %162 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
#       gpu.barrier
#       %203 = tt.extern_elementwise %142 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x32xf32, #mfma>) -> tensor<128x32xf32, #mfma>
#       %207 = arith.truncf %203 : tensor<128x32xf32, #mfma> to tensor<128x32xf16, #mfma>
#       %303 = "tt.reduce"(%203) <{axis = 1 : i32}> ({
#       ^bb0(%arg27: f32, %arg28: f32):
#         %404 = arith.addf %arg27, %arg28 : f32
#         tt.reduce.return %404 : f32
#       }) : (tensor<128x32xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %305 = arith.addf %302, %303 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %211 = triton_gpu.convert_layout %207 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
#       %166 = triton_gpu.convert_layout %165 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
#       %167 = tt.dot %211, %166, %164 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>




      
#       %306 = arith.addf %304, %305 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

#       %169 = arith.mulf %arg23, %129 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %170 = arith.addf %169, %306 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
#       %171 = arith.addi %arg25, %c128_i64 : i64
#       %172 = arith.addi %arg26, %c128_i64 : i64
#       scf.yield %167, %170, %123, %171, %172 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
#     }
#     %84 = tt.expand_dims %83#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
#     %85 = tt.broadcast %84 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
#     %86 = arith.divf %83#0, %85 : tensor<128x128xf32, #mfma>
#     %87 = tt.addptr %arg5, %11 : !tt.ptr<f16, 1>, i32
#     %88 = arith.extsi %arg17 : i32 to i64
#     %89 = arith.truncf %86 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
#     %90 = tt.splat %88 : (i64) -> tensor<128x1xi64, #mfma>
#     %91 = arith.muli %70, %90 : tensor<128x1xi64, #mfma>
#     %92 = tt.splat %87 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
#     %93 = tt.addptr %92, %91 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
#     %94 = tt.broadcast %93 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
#     %95 = tt.addptr %94, %73 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
#     tt.store %95, %89 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
#     tt.return
#   }
# }

# """




ir = """
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mfma = #triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared2 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @_attn_fwd_0d1d2d34d5d6de7de8de9c10de11de12de13c14de15de16de17c18de19de20de21c2223de24de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mfma>
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 1.44269502 : f32
    %0 = arith.mulf %arg3, %cst_2 : f32
    %1 = tt.splat %0 : (f32) -> tensor<128x128xf32, #blocked>
    %2 = triton_gpu.view_slice %1[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %3 = triton_gpu.view_slice %1[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %4 = triton_gpu.view_slice %1[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %5 = triton_gpu.view_slice %1[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #blocked> to tensor<128x32xf32, #blocked>
    %6 = tt.get_program_id x : i32
    %7 = arith.muli %6, %c128_i32 : i32
    %8 = arith.extsi %7 : i32 to i64
    %9 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %10 = tt.get_program_id y : i32
    %11 = arith.muli %10, %arg7 : i32
    %12 = tt.addptr %arg0, %11 : !tt.ptr<f16, 1>, i32
    %13 = tt.splat %12 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked>
    %14 = arith.extsi %arg8 : i32 to i64
    %15 = tt.splat %14 : (i64) -> tensor<128x1xi64, #blocked>
    %16 = tt.addptr %arg1, %11 : !tt.ptr<f16, 1>, i32
    %17 = arith.extsi %arg11 : i32 to i64
    %18 = tt.addptr %arg2, %11 : !tt.ptr<f16, 1>, i32
    %19 = arith.extsi %arg14 : i32 to i64
    %20 = tt.splat %8 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.extsi %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %23 = arith.addi %9, %22 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi64, #blocked>
    %25 = arith.muli %24, %15 : tensor<128x1xi64, #blocked>
    %26 = tt.addptr %13, %25 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi64, #blocked>
    %27 = tt.broadcast %26 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked>
    %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %30 = arith.extsi %29 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %31 = tt.expand_dims %30 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %32 = tt.broadcast %31 : (tensor<1x128xi64, #blocked>) -> tensor<128x128xi64, #blocked>
    %33 = tt.addptr %27, %32 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi64, #blocked>
    %34 = triton_gpu.view_slice %33[0, 96] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
    %35 = triton_gpu.view_slice %33[0, 64] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
    %36 = triton_gpu.view_slice %33[0, 32] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
    %37 = triton_gpu.view_slice %33[0, 0] [128, 32] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked> to tensor<128x32x!tt.ptr<f16, 1>, #blocked>
    %38 = tt.load %34 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
    %39 = tt.load %35 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
    %40 = tt.load %36 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
    %41 = tt.load %37 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
    %42 = arith.extf %38 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
    %43 = arith.extf %39 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
    %44 = arith.extf %40 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
    %45 = arith.extf %41 : tensor<128x32xf16, #blocked> to tensor<128x32xf32, #blocked>
    %46 = arith.mulf %42, %2 : tensor<128x32xf32, #blocked>
    %47 = arith.mulf %43, %3 : tensor<128x32xf32, #blocked>
    %48 = arith.mulf %44, %4 : tensor<128x32xf32, #blocked>
    %49 = arith.mulf %45, %5 : tensor<128x32xf32, #blocked>
    %50 = arith.truncf %46 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
    %51 = arith.truncf %47 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
    %52 = arith.truncf %48 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
    %53 = arith.truncf %49 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
    %54 = triton_gpu.convert_layout %50 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
    %55 = triton_gpu.convert_layout %51 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
    %56 = triton_gpu.convert_layout %52 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
    %57 = triton_gpu.convert_layout %53 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #shared>
    gpu.barrier
    %58 = triton_gpu.convert_layout %54 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    %59 = triton_gpu.convert_layout %55 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    %60 = triton_gpu.convert_layout %56 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    %61 = triton_gpu.convert_layout %57 : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
    %63 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %64 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %65 = arith.extsi %28 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %66 = arith.extsi %62 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mfma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>
    %67 = arith.extsi %63 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %68 = arith.extsi %64 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %69 = arith.addi %20, %65 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %70 = tt.expand_dims %69 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xi64, #mfma>
    %71 = tt.expand_dims %66 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mfma}>>) -> tensor<1x128xi64, #mfma>
    %72 = tt.expand_dims %67 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
    %73 = tt.broadcast %71 : (tensor<1x128xi64, #mfma>) -> tensor<128x128xi64, #mfma>
    %74 = tt.expand_dims %68 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
    %75 = tt.splat %16 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %76 = tt.addptr %75, %74 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
    %77 = tt.broadcast %76 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
    %78 = tt.splat %17 : (i64) -> tensor<1x128xi64, #blocked1>
    %79 = tt.splat %18 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %80 = tt.splat %19 : (i64) -> tensor<1x128xi64, #blocked1>
    %81 = arith.muli %72, %80 : tensor<1x128xi64, #blocked1>
    %82 = tt.broadcast %81 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
    %83:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_1, %arg23 = %cst, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64)  : i32 {
      %96 = tt.splat %arg26 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %97 = arith.addi %96, %67 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %98 = tt.expand_dims %97 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
      %99 = arith.muli %98, %78 : tensor<1x128xi64, #blocked1>
      %100 = tt.broadcast %99 : (tensor<1x128xi64, #blocked1>) -> tensor<128x128xi64, #blocked1>
      %101 = tt.addptr %77, %100 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
      %102 = triton_gpu.view_slice %101[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %103 = triton_gpu.view_slice %101[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %104 = triton_gpu.view_slice %101[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %105 = triton_gpu.view_slice %101[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %106 = tt.load %105 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      gpu.barrier
      %107 = triton_gpu.convert_layout %106 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
      %108 = tt.load %104 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      gpu.barrier
      %109 = triton_gpu.convert_layout %107 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
      %110 = tt.dot %61, %109, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
      gpu.barrier
      %111 = triton_gpu.convert_layout %108 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
      %112 = tt.load %103 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      gpu.barrier
      %113 = triton_gpu.convert_layout %111 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
      %114 = tt.dot %60, %113, %110 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
      gpu.barrier
      %115 = triton_gpu.convert_layout %112 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
      %116 = tt.load %102 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      gpu.barrier
      %117 = triton_gpu.convert_layout %115 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
      %118 = tt.dot %59, %117, %114 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>
      gpu.barrier
      %119 = triton_gpu.convert_layout %116 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared1>
      gpu.barrier
      %120 = triton_gpu.convert_layout %119 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
      %121 = tt.dot %58, %120, %118 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<128x128xf32, #mfma>

      %133 = tt.splat %arg25 : (i64) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %134 = arith.addi %133, %68 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %135 = tt.expand_dims %134 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi64, #blocked1>
      %136 = tt.addptr %79, %135 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi64, #blocked1>
      %137 = tt.broadcast %136 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked1>
      %138 = tt.addptr %137, %82 : tensor<128x128x!tt.ptr<f16, 1>, #blocked1>, tensor<128x128xi64, #blocked1>
      %151 = triton_gpu.view_slice %138[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %602 = triton_gpu.view_slice %138[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %603 = triton_gpu.view_slice %138[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>
      %150 = triton_gpu.view_slice %138[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16, 1>, #blocked1> to tensor<32x128x!tt.ptr<f16, 1>, #blocked1>

      


      rocdl.sched.barrier 0
      %122 = "tt.reduce"(%121) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %173 = arith.maximumf %arg27, %arg28 : f32
        tt.reduce.return %173 : f32
      }) : (tensor<128x128xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

      %123 = arith.maximumf %arg24, %122 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %128 = arith.subf %arg24, %123 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %124 = tt.expand_dims %123 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %125 = tt.broadcast %124 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %126 = arith.subf %121, %125 : tensor<128x128xf32, #mfma>
      %143 = triton_gpu.view_slice %126[0, 0] [128, 32] [1, 1] : tensor<128x128xf32, #mfma> to tensor<128x32xf32, #mfma>
      %140 = triton_gpu.view_slice %126[0, 32] [128, 32] [1, 1] : tensor<128x128xf32, #mfma> to tensor<128x32xf32, #mfma>
      %141 = triton_gpu.view_slice %126[0, 64] [128, 32] [1, 1] : tensor<128x128xf32, #mfma> to tensor<128x32xf32, #mfma>
      %142 = triton_gpu.view_slice %126[0, 96] [128, 32] [1, 1] : tensor<128x128xf32, #mfma> to tensor<128x32xf32, #mfma>

      %200 = tt.extern_elementwise %143 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x32xf32, #mfma>) -> tensor<128x32xf32, #mfma>
      %204 = arith.truncf %200 : tensor<128x32xf32, #mfma> to tensor<128x32xf16, #mfma>
      %300 = "tt.reduce"(%200) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %401 = arith.addf %arg27, %arg28 : f32
        tt.reduce.return %401 : f32
      }) : (tensor<128x32xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

      %201 = tt.extern_elementwise %140 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x32xf32, #mfma>) -> tensor<128x32xf32, #mfma>
      %205 = arith.truncf %201 : tensor<128x32xf32, #mfma> to tensor<128x32xf16, #mfma>
      %301 = "tt.reduce"(%201) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %402 = arith.addf %arg27, %arg28 : f32
        tt.reduce.return %402 : f32
      }) : (tensor<128x32xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %304 = arith.addf %300, %301 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

      %202 = tt.extern_elementwise %141 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x32xf32, #mfma>) -> tensor<128x32xf32, #mfma>
      %206 = arith.truncf %202 : tensor<128x32xf32, #mfma> to tensor<128x32xf16, #mfma>
      %302 = "tt.reduce"(%202) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %403 = arith.addf %arg27, %arg28 : f32
        tt.reduce.return %403 : f32
      }) : (tensor<128x32xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

      %203 = tt.extern_elementwise %142 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128x32xf32, #mfma>) -> tensor<128x32xf32, #mfma>
      %207 = arith.truncf %203 : tensor<128x32xf32, #mfma> to tensor<128x32xf16, #mfma>
      %303 = "tt.reduce"(%203) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32, %arg28: f32):
        %404 = arith.addf %arg27, %arg28 : f32
        tt.reduce.return %404 : f32
      }) : (tensor<128x32xf32, #mfma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %305 = arith.addf %302, %303 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %306 = arith.addf %304, %305 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>

      %129 = tt.extern_elementwise %128 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %130 = tt.expand_dims %129 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
      %131 = tt.broadcast %130 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
      %132 = arith.mulf %arg22, %131 : tensor<128x128xf32, #mfma>

      %152 = tt.load %151 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      %154 = tt.load %602 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      %158 = tt.load %603 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      %162 = tt.load %150 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>

      gpu.barrier
      %153 = triton_gpu.convert_layout %152 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
      %157 = triton_gpu.convert_layout %154 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
      %161 = triton_gpu.convert_layout %158 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>
      %165 = triton_gpu.convert_layout %162 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #shared2>

      gpu.barrier      
      %155 = triton_gpu.convert_layout %153 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %159 = triton_gpu.convert_layout %157 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %163 = triton_gpu.convert_layout %161 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %166 = triton_gpu.convert_layout %165 : (tensor<32x128xf16, #shared2>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>



      %208 = triton_gpu.convert_layout %204 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %156 = tt.dot %208, %155, %132 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      %209 = triton_gpu.convert_layout %205 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %160 = tt.dot %209, %159, %156 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>


      %210 = triton_gpu.convert_layout %206 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %164 = tt.dot %210, %163, %160 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>
      rocdl.sched.barrier 0
      gpu.barrier

      gpu.barrier
      %211 = triton_gpu.convert_layout %207 : (tensor<128x32xf16, #mfma>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %167 = tt.dot %211, %166, %164 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<128x128xf32, #mfma>


      %169 = arith.mulf %arg23, %129 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %170 = arith.addf %169, %306 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %171 = arith.addi %arg25, %c128_i64 : i64
      %172 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %167, %170, %123, %171, %172 : tensor<128x128xf32, #mfma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, i64, i64
    }
    %84 = tt.expand_dims %83#1 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<128x1xf32, #mfma>
    %85 = tt.broadcast %84 : (tensor<128x1xf32, #mfma>) -> tensor<128x128xf32, #mfma>
    %86 = arith.divf %83#0, %85 : tensor<128x128xf32, #mfma>
    %87 = tt.addptr %arg5, %11 : !tt.ptr<f16, 1>, i32
    %88 = arith.extsi %arg17 : i32 to i64
    %89 = arith.truncf %86 : tensor<128x128xf32, #mfma> to tensor<128x128xf16, #mfma>
    %90 = tt.splat %88 : (i64) -> tensor<128x1xi64, #mfma>
    %91 = arith.muli %70, %90 : tensor<128x1xi64, #mfma>
    %92 = tt.splat %87 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #mfma>
    %93 = tt.addptr %92, %91 : tensor<128x1x!tt.ptr<f16, 1>, #mfma>, tensor<128x1xi64, #mfma>
    %94 = tt.broadcast %93 : (tensor<128x1x!tt.ptr<f16, 1>, #mfma>) -> tensor<128x128x!tt.ptr<f16, 1>, #mfma>
    %95 = tt.addptr %94, %73 : tensor<128x128x!tt.ptr<f16, 1>, #mfma>, tensor<128x128xi64, #mfma>
    tt.store %95, %89 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #mfma>
    tt.return
  }
}

"""
# Pick the fp8 data type

# AMD E5M2B16
# float8:tl.constexpr = torch.float8_e5m2fnuz

# AMD E4M3B8
# Note: When picking this f8 data type, scaling is required when using f8
# for the second gemm
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8:tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
    f.write(ir)
    f.flush()
    kernel = triton.compile(f.name)

@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qkv_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(0, 1)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    # it's even better to multiply the qk_scale and convert to f16
    # than doing it inside the loop
    # So conversion is quite cheap
    q = (q * qk_scale).to(q.dtype)
    lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        if pre_load_v:
            v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        #qk = (qk * qk_scale)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(v.dtype), v)
        # -- update m_i and l_i
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    acc = acc / l_i[:, None]
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-2]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q, dtype=v.dtype)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8

        ## hardcoded best perf_configs for MI250
        if Lk == 64:
            ## D_HEAD = 64
            BLOCK_M = 128
            BLOCK_N = 64
            waves_per_eu = 3
            num_warps = 4
            num_stages = 1
            ## causal=False likes to pre load v but causal=True does not
            pre_load_v = False if causal else True
            slice_k_tile = 32
            kpack = 1
        else:
            ## D_HEAD = 128
            ## For fp16, pick BLOCK_M=256, num_warps=8
            ## For fp8, pick BLOCK_M=128, num_warps=4
            ## TODO (zhanglx): add tuning infra for FA
            BLOCK_M = 128 #if TORCH_HAS_FP8E4 and q.dtype == torch.float8_e4m3fnuz else 256
            BLOCK_N = 128
            waves_per_eu = 2
            num_warps = 4
            num_stages = 1
            pre_load_v = False
            slice_k_tile = 32
            kpack = 2

        grid = ( triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)




        kernel[(32, 192, 1)](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2), 
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            q.shape[0], q.shape[1],
            q.shape[2],
            # BLOCK_DMODEL=Lk,
            # BLOCK_M = BLOCK_M,
            # BLOCK_N = BLOCK_N,
            # waves_per_eu = waves_per_eu,
            # num_warps = num_warps,
            # num_stages = num_stages,
            # pre_load_v = pre_load_v,
            # slice_k_tile = slice_k_tile,
            # kpack = kpack,
        )


        return o


attention = _attention.apply

name_to_torch_types = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp8': float8
}

@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD, dtype',
[ (*shape, dtype)
    for shape in [(4, 48, 4096, 128)]
    for dtype in ['fp16']])
def test_op_fwd(Z, H, N_CTX, D_HEAD, dtype):
    torch.manual_seed(20)
    init_dtype = torch.float16 if dtype == 'fp8' else name_to_torch_types[dtype]
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    ref_out = torch.matmul(p, v.transpose(2,3))
    # triton implementation
    # q,k casting for partial fp8
    q = q.to(name_to_torch_types[dtype])
    k = k.to(name_to_torch_types[dtype])
    dout = torch.randn_like(q, dtype=torch.float16)
    tri_out = attention(q, k, v, sm_scale)
    # compare
    atol = 1.4e-1 if dtype == 'fp8' else 1e-2
    rtol = 1e-2 if dtype == 'fp8' else 3e-3
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None

# vary seq length for fixed head and batch=4
configs = []
for dtype in ['fp16']:
    for D_HEAD in [128]:
        for causal in [False]:
            configs.append(triton.testing.Benchmark(
                x_names=['BATCH', 'H','N_CTX'],
                x_vals=[#(16, 16, 1024),
                        # (8, 16, 2048),
                        # (4, 16, 4096),
                        # (2, 16, 8192),
                        # (1, 16, 16384),
                        # (4, 48, 1024),
                        # (4, 48, 2048),
                        (4, 48, 4096),
                        # (4, 48, 8192),
                        # (4, 48, 16384),
                        ],
                line_arg='provider',
                line_vals=['triton'],
                line_names=['Triton'],
                #styles=[('red', '-'), ('blue', '-')],
                ylabel='ms',
                plot_name=f'fused-attention-fwd-d{D_HEAD}-causal={causal}-{dtype}',
                args={
                    'D_HEAD': D_HEAD,
                    'dtype': dtype,
                    'causal': causal})
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, provider, dtype, device="cuda"):
    if dtype == 'fp8' and not TORCH_HAS_FP8E4:
        sys.exit("fp8 is not available")
    warmup = 25
    rep = 100
    init_dtype = torch.float16 if dtype != 'bf16' else torch.bfloat16
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
    # q,k casting for partial fp8
    q = q.to(name_to_torch_types[dtype])
    k = k.to(name_to_torch_types[dtype])
    fn = lambda: attention(q, k, v, sm_scale)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    return total_flops / ms * 1e-9


def main():
    bench_flash_attention.run(save_path='.', print_data=True)

if __name__ == '__main__':
    sys.exit(main())

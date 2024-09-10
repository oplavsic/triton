#ifndef AMDGPU_CONVERSION_PASSES_H
#define AMDGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "amd/include/AMDGPUToLLVM/AMDGPUToLLVMPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "AMD/include/AMDGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif

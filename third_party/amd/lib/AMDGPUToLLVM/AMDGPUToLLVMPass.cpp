#include "AMDGPUToLLVM/AMDGPUToLLVMPass.h"

#include "Dialect/AMDGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "AMDGPUToLLVM/Passes.h.inc"

namespace {

class ConvertAMDGPUToLLVM
    : public ConvertAMDGPUToLLVMBase<ConvertAMDGPUToLLVM> {

public:
  explicit ConvertAMDGPUToLLVM() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

    patterns.add<OpConversionPattern<DummyOp>>(
        &getContext(),
        [&](NoOp op, ArrayRef<Value> operands,
            ConversionPatternRewriter &rewriter) -> LogicalResult {
          rewriter.eraseOp(op);
          return success();
        });

    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};
}
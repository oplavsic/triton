#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = triton::gpu;

static Type getNewType(Type type, Attribute encoding) {
  RankedTensorType tensorType = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

void convertLayout(Attribute encoding, Operation *op) {
  OpBuilder builder(op);
  // Convert operands
  // For load/store with tensor pointers, we don't have to change the
  // operands' type, we do this by changing the outputs' type of
  // `make_tensor_ptr`
  SmallVector<Value, 4> newArgs;
  for (auto operand : op->getOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (tensorType &&
        !isa<ttg::SharedEncodingAttr>(tensorType.getEncoding())) {
      Type newType = getNewType(tensorType, encoding);
      newArgs.push_back(builder.create<ttg::ConvertLayoutOp>(
          op->getLoc(), newType, operand));
    } else {
      newArgs.push_back(operand);
    }
  }

  // Convert output types
  SmallVector<Type, 4> newTypes;
  for (auto t : op->getResultTypes()) {
    bool isAsync = isa<ttg::AsyncCopyGlobalToLocalOp>(op);
    newTypes.push_back(isAsync ? t : getNewType(t, encoding));
  }

  // Construct new op with the new encoding
  Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(),
                                    newArgs, newTypes, op->getAttrs());

  // Cast the results back to the original layout
  for (size_t i = 0; i < op->getNumResults(); i++) {
    Value newResult = newOp->getResult(i);
    if (newTypes[i] != op->getResultTypes()[i]) {
      newResult = builder.create<ttg::ConvertLayoutOp>(
          op->getLoc(), op->getResult(i).getType(), newResult);
    }
    op->getResult(i).replaceAllUsesWith(newResult);
  }
  op->erase();
}

std::optional<triton::LoadOp> getLoadInst(Operation *op, ModuleOp &mod) {
  SmallVector<triton::LoadOp> loadOpsVec;

  mod.dump();
  mod.walk([&](triton::LoadOp loadOp) {
    SetVector<Operation *> forwardSlices;
    getForwardSlice((Operation *)loadOp, &forwardSlices);
    if (std::find(forwardSlices.begin(), forwardSlices.end(), op) !=
        forwardSlices.end()) {
      loadOp.dump();
      loadOpsVec.push_back(loadOp);
    }
  });

  assert(loadOpsVec.size() == 1);
  if (loadOpsVec.empty()) {
    return std::nullopt;
  }
  return loadOpsVec[0];
}


class TritonAMDGPUBypassLDSForDotLayoutPass
    : public TritonAMDGPUBypassLDSForDotLayoutBase<
          TritonAMDGPUBypassLDSForDotLayoutPass> {

public:
  TritonAMDGPUBypassLDSForDotLayoutPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    std::vector<ttg::ConvertLayoutOp> cvtVec;

    m.walk([&](Operation *op) {
      auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(op);
      if (!cvtOp) {
        return;
      }

      auto srcType = cast<RankedTensorType>(cvtOp.getOperand().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());

      if (srcType.getShape().size() != 2) {
        return;
      }

      auto srcBlocked =
          dyn_cast<ttg::BlockedEncodingAttr>(srcType.getEncoding());
      auto dstDotOp =
          dyn_cast<ttg::DotOperandEncodingAttr>(dstType.getEncoding());

      if (!(srcBlocked && dstDotOp)) {
        return;
      }

      auto mfmaLayout = llvm::dyn_cast<ttg::AMDMfmaEncodingAttr>(
          dstDotOp.getParent());
      auto kWidth = dstDotOp.getKWidth();
      auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

      if (kWidth != 8 || warpsPerCTA[0] != 1) {
        return;
      }

      if (dstDotOp.getOpIdx() != 1) {
        return;
      }

      cvtVec.push_back(cvtOp);
    });

    assert(cvtVec.size() == 1);

    auto convert = cvtVec[0];
    auto loadInst = getLoadInst(convert, m);

    if (!loadInst.has_value()) {
      return;
    }

    auto loadType =
        dyn_cast<RankedTensorType>(loadInst.value().getResult().getType());
    if (!loadType)
      return;

    auto dstType = cast<RankedTensorType>(convert.getType());
    auto dstDotOp =
        dyn_cast<ttg::DotOperandEncodingAttr>(dstType.getEncoding());

    convertLayout(dstDotOp, (Operation *)loadInst.value());
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUBypassLDSForDotLayout() {
  return std::make_unique<TritonAMDGPUBypassLDSForDotLayoutPass>();
}

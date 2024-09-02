#include "AMDGPUToLLVM/AMDGPUToLLVMPass.h"
#include "Dialect/AMDGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::AMD::llLoad;
using ::mlir::triton::gpu::getTotalElemsPerThread;
namespace tta = mlir::triton::amdgpu;

#define GEN_PASS_CLASSES
#include "AMDGPUToLLVM/Passes.h.inc"

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalDialect<triton::TritonDialect>();
    addLegalDialect<triton::gpu::TritonGPUDialect>();
    addLegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addLegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct AMDLoadConversionBase {
  explicit AMDLoadConversionBase(ModuleAxisInfoAnalysis &axisAnalysisPass)
      : axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    llvm::outs() << "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n";
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    // The maximum vector size is 128 bits on NVIDIA GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct AMDLoadOpConversion : public ConvertOpToLLVMPattern<tta::AMDLoadOp>,
                          public AMDLoadConversionBase {
  using ConvertOpToLLVMPattern<tta::AMDLoadOp>::ConvertOpToLLVMPattern;

  AMDLoadOpConversion(LLVMTypeConverter &converter,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<tta::AMDLoadOp>(converter, benefit),
        AMDLoadConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(tta::AMDLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    llvm::outs() << "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n";
    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = 8; //getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is GEP with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);
      auto vecTy = LLVM::getFixedVectorType(valueElemTy, vec);
      Value ptr = addrspacecast(ptr_ty(getContext()), ptrElems[vecStart]);

      mlir::Attribute zeroAttr = rewriter.getZeroAttr(valueElemTy);
      auto denseValue =
          DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
      Value zeroVal = rewriter.create<LLVM::ConstantOp>(loc, vecTy, denseValue);

      Value falseVal = zeroVal;
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0) {
        Value v = undef(vecTy);
        for (size_t s = 0; s < vec; ++s) {
          Value otherElem = otherElems[vecStart + s];
          Value indexVal = createIndexAttrConstant(
              rewriter, loc, this->getTypeConverter()->getIndexType(), s);
          v = insert_element(vecTy, v, otherElem, indexVal);
        }
        falseVal = v;
      }

      bool nt = op.getCache() == triton::CacheModifier::CG;
      ptr.dump();
      auto loadVal = llLoad(rewriter, loc, ptr, vecTy, pred, falseVal, nt);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, this->getTypeConverter()->getIndexType(), ii % vec);
        Value loaded = extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {
void populateAMDGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  PatternBenefit benefit) {
  patterns.add<AMDLoadOpConversion>(typeConverter, axisInfoAnalysis, benefit);
}
} // namespace mlir::triton::AMD

class ConvertAMDGPUToLLVM
    : public ConvertAMDGPUToLLVMBase<ConvertAMDGPUToLLVM> {

public:
  explicit ConvertAMDGPUToLLVM() {}
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);
    llvm::outs() << "ppppppppppppppppppppppppppppppppppppppppppp\n";
    // Include this
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    constexpr int AMDBenefit = 10 + 1;
    AMD::populateAMDGPUToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                      AMDBenefit);

    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createConvertAMDGPUToLLVMPass() {
    llvm::outs() << "ppppppppppppppppppppppppppppppppppppppppppp\n";
  return std::make_unique<::ConvertAMDGPUToLLVM>();
}
} // namespace triton
} // namespace mlir
// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef AMDGPU_OPS
#define AMDGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "AMDGPUDialect.td"
include "AMDGPUAttrDefs.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypeInterfaces.td"

def LLVM_PointerGlobal : LLVM_PointerInAddressSpace<1>;
def LLVM_PointerShared : LLVM_PointerInAddressSpace<3>;

class AMDGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<AMDGPU_Dialect, mnemonic,
       !listconcat(traits, [TensorSizeTrait, VerifyTensorLayoutsTrait])> {
}
def AMDGPU_AMDLoadOp : AMDGPU_Op<"load_amd", [
  SameLoadStoreOperandsAndResultShape,
  SameLoadStoreOperandsAndResultEncoding,
  AttrSizedOperandSegments,
  // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  TypesMatchWith<"result matches ptr type", "ptr", "result", "getPointeeType($_self)">,
  TypesMatchWith<"mask type matches ptr type", "ptr", "mask", "getI1SameShape(getPointeeType($_self))",
                 "($_op.getOperands().size() <= 1) || std::equal_to<>()">,
  TypesMatchWith<"other matches ptr type", "ptr", "other", "getPointeeType($_self)",
                 "($_op.getOperands().size() <= 2) || std::equal_to<>()">
]> {
    let summary = "Load from a tensor of pointers or from a tensor pointer";

    let arguments = (
      ins
      AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,
      Optional<TT_BoolLike>:$mask,
      Optional<TT_Type>:$other,

      DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
      OptionalAttr<TT_PaddingOptionAttr>:$padding,
      DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
      DefaultValuedAttr<TT_EvictionPolicyAttr, "::mlir::triton::EvictionPolicy::NORMAL">:$evict,
      DefaultValuedAttr<BoolAttr, "false">:$isVolatile
    );

    let results = (outs TT_Type:$result);

    let builders = [
        // // A tensor of pointers or a pointer to a scalar
        // OpBuilder<(ins "Value":$ptr, "triton::CacheModifier":$cache,
        //                "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // // A tensor pointer with boundary check and padding
        // OpBuilder<(ins "Value":$ptr, "ArrayRef<int32_t>":$boundaryCheck,
        //                "std::optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
        //                "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // // A tensor of pointers or a pointer to a scalar with mask
        // OpBuilder<(ins "Value":$ptr, "Value":$mask, "triton::CacheModifier":$cache,
        //                "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // // A tensor of pointers or a pointer to a scalar with mask and other
        // OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other, "triton::CacheModifier":$cache,
        //                "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // // A utility function to build the operation with all attributes
        // OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other,
        //                "ArrayRef<int32_t>":$boundaryCheck,
        //                "std::optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
        //                "triton::EvictionPolicy":$evict, "bool":$isVolatile)>
    ];

    // Specify `cacheModifier` and `evictionPolicy` explicitly in the
    // assemblyFormat instead of as part of attr-dict so that they get printed
    // as strings rather than opaque integers.
    //
    // Note there's no comma between `other` and `cacheModifier` and between
    // `cacheModifier` and `evictionPolicy`.  This is due to an apparent
    // limitation in the MLIR custom-format parser.  In oilist, the initial
    // keywords of each clause have to be unique, so they can't be `,`.
    //
    // Even if we gave up on order-independence and used vanilla optional
    // clauses, the format (`,` `foo` `=` $foo^)? (`,` `bar` `=` $bar^)?  will
    // not match the string ", bar = 0" because after the initial comma (first
    // token of the first optional clause) we expect to see "foo".
    let assemblyFormat = [{
      $ptr (`,` $mask^)? (`,` $other^)?
      oilist(
        `cacheModifier` `=` $cache |
        `evictionPolicy` `=` $evict
      )
      attr-dict `:` type($ptr)
    }];

    let hasCanonicalizer = 0;
}


#endif

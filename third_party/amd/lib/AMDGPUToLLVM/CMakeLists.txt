add_triton_library(AMDGPUToLLVM
    AMDGPUToLLVMPass.cpp

    DEPENDS
    AMDGPUConversionPassIncGen
    AMDGPUIR

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    
)

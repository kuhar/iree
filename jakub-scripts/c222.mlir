!matA = tensor<2048x1280xf16>
!matB = tensor<1280x1280xf16>
!matC = tensor<2048x1280xf32>

#tile_sizes = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 64]]>

#schedule = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
      subgroup_m_count = 4, subgroup_n_count = 4,
      subgroup_m_tile_count = 1, subgroup_n_tile_count = 1, subgroup_k_tile_count = 4>

#trans = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #schedule}>

module {
  func.func @main(%arg0: !matA, %arg1: !matB) -> !matC {
    %cst = arith.constant 0.000000e+00 : f16
    %5 = tensor.empty() : !matC
    %6 = linalg.fill ins(%cst : f16) outs(%5 : !matC) -> !matC
    %8 = linalg.matmul_transpose_b {
      compilation_info = #iree_codegen.compilation_info<
         lowering_config = #tile_sizes,
         translation_info = #trans,
         workgroup_size = [128, 2, 1],
         subgroup_size = 64>
    } ins(%arg0, %arg1 : !matA, !matB) outs(%6 : !matC) -> !matC
    return %8 : !matC
  }
}


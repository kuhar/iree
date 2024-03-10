#tile_sizes = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 64]]>

#schedule = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
      subgroup_m_count = 4, subgroup_n_count = 4,
      subgroup_m_tile_count = 1, subgroup_n_tile_count = 1, subgroup_k_tile_count = 4>

#trans = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #schedule}>

#comp_info = #iree_codegen.compilation_info<
                lowering_config = #tile_sizes,
                translation_info = #trans,
                workgroup_size = [256, 4, 1],
                subgroup_size = 64
             >

module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>]>]} {
  flow.executable private @main_dispatch_0 {
    flow.executable.export public @main_dispatch_0_generic_2048x1280x1280_f16xf16xf32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_0_generic_2048x1280x1280_f16xf16xf32(%arg0: !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<2048x1280xf32>>) {
        %cst = arith.constant 0.000000e+00 : f16
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>> -> tensor<1280x1280xf16>
        %2 = tensor.empty() : tensor<2048x1280xf32>
        %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
        %4 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
          }
          ins(%0, %1 : tensor<2048x1280xf16>, tensor<1280x1280xf16>)
          outs(%3 : tensor<2048x1280xf32>) attrs = {
            compilation_info = #comp_info     
          }{
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %5 = arith.extf %in : f16 to f32
          %6 = arith.extf %in_0 : f16 to f32
          %7 = arith.mulf %5, %6 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<2048x1280xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : tensor<2048x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x1280xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @main(%input0: tensor<2048x1280xf16>, %input1: tensor<1280x1280xf16>) -> (%output0: tensor<2048x1280xf32>)"}} {
    %0 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<2048x1280xf16>
    %1 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<1280x1280xf16>
    %2 = flow.dispatch @main_dispatch_0::@main_dispatch_0_generic_2048x1280x1280_f16xf16xf32(%0, %1) : (tensor<2048x1280xf16>, tensor<1280x1280xf16>) -> tensor<2048x1280xf32>
    %3 = hal.tensor.export %2 "output0" : tensor<2048x1280xf32> -> !hal.buffer_view
    util.return %3 : !hal.buffer_view
  }
}
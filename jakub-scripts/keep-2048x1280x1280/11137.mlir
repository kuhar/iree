// Configuration(subgroup_size=64, workgroup_size=[128, 2, 1], intrinsic='#iree_gpu.mfma_layout<F16_16x16x16_F32>', tile_sizes=[64, 64, 32], subgroup_m_count=4, subgroup_n_count=2, subgroup_m_tile_count=1, subgroup_n_tile_count=2, subgroup_k_tile_count=2)
module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>]>]} {
  hal.executable private @main$async_dispatch_222 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mfma_layout<F16_16x16x16_F32>, #iree_gpu.mfma_layout<F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none"}>) {
      hal.executable.export public @main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 4, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>], subgroup_size = 64 : index, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>, subgroup_m_count = 4, subgroup_n_count = 2, subgroup_m_tile_count = 1, subgroup_n_tile_count = 2, subgroup_k_tile_count = 2>}>, workgroup_size = [128 : index, 2 : index, 1 : index]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32() {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = hal.interface.constant.load[3] : i32
          %4 = arith.index_castui %0 : i32 to index
          %5 = arith.index_castui %1 : i32 to index
          %6 = arith.index_castui %2 : i32 to index
          %7 = arith.index_castui %3 : i32 to index
          %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
          %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>>
          %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
          %11 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
          %12 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<2048x1280xf16>>
          %13 = flow.dispatch.tensor.load %8, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
          %14 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>> -> tensor<1280x1280xf16>
          %15 = flow.dispatch.tensor.load %10, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
          %16 = flow.dispatch.tensor.load %11, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
          %17 = tensor.empty() : tensor<2048x1280xf16>
          %18 = tensor.empty() : tensor<2048x1280xf32>
          %19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 32]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>
          %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 32]]>} {
          ^bb0(%in: f16, %in_0: f16, %out: f32):
            %22 = arith.extf %in : f16 to f32
            %23 = arith.extf %in_0 : f16 to f32
            %24 = arith.mulf %22, %23 : f32
            %25 = arith.addf %out, %24 : f32
            linalg.yield %25 : f32
          } -> tensor<2048x1280xf32>
          %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %15, %16 : tensor<2048x1280xf32>, tensor<1280xf32>, tensor<2048x1280xf16>) outs(%17 : tensor<2048x1280xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 32]]>} {
          ^bb0(%in: f32, %in_0: f32, %in_1: f16, %out: f16):
            %22 = arith.addf %in, %in_0 : f32
            %23 = arith.truncf %22 : f32 to f16
            %24 = arith.addf %23, %in_1 : f16
            linalg.yield %24 : f16
          } -> tensor<2048x1280xf16>
          flow.dispatch.tensor.store %21, %12, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : tensor<2048x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<2048x1280xf16>>
          return
        }
      }
    }
  }
  util.global private mutable @main$async_dispatch_222_rocm_hsaco_fb_main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32_buffer : !hal.buffer
  util.initializer {
    %c2062433280 = arith.constant 2062433280 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c2062433280}
    util.global.store %buffer, @main$async_dispatch_222_rocm_hsaco_fb_main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @main$async_dispatch_222_rocm_hsaco_fb_main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c1703118336 = arith.constant 1703118336 : index
    %c3 = arith.constant 3 : index
    %c1340526464 = arith.constant 1340526464 : index
    %c362591744 = arith.constant 362591744 : index
    %c2 = arith.constant 2 : index
    %c3276800 = arith.constant 3276800 : index
    %c359314944 = arith.constant 359314944 : index
    %c1 = arith.constant 1 : index
    %c359314752 = arith.constant 359314752 : index
    %c74217792_i32 = arith.constant 74217792 : i32
    %c1208145984_i32 = arith.constant 1208145984 : i32
    %c95189312_i32 = arith.constant 95189312 : i32
    %c68974912_i32 = arith.constant 68974912 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 4, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) : !hal.pipeline_layout
    hal.command_buffer.push_constants<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout) offset(0) values([%c68974912_i32, %c95189312_i32, %c1208145984_i32, %c74217792_i32]) : i32, i32, i32, i32
    %main$async_dispatch_222_rocm_hsaco_fb_main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32_buffer = util.global.load @main$async_dispatch_222_rocm_hsaco_fb_main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%main$async_dispatch_222_rocm_hsaco_fb_main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32_buffer : !hal.buffer)[%c0, %c359314752], 
      %c1 = (%main$async_dispatch_222_rocm_hsaco_fb_main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32_buffer : !hal.buffer)[%c359314944, %c3276800], 
      %c2 = (%main$async_dispatch_222_rocm_hsaco_fb_main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32_buffer : !hal.buffer)[%c362591744, %c1340526464], 
      %c3 = (%main$async_dispatch_222_rocm_hsaco_fb_main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32_buffer : !hal.buffer)[%c1703118336, %c359314752]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@main$async_dispatch_222::@rocm_hsaco_fb::@main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@main$async_dispatch_222) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@main$async_dispatch_222::@rocm_hsaco_fb::@main$async_dispatch_222_matmul_transpose_b_2048x1280x1280_f16xf16xf32) : index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z])
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device_0 : !hal.device> affinity(%c-1_i64) wait(%1) signal(%fence) commands([%cmd])
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}

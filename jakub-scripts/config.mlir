// Transform dialect specification for attention on MI300 with MFMA.
// This script only supports variants of attention with a sequence
// length that is a multiple of 64. There are two near duplicate
// scripts because when the sequence length is exactly 64, some transformations
// degenerate causing the script to fail due to invalid handles.

#layout = #iree_gpu.mfma_layout<F16_16x16x16_F32>


#tile_sizes1 = #iree_codegen.lowering_config<tile_sizes = [[64, 80, 64]]>
#schedule1 = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
      subgroup_m_count = 2, subgroup_n_count = 1,
      subgroup_m_tile_count = 2, subgroup_n_tile_count = 5, subgroup_k_tile_count = 4>
#trans1 = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #schedule1}>
#comp_info1 = #iree_codegen.compilation_info<
                lowering_config = #tile_sizes1,
                translation_info = #trans1,
                workgroup_size = [64, 2, 1],
                subgroup_size = 64
             >

#tile_sizes2 = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 32]]>
#schedule2 = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
      subgroup_m_count = 2, subgroup_n_count = 2,
      subgroup_m_tile_count = 4, subgroup_n_tile_count = 4, subgroup_k_tile_count = 2>
#trans2 = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #schedule2}>
#comp_info2 = #iree_codegen.compilation_info<
                lowering_config = #tile_sizes2,
                translation_info = #trans2,
                workgroup_size = [128, 2, 1],
                subgroup_size = 64
             >

#schedule3 = #iree_gpu.mma_schedule<
      intrinsic = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
      subgroup_m_count = 2, subgroup_n_count = 2,
      subgroup_m_tile_count = 2, subgroup_n_tile_count = 5, subgroup_k_tile_count = 4>
#trans3 = #iree_codegen.translation_info<LLVMGPUVectorDistribute, {mma_schedule = #schedule3}>
#comp_info3 = #iree_codegen.compilation_info<
                lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 160, 64]]>,
                translation_info = #trans3,
                workgroup_size = [128, 2, 1],
                subgroup_size = 64
             >

module attributes { transform.with_named_sequence } {

  transform.named_sequence @__attention_main(%variant_op: !transform.any_op {transform.consumed}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %tiled_attention, %forall_grid =
    transform.structured.tile_using_forall %attention tile_sizes [1, 128]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Tile batch dimensions of attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %batch_tiled_attn, %loop = transform.structured.tile_using_for %attention2 [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %top_level_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %top_level_func : !transform.any_op

    // Promote query and output operands
    // ==========================================
    //%attention3 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    //%promoted_attention, %alloc_a0, %alloc_a1 = transform.iree.promote_operands %attention3 [0, 3]
    //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile and decompose attention
    // ==========================================
    %attention4 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %acc_fill, %max_fill, %sum_fill, %inner_loop, %final_scaling, %last_truncate, %blocked_attention = transform.iree.tile_attention %attention4 {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %fill_op, %first_matmul, %reduce_max, %partial_softmax, %scale_factor, %update, %reduce_sum, %truncate, %scale_acc, %second_matmul
        = transform.iree.decompose_tiled_attention %blocked_attention {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Promote key and value operands
    // ==========================================
    %promoted_first_matmul, %alloc0 = transform.iree.promote_operands %first_matmul [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %promoted_second_matmul, %alloc1 = transform.iree.promote_operands %second_matmul [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile and fuse attention ops
    // ==========================================
    %tiled_matmul, %forall = transform.structured.tile_using_forall %promoted_second_matmul tile_sizes [32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_reduce_sum, %forall_reduce = transform.structured.tile_using_forall %reduce_sum tile_sizes [32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


    %f0, %loop0 = transform.structured.fuse_into_containing_op %scale_acc into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f1, %loop1 = transform.structured.fuse_into_containing_op %truncate into %loop0 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %loop4 = transform.loop.fuse_sibling %forall_reduce into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f5_1, %loop5_1 = transform.structured.fuse_into_containing_op %update into %loop4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f5, %loop5 = transform.structured.fuse_into_containing_op %scale_factor into %loop5_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f6, %loop6 = transform.structured.fuse_into_containing_op %partial_softmax into %loop5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f7, %loop7 = transform.structured.fuse_into_containing_op %reduce_max into %loop6 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f8, %loop8 = transform.structured.fuse_into_containing_op %promoted_first_matmul into %loop7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f9, %loop9 = transform.structured.fuse_into_containing_op %fill_op into %loop8 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Distribute fills
    // ==========================================
    %fills = transform.merge_handles %acc_fill, %max_fill, %sum_fill : !transform.any_op
    %tiled_fill, %fill_grid = transform.structured.tile_using_forall %fills tile_sizes[32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Distribute last_truncate and fuse final_scaling into it
    // ==========================================
    %tiled_truncate, %loop_truncate = transform.structured.tile_using_forall %last_truncate tile_sizes[32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %final_scaling into %loop_truncate : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> (!transform.any_op)

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
      transform.apply_patterns.iree.fold_arith_ext_into_contraction
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [64, 4, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    transform.apply_patterns to %func_7 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %func_7 : !transform.any_op
    transform.apply_patterns to %func_7 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_7 : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %func_7
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    // Get the vector.contract ops.
    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op_3 :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %layout16x16x16 = transform.param.constant #layout -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract1, %layout16x16x16 : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %layout16x16x16 : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.amdgpu_distribute_vectors %distribute_func : !transform.any_op

    %distribute_func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.gpu_distribute_shared_memory_copy %func_10 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_10 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %func_10 : !transform.any_op

    %func_11 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.amdgpu.optimize_shared_memory_reads_and_writes %func_11 : (!transform.any_op) -> ()

    transform.yield
  }

  transform.named_sequence @__attention_main_len_64(%variant_op: !transform.any_op {transform.consumed}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %tiled_attention, %forall_grid =
    transform.structured.tile_using_forall %attention tile_sizes [1, 128]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Tile batch dimensions of attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %batch_tiled_attn, %loop = transform.structured.tile_using_for %attention2 [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %top_level_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %top_level_func : !transform.any_op

    // Promote query and output operands
    // ==========================================
    //%attention3 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    //%promoted_attention, %alloc_a0, %alloc_a1 = transform.iree.promote_operands %attention3 [0, 3]
    //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile and decompose attention
    // ==========================================
    %attention4 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %acc_fill, %max_fill, %sum_fill, %inner_loop, %final_scaling, %last_truncate, %blocked_attention = transform.iree.tile_attention %attention4 {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %fill_op, %first_matmul, %reduce_max, %partial_softmax, %scale_factor, %update, %reduce_sum, %truncate, %scale_acc, %second_matmul
        = transform.iree.decompose_tiled_attention %blocked_attention {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Promote key and value operands
    // ==========================================
    %promoted_first_matmul, %alloc0 = transform.iree.promote_operands %first_matmul [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %promoted_second_matmul, %alloc1 = transform.iree.promote_operands %second_matmul [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile and fuse attention ops
    // ==========================================
    %tiled_matmul, %forall = transform.structured.tile_using_forall %promoted_second_matmul tile_sizes [32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_reduce_sum, %forall_reduce = transform.structured.tile_using_forall %reduce_sum tile_sizes [32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


    %f0, %loop0 = transform.structured.fuse_into_containing_op %scale_acc into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f1, %loop1 = transform.structured.fuse_into_containing_op %truncate into %loop0 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %loop4 = transform.loop.fuse_sibling %forall_reduce into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f5_1, %loop5_1 = transform.structured.fuse_into_containing_op %update into %loop4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f5, %loop5 = transform.structured.fuse_into_containing_op %scale_factor into %loop5_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f6, %loop6 = transform.structured.fuse_into_containing_op %partial_softmax into %loop5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f7, %loop7 = transform.structured.fuse_into_containing_op %reduce_max into %loop6 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f8, %loop8 = transform.structured.fuse_into_containing_op %promoted_first_matmul into %loop7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f9, %loop9 = transform.structured.fuse_into_containing_op %fill_op into %loop8 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Distribute fills
    // ==========================================
    %fills = transform.merge_handles %max_fill, %sum_fill : !transform.any_op
    %tiled_fill, %fill_grid = transform.structured.tile_using_forall %fills tile_sizes[32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Distribute last_truncate and fuse final_scaling into it
    // ==========================================
    %tiled_truncate, %loop_truncate = transform.structured.tile_using_forall %last_truncate tile_sizes[32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %final_scaling into %loop_truncate : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> (!transform.any_op)

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
      transform.apply_patterns.iree.fold_arith_ext_into_contraction
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [64, 4, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    transform.apply_patterns to %func_7 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %func_7 : !transform.any_op
    transform.apply_patterns to %func_7 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_7 : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %func_7
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    // Get the vector.contract ops.
    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op_3 :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %layout16x16x16 = transform.param.constant #layout -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract1, %layout16x16x16 : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %layout16x16x16 : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.amdgpu_distribute_vectors %distribute_func : !transform.any_op

    %distribute_func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.gpu_distribute_shared_memory_copy %func_10 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_10 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %func_10 : !transform.any_op

    %func_11 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.amdgpu.optimize_shared_memory_reads_and_writes %func_11 : (!transform.any_op) -> ()

    transform.yield
  }

  // Send it down a custom transform dialect pipeline.
  transform.named_sequence @custom_attention_len_64(%attention: !transform.any_op {transform.readonly}) {
    %variant_op = transform.get_parent_op %attention {op_name = "hal.executable.variant"} : (!transform.any_op) -> !transform.any_op
    %exports = transform.structured.match ops{["hal.executable.export"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main_len_64, {"amdgpu-waves-per-eu" = 2}> -> !transform.any_param
    transform.annotate %exports "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_attention_len_64(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x64xf16> : !transform.any_value
    transform.yield %attention : !transform.any_op
  }

  // Send it down a custom transform dialect pipeline.
  transform.named_sequence @custom_attention(%attention: !transform.any_op {transform.readonly}) {
    %variant_op = transform.get_parent_op %attention {op_name = "hal.executable.variant"} : (!transform.any_op) -> !transform.any_op
    %exports = transform.structured.match ops{["hal.executable.export"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main, {"amdgpu-waves-per-eu" = 2}> -> !transform.any_param
    transform.annotate %exports "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_attention(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?xf16> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[2], 64 : !transform.any_value
    transform.yield %attention : !transform.any_op
  }

  transform.named_sequence @match_mmt_f16_f16_f32(%root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    // transform.print %root {name = "Generic"} : !transform.any_op
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %out: tensor<?x?xf32>):
      %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                            affine_map<(d0, d1, d2) -> (d1, d2)>,
                                            affine_map<(d0, d1, d2) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>) outs(%out : tensor<?x?xf32>) {
        ^bb0(%in: f16, %in_0: f16, %acc: f32):
          %8 = arith.extf %in : f16 to f32
          %9 = arith.extf %in_0 : f16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %acc, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %root : !transform.any_op
  }
  
  transform.named_sequence @apply_mmt_config(%matmul: !transform.any_op {transform.readonly}, %config: !transform.any_param {transform.readonly}) {
    transform.annotate %matmul "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.print %matmul {name = "Applied"} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_mmt_2048x1280x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul)
        : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x1280xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<1280x1280xf16> : !transform.any_value
    %config = transform.param.constant #comp_info1 -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_2048x10240x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul)
        : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x1280xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<10240x1280xf16> : !transform.any_value
    %config = transform.param.constant #comp_info2 -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_2048x1280x5120(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul)
        : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x5120xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<1280x5120xf16> : !transform.any_value
    %config = transform.param.constant #comp_info3 -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        // @match_matmul -> @print_matmul,
        @match_mmt_2048x1280x1280 -> @apply_mmt_config,
        @match_mmt_2048x10240x1280 -> @apply_mmt_config,
        @match_mmt_2048x1280x5120 -> @apply_mmt_config,
        @match_attention_len_64 -> @custom_attention_len_64,
        @match_attention -> @custom_attention
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module

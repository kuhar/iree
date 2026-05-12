// RUN: iree-opt --iree-codegen-test-materialize-smt-constraint-assignments='assignments=wg_0=64,wg_1=128,red_2=8,mma_idx=1,sg_m_cnt=2,sg_n_cnt=1,wg_size_x=128,wg_size_y=1,wg_size_z=1,sg_size=64' %s | FileCheck %s
// RUN: not iree-opt --iree-codegen-test-materialize-smt-constraint-assignments='assignments=wg_0=64,wg_1=128,red_2=8,sg_m_cnt=2,sg_n_cnt=1,wg_size_x=128,wg_size_y=1,wg_size_z=1,sg_size=64' %s 2>&1 | FileCheck %s --check-prefix=ERR

module {
  iree_codegen.smt.constraints
      target = <set = 0>,
      pipeline = #iree_gpu.pipeline<VectorDistribute>,
      knobs = {
        workgroup = [#iree_codegen.smt.int_knob<"wg_0">, #iree_codegen.smt.int_knob<"wg_1">, 0],
        reduction = [0, 0, #iree_codegen.smt.int_knob<"red_2">],
        mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>]>,
        subgroup_basis = [[#iree_codegen.smt.int_knob<"sg_m_cnt">, #iree_codegen.smt.int_knob<"sg_n_cnt">, 1], [0, 1, 2]],
        workgroup_size = [#iree_codegen.smt.int_knob<"wg_size_x">, #iree_codegen.smt.int_knob<"wg_size_y">, #iree_codegen.smt.int_knob<"wg_size_z">],
        subgroup_size = #iree_codegen.smt.int_knob<"sg_size">
      }
      dims() {
      }
}

// CHECK: #translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [128, 1, 1] subgroup_size = 64>
// CHECK: #compilation = #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, reduction = [0, 0, 8], subgroup_basis = {{\[}}[2, 1, 1], [0, 1, 2]], workgroup = [64, 128, 0]}>, translation_info = #translation>
// CHECK-LABEL: iree_codegen.smt.constraints
// CHECK:       dims() attributes {test.materialized_compilation_info = #compilation}

// ERR: failed to materialize compilation_info from constraints: missing assignment for knob 'mma_idx'

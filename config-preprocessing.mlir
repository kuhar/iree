module attributes { transform.with_named_sequence } {
//===----------------------------------------------------------------------===//
// Matmul SplitK tuning
//===----------------------------------------------------------------------===//
  transform.named_sequence @match_generic(%entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.generic"] : !transform.any_op
    transform.yield %entry : !transform.any_op
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

  transform.named_sequence @apply_splitk_config(%op: !transform.any_op {transform.readonly},
                                                %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "iree_flow_split_reduction_ratio" = %config : !transform.any_op, !transform.any_param
    transform.print %op {name = "Applied"} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_mmt_2048x1280x5120(%matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x5120xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<1280x5120xf16> : !transform.any_value
    %config = transform.param.constant 4 -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.consumed}) {
    %generic = transform.collect_matching @match_generic in %variant_op
      : (!transform.any_op) -> !transform.any_op

    transform.print %generic {name = "Main"} : !transform.any_op
    // transform.foreach_match in %variant_op
    //   @match_mmt_2048x1280x5120 -> @apply_splitk_config
    //  : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }

  transform.named_sequence @__preprocessing_main(%variant_op: !transform.any_op {transform.readonly}) {
    transform.yield
  }
} ////  module

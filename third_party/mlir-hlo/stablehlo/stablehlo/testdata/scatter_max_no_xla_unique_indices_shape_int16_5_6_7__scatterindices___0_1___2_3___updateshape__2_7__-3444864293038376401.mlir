// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi16>, tensor<2x7xi16>)
    %2 = call @expected() : () -> tensor<5x6x7xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xi16>, tensor<2x2xi32>, tensor<2x7xi16>) -> tensor<5x6x7xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi16>, tensor<2x7xi16>) {
    %0 = stablehlo.constant dense<"0x0000FDFFFBFF030001000000FFFF000006000100F9FF0200000007000300FBFF0000FFFF0400000000000400030001000300FAFF0000FCFF0200FFFF0000FFFF0000FDFF030000000200020004000000FEFF0000FEFF00000100FFFF0200FAFFFFFF0000FDFFFDFFFCFF050000000000020001000500010006000600FFFF00000000020000000300FDFFFFFF0100FDFFFEFF00000000FEFF0000000002000600FFFF0000FFFF040003000100FEFF0100FCFF0000010004000000FDFF01000000FAFFFEFF0000F9FFFEFF00000100FFFF0300FEFF030000000300FEFF010002000000FFFF0300FFFFFBFF00000000FEFF00000400FFFF0600FFFF0200FFFF0300FEFF010000000000F7FFFEFFFDFF0000FFFF000002000400FFFF030000000300FFFFFEFFFDFF0200FEFFFFFF0000FFFF0200FFFFFDFFFFFF0700FFFF00000600FEFFFBFF01000000FCFFFEFF0000FCFF0500FFFF0000FDFF030000000000FEFF000006000200000002000000000002000000FFFF01000200FEFFFEFF01000000FDFF0400FDFF000000000200000000000000FCFF0200FAFF060002000500000000000400"> : tensor<5x6x7xi16>
    %1 = stablehlo.constant dense<[[0, 1, -3, 2, 0, -4, 1], [3, 3, 4, 0, -1, 1, 0]]> : tensor<2x7xi16>
    return %0, %1 : tensor<5x6x7xi16>, tensor<2x7xi16>
  }
  func.func private @expected() -> tensor<5x6x7xi16> {
    %0 = stablehlo.constant dense<"0x0000FDFFFBFF030001000000FFFF00000600010002000200000007000300FBFF0000FFFF0400000000000400030001000300FAFF0000FCFF0200FFFF0000FFFF0000FDFF030000000200020004000000FEFF0000FEFF00000100FFFF0200FAFFFFFF0000FDFFFDFFFCFF050000000000020001000500010006000600FFFF00000000020000000300FDFFFFFF0100FDFFFEFF00000000FEFF0000000002000600FFFF0000FFFF040003000100FEFF0100FCFF0000010004000000FDFF01000000FAFFFEFF0000F9FFFEFF00000100FFFF03000300030004000300FFFF010002000000FFFF0300FFFFFBFF00000000FEFF00000400FFFF0600FFFF0200FFFF0300FEFF010000000000F7FFFEFFFDFF0000FFFF000002000400FFFF030000000300FFFFFEFFFDFF0200FEFFFFFF0000FFFF0200FFFFFDFFFFFF0700FFFF00000600FEFFFBFF01000000FCFFFEFF0000FCFF0500FFFF0000FDFF030000000000FEFF000006000200000002000000000002000000FFFF01000200FEFFFEFF01000000FDFF0400FDFF000000000200000000000000FCFF0200FAFF060002000500000000000400"> : tensor<5x6x7xi16>
    return %0 : tensor<5x6x7xi16>
  }
}


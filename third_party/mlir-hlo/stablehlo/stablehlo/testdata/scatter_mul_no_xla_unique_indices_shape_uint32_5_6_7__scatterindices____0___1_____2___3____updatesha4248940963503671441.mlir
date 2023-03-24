// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui32>, tensor<5x2x2x7xui32>)
    %2 = call @expected() : () -> tensor<5x6x7xui32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui32>
      stablehlo.return %5 : tensor<ui32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui32>, tensor<2x2x1xi32>, tensor<5x2x2x7xui32>) -> tensor<5x6x7xui32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui32>, tensor<5x6x7xui32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui32>, tensor<5x2x2x7xui32>) {
    %0 = stablehlo.constant dense<"0x010000000100000001000000050000000000000000000000000000000200000001000000020000000100000000000000070000000300000001000000030000000300000003000000020000000100000000000000020000000100000001000000010000000400000000000000020000000400000002000000020000000000000003000000000000000000000004000000010000000000000001000000000000000200000002000000030000000100000003000000030000000000000002000000010000000200000000000000050000000200000004000000020000000400000000000000000000000000000000000000050000000000000000000000020000000800000000000000040000000400000000000000020000000000000000000000010000000300000001000000000000000000000003000000040000000200000000000000010000000500000001000000010000000700000001000000000000000000000006000000020000000100000000000000020000000300000002000000010000000000000003000000010000000100000003000000020000000200000000000000030000000200000003000000020000000100000002000000000000000100000000000000030000000100000004000000020000000700000000000000020000000000000007000000010000000100000001000000030000000100000006000000010000000300000004000000000000000000000000000000050000000400000001000000020000000600000000000000010000000100000002000000030000000200000006000000010000000500000002000000010000000000000000000000030000000300000002000000020000000400000000000000000000000600000002000000010000000100000000000000000000000000000001000000040000000300000001000000040000000100000003000000000000000300000000000000000000000500000004000000020000000300000002000000000000000200000000000000010000000100000004000000020000000800000002000000000000000300000005000000050000000300000004000000000000000000000003000000000000000200000003000000020000000400000005000000020000000000000002000000"> : tensor<5x6x7xui32>
    %1 = stablehlo.constant dense<"0x0000000001000000000000000000000002000000000000000000000004000000070000000300000001000000010000000500000003000000010000000200000003000000000000000400000002000000000000000500000002000000000000000600000001000000010000000300000000000000010000000200000000000000010000000000000002000000060000000200000004000000010000000200000007000000000000000100000001000000040000000100000002000000000000000200000001000000040000000000000001000000010000000700000004000000000000000300000005000000000000000500000001000000020000000100000004000000030000000000000001000000040000000400000002000000030000000700000004000000050000000400000004000000010000000100000003000000010000000200000002000000050000000100000002000000040000000100000001000000010000000200000001000000020000000100000006000000020000000300000000000000060000000500000007000000010000000000000004000000040000000100000000000000060000000100000001000000050000000000000001000000000000000200000002000000000000000200000000000000010000000100000001000000030000000100000002000000000000000400000003000000020000000600000000000000020000000400000001000000020000000000000004000000000000000200000000000000"> : tensor<5x2x2x7xui32>
    return %0, %1 : tensor<5x6x7xui32>, tensor<5x2x2x7xui32>
  }
  func.func private @expected() -> tensor<5x6x7xui32> {
    %0 = stablehlo.constant dense<"0x0000000001000000000000000000000000000000000000000000000008000000070000000600000001000000000000002300000009000000010000000600000009000000000000000800000002000000000000000A0000000200000000000000060000000400000000000000060000000400000002000000020000000000000003000000000000000000000004000000010000000000000001000000000000000200000002000000000000000100000006000000000000000000000000000000020000000C000000000000001400000002000000080000000E00000000000000000000000000000000000000000000000A000000000000000000000002000000200000000000000004000000040000000000000008000000000000000000000001000000030000000100000000000000000000000300000004000000020000000000000001000000050000000100000000000000150000000500000000000000000000000600000004000000010000000000000006000000000000000200000004000000000000000600000003000000070000000C0000000A000000080000000000000003000000020000000900000002000000020000000400000000000000010000000000000003000000010000000400000002000000070000000000000002000000000000000700000001000000010000000100000003000000020000001800000001000000030000000400000000000000000000000000000005000000180000000200000006000000000000000000000005000000070000000200000000000000080000001800000001000000000000000C000000010000000000000000000000000000000300000002000000020000000400000000000000000000000600000002000000010000000100000000000000000000000000000001000000040000000000000002000000080000000000000006000000000000000300000000000000000000000F0000000400000004000000000000000800000000000000040000000000000000000000020000001000000002000000100000000000000000000000000000000A000000000000000300000004000000000000000000000003000000000000000200000003000000020000000400000005000000020000000000000002000000"> : tensor<5x6x7xui32>
    return %0 : tensor<5x6x7xui32>
  }
}


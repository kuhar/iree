// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi32>, tensor<5x2x2x7xi32>)
    %2 = call @expected() : () -> tensor<5x6x7xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi32>, tensor<2x2x1xi32>, tensor<5x2x2x7xi32>) -> tensor<5x6x7xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi32>, tensor<5x6x7xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi32>, tensor<5x2x2x7xi32>) {
    %0 = stablehlo.constant dense<"0xFEFFFFFF0200000000000000FFFFFFFFFFFFFFFFFEFFFFFF0200000008000000040000000100000001000000FEFFFFFFFEFFFFFF0100000000000000FFFFFFFF00000000FDFFFFFF05000000FFFFFFFFFFFFFFFF02000000FCFFFFFFFCFFFFFFFEFFFFFF0200000000000000FDFFFFFF01000000FFFFFFFF0000000005000000FFFFFFFF05000000FDFFFFFFFFFFFFFF04000000FFFFFFFF000000000400000003000000FEFFFFFFFBFFFFFFFDFFFFFFFAFFFFFF000000000400000000000000FFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FDFFFFFF000000000500000004000000FEFFFFFF03000000FDFFFFFF0000000005000000FBFFFFFF020000000100000002000000FDFFFFFF0400000002000000FFFFFFFF0000000000000000FCFFFFFF0300000001000000FDFFFFFFFEFFFFFF0000000005000000FBFFFFFF00000000FFFFFFFF0000000002000000FAFFFFFF0100000001000000FCFFFFFF00000000FEFFFFFFFBFFFFFF02000000FBFFFFFFFEFFFFFFFDFFFFFF000000000300000001000000FDFFFFFFFFFFFFFFFFFFFFFF0000000003000000FFFFFFFFFDFFFFFF00000000FFFFFFFF0100000001000000050000000C000000FDFFFFFFFBFFFFFFFFFFFFFFFCFFFFFF03000000FFFFFFFF00000000000000000000000002000000FFFFFFFF01000000FFFFFFFF0300000000000000FDFFFFFF000000000000000000000000FDFFFFFF0000000001000000000000000000000004000000FEFFFFFFFCFFFFFF00000000010000000200000000000000FDFFFFFF0300000005000000FDFFFFFF04000000FEFFFFFF06000000FEFFFFFFFCFFFFFF04000000FAFFFFFF0200000001000000FFFFFFFF00000000020000000000000000000000020000000000000000000000FDFFFFFFFEFFFFFFFFFFFFFFFDFFFFFF0100000000000000F8FFFFFF0300000000000000010000000000000001000000020000000000000003000000FDFFFFFF01000000FDFFFFFF01000000FDFFFFFF00000000FDFFFFFFF9FFFFFF010000000100000000000000FCFFFFFF0000000004000000FFFFFFFF04000000FFFFFFFF010000000100000001000000FEFFFFFF00000000FFFFFFFFFFFFFFFF0100000000000000030000000000000002000000FDFFFFFF0100000005000000"> : tensor<5x6x7xi32>
    %1 = stablehlo.constant dense<"0x030000000300000001000000FDFFFFFF010000000200000001000000FBFFFFFFFDFFFFFF00000000FFFFFFFF02000000F9FFFFFF040000000100000000000000FCFFFFFF000000000300000001000000000000000300000003000000000000000000000000000000FAFFFFFF06000000FFFFFFFFFBFFFFFF000000000100000005000000FFFFFFFF000000000100000000000000FFFFFFFF01000000000000000200000003000000FCFFFFFF00000000000000000000000000000000FCFFFFFF00000000FFFFFFFF06000000FBFFFFFF000000000200000004000000000000000000000001000000FEFFFFFF03000000FBFFFFFFFDFFFFFF0400000000000000FFFFFFFFFFFFFFFF0200000000000000000000000200000003000000FFFFFFFFFEFFFFFF0000000000000000030000000100000003000000FDFFFFFFFFFFFFFF04000000000000000200000001000000FFFFFFFF00000000FFFFFFFF00000000FDFFFFFF030000000100000000000000FFFFFFFF0100000004000000FEFFFFFF01000000FEFFFFFFFFFFFFFFFCFFFFFFFEFFFFFFF9FFFFFF0400000003000000FFFFFFFFFEFFFFFF05000000FFFFFFFF040000000000000003000000030000000100000000000000FFFFFFFFFDFFFFFFFEFFFFFFFFFFFFFFFEFFFFFF01000000FCFFFFFFFFFFFFFFFBFFFFFF02000000FEFFFFFFFBFFFFFF02000000FDFFFFFF020000000100000000000000FEFFFFFF0200000001000000060000000100000001000000000000000300000000000000"> : tensor<5x2x2x7xi32>
    return %0, %1 : tensor<5x6x7xi32>, tensor<5x2x2x7xi32>
  }
  func.func private @expected() -> tensor<5x6x7xi32> {
    %0 = stablehlo.constant dense<"0x010000000500000001000000FCFFFFFF0000000000000000030000000300000001000000010000000000000000000000F7FFFFFF0500000001000000FFFFFFFFFCFFFFFFFDFFFFFF0800000000000000FFFFFFFF05000000FFFFFFFFFCFFFFFFFEFFFFFF02000000FAFFFFFF0300000001000000FFFFFFFF0000000005000000FFFFFFFF05000000FDFFFFFFFFFFFFFF04000000FFFFFFFF000000000400000003000000FEFFFFFFFAFFFFFFF8FFFFFFFAFFFFFF0100000009000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF01000000FDFFFFFF020000000800000000000000FEFFFFFF03000000FDFFFFFF0000000001000000FBFFFFFF0100000007000000FDFFFFFFFDFFFFFF0600000006000000FFFFFFFF0000000000000000FCFFFFFF0300000001000000FDFFFFFFFEFFFFFF0000000005000000FBFFFFFF00000000FFFFFFFF0000000002000000FAFFFFFF02000000FFFFFFFFFFFFFFFFFBFFFFFFFBFFFFFFFFFFFFFF02000000FAFFFFFFFDFFFFFFFFFFFFFF00000000030000000300000000000000FEFFFFFFFDFFFFFF000000000300000002000000FEFFFFFF03000000FCFFFFFF0000000005000000050000000E000000FEFFFFFFFBFFFFFFFFFFFFFFFCFFFFFF03000000FFFFFFFF00000000000000000000000002000000FFFFFFFF01000000FFFFFFFF0300000000000000FCFFFFFF00000000FFFFFFFF00000000FAFFFFFF030000000200000000000000FFFFFFFF0500000002000000FAFFFFFF01000000FFFFFFFF01000000FCFFFFFFFBFFFFFFFCFFFFFF090000000000000003000000FCFFFFFF0B000000FDFFFFFF0000000004000000FDFFFFFF0500000001000000FFFFFFFF00000000020000000000000000000000020000000000000000000000FDFFFFFFFEFFFFFFFFFFFFFFFDFFFFFF0100000001000000F8FFFFFF02000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF03000000FCFFFFFF02000000F8FFFFFF03000000FBFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFAFFFFFF01000000FFFFFFFF02000000FDFFFFFF0600000005000000000000000400000002000000010000000100000001000000FEFFFFFF00000000FFFFFFFFFFFFFFFF0100000000000000030000000000000002000000FDFFFFFF0100000005000000"> : tensor<5x6x7xi32>
    return %0 : tensor<5x6x7xi32>
  }
}


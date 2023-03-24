// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xi16>, tensor<3x5x2xi16>)
    %2 = call @expected() : () -> tensor<3x5x40xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>} : (tensor<3x5x40xi16>, tensor<2x1xi32>, tensor<3x5x2xi16>) -> tensor<3x5x40xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xi16>, tensor<3x5x40xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xi16>, tensor<3x5x2xi16>) {
    %0 = stablehlo.constant dense<"0xFFFF0300FFFFFBFF03000000FEFF02000200FCFFFFFF0000FCFF0100FFFFFEFF0000FFFF00000000FFFFFDFF0000FDFF000002000000FDFF0000FFFFFFFFFFFF000004000000FEFFFBFF0100FEFF020001000000FDFF00000300FEFF060002000000FEFFFDFFFEFFFEFFFDFFFFFFF9FFFCFF00000000000000000100FAFF00000500010001000500000001000400FEFF0400FEFF010002000100030001000200FEFFFCFFFFFF0100FCFF0100FFFFFEFF07000400FFFF0200FFFFFFFF00000400FFFF0300FDFF00000000FDFFFDFF000004000000FFFF030003000200FCFF020003000200FEFFFEFF010002000500FEFF040004000300FFFF0200050000000200FAFF040002000000FFFF0000FDFF0000FEFFFFFFFEFFFFFFFDFFFFFF0200FFFFFFFFFFFFFFFF0000FFFFFDFF00000200FDFFFFFF00000000FFFF0100FDFF0200000000000300FEFF010002000100FFFF00000600FFFFFFFF01000000020000000000FFFFFCFFFDFF010003000000FDFF0200FCFF020000000000FEFF0300FEFF040000000000FEFF020000000000FEFFFFFFFFFF01000400FDFF0000FDFFFDFFFEFF00000100000000000400FEFF0000FCFFFEFF0000FAFF060000000200FEFF00000300000006000400000000000400000000000000010001000000FDFF01000000FFFF02000000FFFF00000100FFFFF8FFFEFFFFFFFFFFFEFF0000000000000600FFFFFFFF0200030000000000FFFF0000FAFFFDFFFFFFFFFF0000FEFFFEFF01000000FDFF0000FFFFFCFF000001000300FFFFFCFF01000000FEFF00000300FEFF00000000FBFF0000FEFF02000200000003000000060001000100FFFF0400FBFF0000FEFFFFFF0400030003000000FBFFFCFF0000FEFF0100FFFFFFFFFFFF00000200010001000100030003000300FFFF00000000FBFF0100FEFF03000100000001000000FEFF00000300010001000400FEFFFDFFFDFFFDFFFFFF00000300FCFF03000100000000000000010000000200FDFFFDFF05000000FFFFFEFF020005000000FDFFFDFF0500FCFFFFFFFEFF040000000100FEFF0000020000000000FAFFFBFF0100FCFF00000400FFFF02000000010000000200FEFFFDFF0000020000000200FEFFFFFF0200FAFFF9FF03000300FAFF0100FDFF010002000300FFFFFFFF00000300FEFF03000000FEFF0200FCFF00000000FDFF0100FBFF0000010002000300FCFF02000100020001000400FFFF0300FFFF0000FFFFFCFFFFFF000001000000090000000000010000000200FDFF000000000000FAFFFBFF00000100FFFFFAFF00000300FFFF000002000300FEFF00000000FDFF0400FDFF0200010004000000FFFF000000000000FEFFFBFF0100FEFFFEFFFFFF0000010002000000000003000000FFFF00000200FFFFFEFFFCFFFDFF000001000000FFFF03000700000002000200FFFF0000000001000100FEFFFEFF0000FFFFF7FFFAFF0200FFFF05000100080000000300FBFF01000300FAFFFBFF0300000000000000FEFFF8FF0500010004000100FDFF04000100000002000200000000000200040000000000FEFF00000200000000000200FFFF020006000000FFFF0200FFFF0000FFFFFEFFFCFFFDFF0100FFFF0300FDFFFEFF0300FFFF0400000002000200FEFF000002000600000000000200FBFF0000FDFFFAFF"> : tensor<3x5x40xi16>
    %1 = stablehlo.constant dense<[[[3, -2], [6, 1], [3, 0], [0, -3], [2, 0]], [[-1, 4], [1, 0], [0, -2], [0, -2], [-4, 0]], [[1, 0], [-1, 0], [-1, -1], [-7, 0], [0, -6]]]> : tensor<3x5x2xi16>
    return %0, %1 : tensor<3x5x40xi16>, tensor<3x5x2xi16>
  }
  func.func private @expected() -> tensor<3x5x40xi16> {
    %0 = stablehlo.constant dense<"0xFFFFEEFFFFFFFBFF03000000FEFF02000200FCFFFFFF0000FCFF0100FFFFFEFF0000FFFF00000000FFFFFDFF0000FDFF000002000000FDFF0000FFFFFFFFFFFF000004000000FEFFFBFF0100FEFF020001000000FDFF00000300FEFF060002000000FEFFFDFFFEFFFEFFFDFFFFFFF9FFFCFF00000000000000000100FAFF00000500010001000500000001000400FEFF0400FEFF010002000100030001000200FEFF0000FFFF0100FCFF0100FFFFFEFF07000400FFFF0200FFFFFFFF00000400FFFF0300FDFF00000000FDFFFDFF000004000000FFFF030003000200FCFF020003000200FEFFFEFF010002000500FEFF040000000300FFFF0200050000000200FAFF040002000000FFFF0000FDFF0000FEFFFFFFFEFFFFFFFDFFFFFF0200FFFFFFFFFFFFFFFF0000FFFFFDFF00000200FDFFFFFF00000000FFFF0100FDFF0200000000000300FEFF010002000100FFFF00000600FFFFFFFF01000000020000000000FFFFFCFFFDFF010003000000FDFF0200FCFF020000000000FEFF0300FEFF040000000000FEFF020000000000FEFFFFFF040001000400FDFF0000FDFFFDFFFEFF00000100000000000400FEFF0000FCFFFEFF0000FAFF060000000200FEFF00000300000006000400000000000400000000000000010001000000FDFF01000000000002000000FFFF00000100FFFFF8FFFEFFFFFFFFFFFEFF0000000000000600FFFFFFFF0200030000000000FFFF0000FAFFFDFFFFFFFFFF0000FEFFFEFF01000000FDFF0000FFFFFCFF0000010003000000FCFF01000000FEFF00000300FEFF00000000FBFF0000FEFF02000200000003000000060001000100FFFF0400FBFF0000FEFFFFFF0400030003000000FBFFFCFF0000FEFF0100FFFFFFFFFFFF00000000010001000100030003000300FFFF00000000FBFF0100FEFF03000100000001000000FEFF00000300010001000400FEFFFDFFFDFFFDFFFFFF00000300FCFF030001000000000000000100000002000000FDFF05000000FFFFFEFF020005000000FDFFFDFF0500FCFFFFFFFEFF040000000100FEFF0000020000000000FAFFFBFF0100FCFF00000400FFFF02000000010000000200FEFFFDFF0000020000000000FEFFFFFF0200FAFFF9FF03000300FAFF0100FDFF010002000300FFFFFFFF00000300FEFF03000000FEFF0200FCFF00000000FDFF0100FBFF0000010002000300FCFF02000100020001000400FFFF0000FFFF0000FFFFFCFFFFFF000001000000090000000000010000000200FDFF000000000000FAFFFBFF00000100FFFFFAFF00000300FFFF000002000300FEFF00000000FDFF0400FDFF0200010004000000FFFF000000000000FEFFFBFF0100FEFFFEFFFFFF0000010002000000000003000000FFFF00000200FFFFFEFFFCFFFDFF000001000000FFFF03000700000002000200FFFF0000000001000100FEFF00000000FFFFF7FFFAFF0200FFFF05000100080000000300FBFF01000300FAFFFBFF0300000000000000FEFFF8FF0500010004000100FDFF04000100000002000200000000000200040000000000FEFF00000200000000000200FFFF020006000000FFFF0200FFFF0000FFFFFEFFFCFFFDFF0100FFFF0300FDFFFEFF0300FFFF0400000002000200FEFF000002000600000000000200FBFF0000FDFFFAFF"> : tensor<3x5x40xi16>
    return %0 : tensor<3x5x40xi16>
  }
}


// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x30xi16>
    %1 = call @expected() : () -> tensor<20x30xi16>
    %2 = stablehlo.multiply %0, %0 : tensor<20x30xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x30xi16>, tensor<20x30xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x30xi16> {
    %0 = stablehlo.constant dense<"0x0500FEFF0400FFFF0300FFFF030000000200040002000000FEFFFFFFFEFFFCFF0200000003000500FFFFFEFF020001000000000001000200030008000000FEFFFFFF0400FDFF00000000FFFFFAFF01000300FEFF03000000FDFFFFFF00000000FAFF0000030001000000FDFFFFFFFEFF0100FFFF0100FEFFFEFF03000000FDFF0000000000000100FDFF02000000FEFF0100020000000400030001000000FEFF070004000000FEFF0300FFFF01000200FAFF0000000000000100FFFF00000200FBFF00000300FDFFFCFF000002000000FFFF00000000FFFF0100000002000200FDFF00000100FDFF0000FDFFFBFF0000FDFF0000000000000000FCFF0100FFFF000003000300FFFF00000000FDFF0200FBFF000001000000020002000000FDFF0000FEFF0400FCFF05000300FFFF000005000300FEFFFFFF020000000500FBFFFCFFFEFF0700000000000100FEFF0100FEFF0000000005000400FDFFFCFF07000000FFFF0100FDFF0000FEFF0000FDFFFBFF0100FDFF02000100FEFF050001000400000000000000000000000100FDFFFBFFFFFF02000100FEFF02000200FCFF000000000000FCFFFAFF02000500FFFF0100FEFF0200FEFF0000FDFFF9FFFEFFFCFFFFFFFEFF0200000000000000FCFFFEFF000006000000FFFFFFFF0000FDFF00000000FDFF0300FFFFFFFFFCFF00000100030002000100000007000400FFFFFEFF0100FFFF00000100FAFF020000000000FFFF0000FDFFFBFF0300000000000200FEFF0200000000000300FCFF0000000000000000020004000000FDFF000002000600FFFF0000FDFF00000300FFFF0500FFFFFAFFFFFFFAFFFFFF020000000000FBFFFFFFFEFFFFFF00000100FEFF000001000000FBFFFFFF00000000FDFF00000000020000000300FFFF000001000400010001000000050001000000000008000000010004000100FFFFFEFF05000100FFFFFDFF05000500FEFF0200FEFF0000FDFF04000100FEFF0100FDFFFFFFF9FFFFFFFDFF00000100FEFF0000020003000000FEFFFFFF010002000100000000000000FDFF02000200F8FF0700FEFFFEFFFDFF03000100FEFFFFFFFFFF00000500FEFF0200FFFF0000FFFFFFFFFCFFFBFF0100000003000200010001000000FFFFF8FF0000FCFF00000000FEFFFEFFFFFF0000FDFFFDFFFCFF00000300FEFF03000400FFFFFEFF0000FFFFFEFFFCFFFDFF05000400FEFFF9FF0200000000000200FEFF08000400FFFF0300FDFF000002000100FEFF00000000FFFF0000FEFF0000FEFF0100FDFFFFFF050002000200010000000200FEFFFFFF030000000000FDFFFFFF0000F8FF0100000000000200020000000100FFFF00000000000000000000FCFF03000300FEFF0500000001000100FFFF0200FCFFFCFFFBFFFCFF01000000FEFF0300FFFFFEFFFDFFFEFFFFFFFCFF0600FEFFFDFFFEFF000003000200FFFF00000400FDFF0100FEFF0100020001000300FDFF0100FFFF0400FDFF0000FDFFFEFF0100FFFF0000020002000100FFFF040001000000FFFF000001000300040000000000010000000300FEFFFEFF0100FEFF01000100FDFFFCFFFFFF0100FEFF020000000000FFFFFBFF020000000200020002000200000000000000FDFF03000000000000000800000002000000000000000300FEFFFFFF0400FDFFFDFF"> : tensor<20x30xi16>
    return %0 : tensor<20x30xi16>
  }
  func.func private @expected() -> tensor<20x30xi16> {
    %0 = stablehlo.constant dense<"0x190004001000010009000100090000000400100004000000040001000400100004000000090019000100040004000100000000000100040009004000000004000100100009000000000001002400010009000400090000000900010000000000240000000900010000000900010004000100010001000400040009000000090000000000000001000900040000000400010004000000100009000100000004003100100000000400090001000100040024000000000000000100010000000400190000000900090010000000040000000100000000000100010000000400040009000000010009000000090019000000090000000000000000001000010001000000090009000100000000000900040019000000010000000400040000000900000004001000100019000900010000001900090004000100040000001900190010000400310000000000010004000100040000000000190010000900100031000000010001000900000004000000090019000100090004000100040019000100100000000000000000000000010009001900010004000100040004000400100000000000000010002400040019000100010004000400040000000900310004001000010004000400000000000000100004000000240000000100010000000900000000000900090001000100100000000100090004000100000031001000010004000100010000000100240004000000000001000000090019000900000000000400040004000000000009001000000000000000000004001000000009000000040024000100000009000000090001001900010024000100240001000400000000001900010004000100000001000400000001000000190001000000000009000000000004000000090001000000010010000100010000001900010000000000400000000100100001000100040019000100010009001900190004000400040000000900100001000400010009000100310001000900000001000400000004000900000004000100010004000100000000000000090004000400400031000400040009000900010004000100010000001900040004000100000001000100100019000100000009000400010001000000010040000000100000000000040004000100000009000900100000000900040009001000010004000000010004001000090019001000040031000400000000000400040040001000010009000900000004000100040000000000010000000400000004000100090001001900040004000100000004000400010009000000000009000100000040000100000000000400040000000100010000000000000000000000100009000900040019000000010001000100040010001000190010000100000004000900010004000900040001001000240004000900040000000900040001000000100009000100040001000400010009000900010001001000090000000900040001000100000004000400010001001000010000000100000001000900100000000000010000000900040004000100040001000100090010000100010004000400000000000100190004000000040004000400040000000000000009000900000000000000400000000400000000000000090004000100100009000900"> : tensor<20x30xi16>
    return %0 : tensor<20x30xi16>
  }
}

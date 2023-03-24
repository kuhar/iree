// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xi8>, tensor<3x5x2xi8>)
    %2 = call @expected() : () -> tensor<3x5x40xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>} : (tensor<3x5x40xi8>, tensor<2x1xi32>, tensor<3x5x2xi8>) -> tensor<3x5x40xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xi8>, tensor<3x5x40xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xi8>, tensor<3x5x2xi8>) {
    %0 = stablehlo.constant dense<"0xFF0000FA0001FF00010202FFFF03FE01FF0102FC00020002040003010100FE0105040100F902FF03FCFF000200FAFBFFFF010003FF00040100FF0200FB0000FCF9FBFF0500FFFC02FFFE03010400FC04050007FFFFFF0000FD060003FE00FE0100FE00FDFD03FE0500FD00FD00FFFFF9FE00FCFF00FE00FCFE000103FFFEFE01040000000100050002000000010200FDFFF800FE00FB0002FE0200FFFD04000002FCFFFE0102FDFB00FA010100FB03FE020101FAFFFB03FF00FDFDFCFD0000080505FC03000000FB0105FEFF000100FF0200FD0202FC0101FE01FE01FEFF010401FFFF000401FDFFFEFFFF030303FF01000004FD00FF050100FF0200FFFE05FFFC00FFFBFE02000205FD04000102FF00FD00FD03FD0102FF020200FEFE0002FFFE03FB06FF0000FBFF0000FF00FF04FF0302FCF8FF01FC02FBFB020107FF0300F9F905FE05FDFC0002020202070100FC0006FFFFFF0004FFFDFF0200050003000102FD0003FE000304FF0201FF000002FF020002000500FCFE040202F9000000020200000500FF0402FE02FA0500FEFF04FE03FEFDFF00FD03FE0202FAFC01FC0000FE0102FE00FF01040002050000000001FB02FDFC0A00010001FE000102FE02FB00FC0004030001000001FB00000100F902FB01000001FBFEFFFF00FC0000FDFC05000606FC08FF0100FF0002FFFF02FF00010002000003FF050100FE0400FEFF02010301FD02FEFD000302020306FD0200FEFE0102FD000101FE0302FDFC00FF00FD00000400FD03FF00FD00FE0301000700FFFF03FFFF05FEFC01FE000102FB0105FFFC03020001020102FEFF00030605FFFC01FE04"> : tensor<3x5x40xi8>
    %1 = stablehlo.constant dense<[[[4, 0], [-4, -3], [-1, -4], [0, 2], [-1, 1]], [[3, 0], [0, -6], [-5, -4], [0, -1], [1, 0]], [[0, -7], [0, 3], [-4, 0], [0, 1], [-4, 1]]]> : tensor<3x5x2xi8>
    return %0, %1 : tensor<3x5x40xi8>, tensor<3x5x2xi8>
  }
  func.func private @expected() -> tensor<3x5x40xi8> {
    %0 = stablehlo.constant dense<"0xFF0000FA0001FF00010202FFFF03FE01FF0102FC00020002040003010100FE0105040100F902FF03FCF4000200FAFBFFFF010003FF00040100FF0200FB0000FCF9FBFF0500FFFC02FFFE03010400FC04050007FFFFFF0000FD060003FE00FE0100FE00FDFD03FE0500FD00FD00FFFFF9FE00FCFF00FE00FCFE000103FFFEFE01040000000100050002000000010200FDFFF800FE00FB0002FE0200FFFD0400000204FFFE0102FDFB00FA010100FB03FE020101FAFFFB03FF00FDFDFCFD0000080505FC03000000FB0100FEFF000100FF0200FD0202FC0101FE01FE01FEFF010401FFFF000401FDFFFEFFFF030303FF01000004FD00FF050100FF0200FFFE05FFFC00FFFBFE02000205FD04000102FF00FD00FD03FD0102FF022800FEFE0002FFFE03FB06FF0000FBFF0000FF00FF04FF0302FCF8FF01FC02FBFB020107FF0300F90005FE05FDFC0002020202070100FC0006FFFFFF0004FFFDFF0200050003000102FD0003FE000304000201FF000002FF020002000500FCFE040202F9000000020200000500FF0402FE02FA0500FEFF040003FEFDFF00FD03FE0202FAFC01FC0000FE0102FE00FF01040002050000000001FB02FDFC0A00010001FE000102FE02FB00FC0004030001000001FB00000100F902FB01000001FBFEFFFF00FC0000FD0005000606FC08FF0100FF0002FFFF02FF00010002000003FF050100FE0400FEFF02010301FD02FE00000302020306FD0200FEFE0102FD000101FE0302FDFC00FF00FD00000400FD03FF00FD00FE0301000700FFFF03FFFF05FEFC01FE000102FB0105FFFC03020001020102FEFF00030605FFFC01FE04"> : tensor<3x5x40xi8>
    return %0 : tensor<3x5x40xi8>
  }
}


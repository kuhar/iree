// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xui8>, tensor<3x5x2xui8>)
    %2 = call @expected() : () -> tensor<3x5x40xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>} : (tensor<3x5x40xui8>, tensor<2x1xi32>, tensor<3x5x2xui8>) -> tensor<3x5x40xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xui8>, tensor<3x5x40xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xui8>, tensor<3x5x2xui8>) {
    %0 = stablehlo.constant dense<"0x000104030201030301000204000101040300030001040002020300030003010104040402000000010100010001000201020101050003000707010003010002020201000301000005020202020304020303000103010303030203000002050502010102000202000104030105020100060201000102000003040001010103040102020000000301030003000102010400000203000003020601030703000001000300000203010100000001020001010704020102000005000004030402020301010201060102050207000704010100010201050002020305000106020101010302010201050000010502010506020306020204000400000100010202010204030202030004000501040501030202030001010100000202020003030003010302000004000002030102020002010403040100030101040002020000000001030302010204000003020602000300000204010300010105040106010303000205010201000203010107010300020100020503000000000603000201040401020301010302010402030101060603010103000303020404000501010403040101000000000502020100010201050102010400040103010001030401030301050601050202000000020002050103040200010501000103000303010703010002000001010401030508000103020502000100010206050202010105030205000204010203030002010106030000050104020005000100020402000000020104020301010600020200010100000303020200010504030400020100050001000301000103000002000302010700010206010000020201020405000306"> : tensor<3x5x40xui8>
    %1 = stablehlo.constant dense<[[[0, 2], [0, 3], [0, 2], [4, 0], [2, 0]], [[4, 1], [3, 4], [1, 2], [2, 1], [3, 1]], [[4, 2], [3, 2], [0, 8], [2, 4], [1, 2]]]> : tensor<3x5x2xui8>
    return %0, %1 : tensor<3x5x40xui8>, tensor<3x5x2xui8>
  }
  func.func private @expected() -> tensor<3x5x40xui8> {
    %0 = stablehlo.constant dense<"0x000304030201030301000204000101040300030001040002020300030003010104040402000000010103010001000201020101050003000707010003010002020201000301000005020202020304020303020103010303030203000002050502010102000202000104030105020100060201000102000003040401010103040102020000000301030003000102010400000203000003020601030703000001000302000203010100000001020001010704020102000005000004030402020301010201060102050207050704010100010201050002020305000106020101010302010201050000010502010506020306020904000400000100010202010204030202030004000501040501030202030001010100000202020006030003010302000004000002030102020002010403040100030101040002020000000001030302040204000003020602000300000204010300010105040106010303000205010201000203010107010700020100020503000000000603000201040401020301010302010402030101060603010103000309020404000501010403040101000000000502020100010201050102010400040103010001030401080301050601050202000000020002050103040200010501000103000303010703010002000001010C01030508000103020502000100010206050202010105030205000204010203030002010106030006050104020005000100020402000000020104020301010600020200010100000303020200010504060400020100050001000301000103000002000302010700010206010000020201020405000306"> : tensor<3x5x40xui8>
    return %0 : tensor<3x5x40xui8>
  }
}


// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xui8>, tensor<20x20xui8>)
    %1 = call @expected() : () -> tensor<20x20xui8>
    %2 = stablehlo.subtract %0#0, %0#1 : tensor<20x20xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xui8>, tensor<20x20xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xui8>, tensor<20x20xui8>) {
    %0 = stablehlo.constant dense<"0x01030102000000020206010501030201000100030200010503040001030103000207030103010000010302030101060401020001000004060300020103010202000600020106000303030100040004030003030401050302050103060404030201010200020002020504020005010101030301020101010200000301010005010100010001010202030302020004020606050201020004030000030103010404000002010001020001000003010002000002020000010203020003060202000605060001020002010307010002050402010002020101030104010204030000030102000003000503050007020201030403020601000204010001010303010100030400020204010202010200030201000300020404050001010600020301000000010203000001050200020302030201010001030001070001030500030301040200000102040204030102050201000400020007020002010300010300020602020004050301030206000302000103020202010200030001010002020104060000020002010001010200010104020103"> : tensor<20x20xui8>
    %1 = stablehlo.constant dense<"0x08020100020002020002010300000700010103040002030400040100000507000200010401000102030302020302050003050202000200030100020003010300020002010000000103050201020102030202030201020103000000030303020302040400010300000902000200040201010500000400020500040203010001000002000603030200010001040300010003040100010004030302020002020100020801000000000301020600030000020102030100000500010001010403000001010503000100010102010000000302010101000101010101010101010005020100020204030001020004020103000002040100040002000202010102030302060002000102000300020401030100010200020202000304000203010004000006050101040102020105010001020000020302030003020202040200020301050004000003010300030302020400040301040101000703010003000300010000020101000202010100050104050301010001020202020201020304030403000002060004010102000402030003000001"> : tensor<20x20xui8>
    return %0, %1 : tensor<20x20xui8>, tensor<20x20xui8>
  }
  func.func private @expected() -> tensor<20x20xui8> {
    %0 = stablehlo.constant dense<"0xF9010002FE00FE00020400020103FB01FF00FDFF02FEFE010300FF0103FCFC00000702FD0201FFFEFE000001FEFF0104FEFDFEFF00FE0403020000010000FF02FE06FE010106000200FEFFFF02FF0200FE010002000302FF05010303010101FFFFFDFE0001FD0202FC0202FE05FDFF0002FE0102FD01FFFD00FC01FE0000040101FE01FAFEFE0002020301FEFD0401060301010101000000FDFE010101FF0304FEF80101000102FD00FEFA03FE0002FEFF00FFFF0001FD0301000205FEFF00060405FBFE02FF0200020500000205010000FF010200000200030001030200FB010002FEFEFFFD05020300030001FE030401FE0501FC020201FEFF000201FEFEFEFD04FE02010201FF02FFFEFF000101FF010000020205FDFD0104FD0103FD0000FAFC0102FCFFFF0301FB010301010201FFFDFF0000FE05FEFFFF0300010000FF02FC0001FF03FF0400FE0003FE01FC01FFFEFF0602F9FF0003FD01000001060200FF030501FF020106FB02FEFBFE02010201FF00FE01FE00FFFDFEFFFD010600FEFC00FE00FFFF01FEFEFE0101020102"> : tensor<20x20xui8>
    return %0 : tensor<20x20xui8>
  }
}

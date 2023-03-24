// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xf32>, tensor<4x3xf32>)
    %2 = call @expected() : () -> tensor<4x2x3x5xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf32>, tensor<2xi32>, tensor<4x3xf32>) -> tensor<4x2x3x5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf32>, tensor<4x2x3x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf32>, tensor<4x3xf32>) {
    %0 = stablehlo.constant dense<"0x3BAD2340E5CF43BED787C240ED72973EDF1578C089177AC010AE89BF9E93593F1A9AD44066CB14C06080A23E1B704C3F639B83C0DE187FBFCC6A4F40BB47BE406559613FBFD52DBFF5EF9DBFBD77EABFFB856A3C14B079BF20ED44C0C8EE933E6C9EECBEB96C11C0F409A53FA715A1BF9FD55340F8F7804052FB6AC0E6CA6BC06843D63F97CDD5BFBF680E4023392A409C60863FBD0597C0D476D3BEFD6D70C0A9D5623F2A082E3F8CACA9C0DBF44440604128C0B6025EC002220A4098E07B3D38F615C001424B40D886A53FDCC69B40940883BFEB7210C0ACDCC9BFAAB431BE0AA3C6BF23AD73403212EE3EABA700C0809330C053542E4086F62BC0E6C4044040C8114082C8B7BFC83368C03A853F4017D608C08CC310C0CDA4684048AE57400284C43F1D444EBFBC6E3D40E957853F0CDC404057B1BBBFF5D41FC06F3952C0CEAD24C06E2B98BF0D6D0F40B56DAA3F7BFC1740B91C493F2ECE353FB24E1FBE0686F63F5414C4BF94BC30BE52B5E0BFBBCB2FC0DB9086C066A21AC05F184FC03BCB4F4087A43DC035DE6ABFCB07A03F3A5E9D3F3F70FDBFA378E9BF15954BC0D3A31CC043C9FCBFD88AA23F13C304C09747E03F1CD896BF2EA21A402DAEC73FDA0DFBBF1B21EEBE2481CC3F30C4C23E2957BB3EA05D1DC00B3B14BF6D5802BC"> : tensor<4x2x3x5xf32>
    %1 = stablehlo.constant dense<[[5.165450e+00, 2.35202336, -1.06954634], [-0.0223388262, -3.94093156, -4.79030514], [-2.50940609, 0.152402326, 3.52024436], [6.06733084, -4.54139471, 5.74313974]]> : tensor<4x3xf32>
    return %0, %1 : tensor<4x2x3x5xf32>, tensor<4x3xf32>
  }
  func.func private @expected() -> tensor<4x2x3x5xf32> {
    %0 = stablehlo.constant dense<"0x3BAD2340E5CF43BED787C240ED72973EBA01A53F89177AC010AE89BF9E93593F1A9AD4408013DE3C6080A23E1B704C3F639B83C0DE187FBF5AF70A40BB47BE406559613FBFD52DBFF5EF9DBFBD77EABFFB856A3C14B079BF20ED44C0C8EE933E6C9EECBEB96C11C0F409A53FA715A1BF9FD55340F8F7804052FB6AC0E6CA6BC06843D63F97CDD5BFBFFA0C4023392A409C60863FBD0597C0D476D3BE1B53F6C0A9D5623F2A082E3F8CACA9C0DBF44440DE6AEDC0B6025EC002220A4098E07B3D38F615C001424B40D886A53FDCC69B40940883BFEB7210C0ACDCC9BFAAB431BE0AA3C6BF23AD73403212EE3EABA700C0809330C053542E4086F62BC0E6C40440C01D6DBE82C8B7BFC83368C03A853F4017D608C0960207C0CDA4684048AE57400284C43F1D444EBF365DCF40E957853F0CDC404057B1BBBFF5D41FC06F3952C0CEAD24C06E2B98BF0D6D0F40B56DAA3F7BFC1740B91C493F2ECE353FB24E1FBE0686F63F5414C4BF94BC30BE52B5E0BFBBCB2FC0DB9086C0C0AC69405F184FC03BCB4F4087A43DC035DE6ABF50A252C03A5E9D3F3F70FDBFA378E9BF15954BC0C7EB524043C9FCBFD88AA23F13C304C09747E03F1CD896BF2EA21A402DAEC73FDA0DFBBF1B21EEBE2481CC3F30C4C23E2957BB3EA05D1DC00B3B14BF6D5802BC"> : tensor<4x2x3x5xf32>
    return %0 : tensor<4x2x3x5xf32>
  }
}


// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>)
    %2 = call @expected() : () -> tensor<5x6x7xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      stablehlo.return %arg1 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xbf16>, tensor<2x2xi32>, tensor<2x7xbf16>) -> tensor<5x6x7xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>) {
    %0 = stablehlo.constant dense<"0xEF3E0A3B28C0B9C080C0AC3E2D409740A53E8CBC01BF453F663F2CBF543E8740AC3E18C0F1BFC1403DC07940DD3F21C07EBF9A3FCC3F743E364047C0993F7AC0534087C0393FA9C0254018C0D43D62409DC07CC06BBFF53F27C08E4095BF4CC0943FBA3FC03F7EC0A2BF60BF88BFA73F9BBF3340E2BEE93FED3FAEBEB03F423FAE3FBAC0FE3EA93F2A407CC0B3C0813F7BC03FC0D8BF1EC0B64047C076BFF4C062BFA9C0E13F10C06DBDEE3F11405B407540883F27C0AABF38406E401EBFD7BF003F7440AFBF55C009BF6FC06F40DFBF734048BF3DC0A1BF1D40EFBFD83FBC3F034125C0EF3F593F66405DC0BCBF5DBFBCBE19407EBF853EBE3F5AC009C0893E35C04A40184092C053BFC63FA53F5140ED3F84C0A8C0BA3ECB3FABBFB73F60401BBE7CC0DEBFFE40ACC09A3F894048C02BBF9540F540B33EE3BF3040D83FEE3F96C0E6BFA13FF4BF8C4097BFD34004C0D1BF0FC060C0B14005C0C8BE74C0EC3F34C02EC09B403040144061C0C7C07DC0943F5E40C63F88C0D63E4C40B43E024024BE543F0E405640AD40143FBCBF143FFFBB29BFCA3FBFC0054040BEF4BFD9BF9BBF8BC0"> : tensor<5x6x7xbf16>
    %1 = stablehlo.constant dense<[[2.296880e+00, -2.203130e+00, -1.085940e+00, -2.484380e+00, -1.828130e+00, 6.625000e+00, -8.007810e-01], [-4.980470e-01, 8.789060e-01, 2.390630e+00, -6.289060e-01, 6.468750e+00, 5.312500e-01, 1.164060e+00]]> : tensor<2x7xbf16>
    return %0, %1 : tensor<5x6x7xbf16>, tensor<2x7xbf16>
  }
  func.func private @expected() -> tensor<5x6x7xbf16> {
    %0 = stablehlo.constant dense<"0xEF3E0A3B28C0B9C080C0AC3E2D4013400DC08BBF1FC0EABFD4404DBF543E8740AC3E18C0F1BFC1403DC07940DD3F21C07EBF9A3FCC3F743E364047C0993F7AC0534087C0393FA9C0254018C0D43D62409DC07CC06BBFF53F27C08E4095BF4CC0943FBA3FC03F7EC0A2BF60BF88BFA73F9BBF3340E2BEE93FED3FAEBEB03F423FAE3FBAC0FE3EA93F2A407CC0B3C0813F7BC03FC0D8BF1EC0B64047C076BFF4C062BFA9C0E13F10C06DBDEE3F11405B407540883F27C0AABF38406E401EBFD7BF003F7440AFBF55C009BF6FC06F40DFBF7340FFBE613F194021BFCF40083F953F034125C0EF3F593F66405DC0BCBF5DBFBCBE19407EBF853EBE3F5AC009C0893E35C04A40184092C053BFC63FA53F5140ED3F84C0A8C0BA3ECB3FABBFB73F60401BBE7CC0DEBFFE40ACC09A3F894048C02BBF9540F540B33EE3BF3040D83FEE3F96C0E6BFA13FF4BF8C4097BFD34004C0D1BF0FC060C0B14005C0C8BE74C0EC3F34C02EC09B403040144061C0C7C07DC0943F5E40C63F88C0D63E4C40B43E024024BE543F0E405640AD40143FBCBF143FFFBB29BFCA3FBFC0054040BEF4BFD9BF9BBF8BC0"> : tensor<5x6x7xbf16>
    return %0 : tensor<5x6x7xbf16>
  }
}


// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xi16>, tensor<i16>)
    %1 = call @expected() : () -> tensor<2x7xi16>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -2], high = [0, -2], interior = [0, 4] : (tensor<2x3xi16>, tensor<i16>) -> tensor<2x7xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x7xi16>, tensor<2x7xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi16>, tensor<i16>) {
    %0 = stablehlo.constant dense<0> : tensor<2x3xi16>
    %1 = stablehlo.constant dense<0> : tensor<i16>
    return %0, %1 : tensor<2x3xi16>, tensor<i16>
  }
  func.func private @expected() -> tensor<2x7xi16> {
    %0 = stablehlo.constant dense<0> : tensor<2x7xi16>
    return %0 : tensor<2x7xi16>
  }
}

// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x2x3xf16>, tensor<1x3xf16>)
    %2 = call @expected() : () -> tensor<1x2x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x2x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x2x3xf16>, tensor<1x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x2x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<[[[3.328130e+00, 6.499020e-01, 7.339840e+00], [2.210940e+00, 4.841310e-01, 1.777340e+00]]]> : tensor<1x2x3xf16>
    %1 = stablehlo.constant dense<[[-3.804690e+00, 3.628910e+00, -1.456050e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x2x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x2x3xf16> {
    %0 = stablehlo.constant dense<[[[3.328130e+00, 6.499020e-01, 7.339840e+00], [2.210940e+00, 3.628910e+00, 1.777340e+00]]]> : tensor<1x2x3xf16>
    return %0 : tensor<1x2x3xf16>
  }
}


// RUN: mlir-hlo-opt %s --stablehlo-legalize-to-linalg --split-input-file \
// RUN:   --canonicalize | \
// RUN: FILECHECK_OPTS="" FileCheck %s

// RUN: mlir-hlo-opt %s --stablehlo-legalize-to-linalg="enable-primitive-ops=true" \
// RUN:   --split-input-file --canonicalize | \
// RUN: FILECHECK_OPTS="" FileCheck %s --check-prefix=CHECK-PRIMITIVE

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @float_add
// CHECK-PRIMITIVE-LABEL: func @float_add
func.func @float_add(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = arith.addf %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.addf
  %0 = "stablehlo.add"(%lhs, %rhs) {someattr}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_add_dynamic_encoding
// CHECK-PRIMITIVE-LABEL: func @float_add_dynamic_encoding
func.func @float_add_dynamic_encoding(
  %lhs: tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>,
  %rhs: tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>)
    -> tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.yield

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.addf
  %0 = "stablehlo.add"(%lhs, %rhs)
      : (tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>,
         tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>)
      -> tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>
  func.return %0 : tensor<2x?xf32, #stablehlo.type_extensions<bounds = [?, 2]>>
}

// -----

// CHECK-LABEL: integer_add
// CHECK-PRIMITIVE-LABEL: integer_add
func.func @integer_add(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: addi
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: addi
  %0 = "stablehlo.add"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_add
// CHECK-PRIMITIVE-LABEL: complex_add
func.func @complex_add(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.add
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.add
  %0 = "stablehlo.add"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_atan2
// CHECK-PRIMITIVE-LABEL: func @complex_atan2
func.func @complex_atan2(%lhs: tensor<2x2xcomplex<f32>>,
    %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "stablehlo.atan2"(%lhs, %rhs)
      : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
      -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.atan2
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.atan2
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}


// -----

// CHECK-LABEL: func @float_mul
// CHECK-PRIMITIVE-LABEL: func @float_mul
func.func @float_mul(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: mulf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: mulf
  %0 = "stablehlo.multiply"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_mul
// CHECK-PRIMITIVE-LABEL: func @integer_mul
func.func @integer_mul(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: muli
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: muli
  %0 = "stablehlo.multiply"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_mul
// CHECK-PRIMITIVE-LABEL: func @complex_mul
func.func @complex_mul(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.mul
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.mul
  %0 = "stablehlo.multiply"(%lhs, %rhs)
          : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_remainder
// CHECK-PRIMITIVE-LABEL: func @float_remainder
func.func @float_remainder(%lhs: tensor<2x2xf32>,
                      %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: remf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: remf
  %0 = "stablehlo.remainder"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_remainder
// CHECK-PRIMITIVE-LABEL: func @integer_remainder
func.func @integer_remainder(%lhs: tensor<2x2xi32>,
                        %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.remsi
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.remsi
  %0 = "stablehlo.remainder"(%lhs, %rhs) : (tensor<2x2xi32>,
                                          tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @population_count_integer
// CHECK-PRIMITIVE-LABEL: func @population_count_integer
func.func @population_count_integer(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: math.ctpop
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ctpop
  %0 = "stablehlo.popcnt"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_sqrt
// CHECK-PRIMITIVE-LABEL: func @complex_sqrt
func.func @complex_sqrt(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "stablehlo.sqrt"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.sqrt
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sqrt
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_rsqrt
// CHECK-PRIMITIVE-LABEL: func @float_rsqrt
func.func @float_rsqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "stablehlo.rsqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: linalg.generic
  // CHECK: rsqrt
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: rsqrt
  func.return %tensor_result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_rsqrt
// CHECK-PRIMITIVE-LABEL: func @complex_rsqrt
func.func @complex_rsqrt(%operand: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "stablehlo.rsqrt"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.rsqrt
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.rsqrt
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_cbrt
// CHECK-PRIMITIVE-LABEL: func @float_cbrt
func.func @float_cbrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "stablehlo.cbrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:.+]] = math.cbrt %[[IN]]
  // CHECK: linalg.yield %[[RESULT]]

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.cbrt
  func.return %tensor_result : tensor<2x2xf32>
}

// -----


// CHECK-LABEL: func @float_sub
// CHECK-PRIMITIVE-LABEL: func @float_sub
func.func @float_sub(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: subf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: subf
  %0 = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_sub
// CHECK-PRIMITIVE-LABEL: func @integer_sub
func.func @integer_sub(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: subi
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: subi
  %0 = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_sub
// CHECK-PRIMITIVE-LABEL: complex_sub
func.func @complex_sub(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sub
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sub
  %0 = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_abs
// CHECK-PRIMITIVE-LABEL: func @float_abs
func.func @float_abs(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: math.absf
  // CHECK-PRIMITIVE: linalg.map { math.absf }
  // CHECK-PRIMITIVE-SAME: ins(
  // CHECK-PRIMITIVE-SAME: outs(
  // CHECK-PRIMITIVE-SAME: {someattr}
  %0 = "stablehlo.abs"(%arg0) {someattr} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_exp
// CHECK-PRIMITIVE-LABEL: func @float_exp
func.func @float_exp(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: exp
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: exp
  %0 = "stablehlo.exponential"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_exp
func.func @complex_exp(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.exp
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.exp
  %0 = "stablehlo.exponential"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                 -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_expm1
func.func @float_expm1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: expm1
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: expm1
  %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_expm1
func.func @complex_expm1(%arg0: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.expm1
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.expm1
  %0 = "stablehlo.exponential_minus_one"(%arg0)
    : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_log
func.func @float_log(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.log
  %0 = "stablehlo.log"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log
func.func @complex_log(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.log
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.log
  %0 = "stablehlo.log"(%arg0) : (tensor<2x2xcomplex<f32>>)
                         -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_log1p
// CHECK-PRIMITIVE-LABEL: func @float_log1p
func.func @float_log1p(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log1p
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.log1p
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log1p
// CHECK-PRIMITIVE-LABEL: func @complex_log1p
func.func @complex_log1p(%arg0: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.log1p
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.log1p
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                  -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_logistic
// CHECK-PRIMITIVE-LABEL: func @float_logistic
func.func @float_logistic(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[C1:.*]] = arith.constant 1.{{.*}}e+00
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG:.*]]: f32, %{{.*}}: f32):
  // CHECK: %[[NEG_ARG:.*]] = arith.negf %[[ARG]]
  // CHECK: %[[EXP_NEG_ARG:.*]] = math.exp %[[NEG_ARG]]
  // CHECK: %[[ONE_ADD_EXP_NEG_ARG:.*]] = arith.addf %[[EXP_NEG_ARG]], %[[C1]]
  // CHECK: %[[RESULT:.*]] = arith.divf %[[C1]], %[[ONE_ADD_EXP_NEG_ARG]]
  // CHECK: linalg.yield %[[RESULT]]
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.negf
  // CHECK-PRIMITIVE: math.exp
  // CHECK-PRIMITIVE: arith.addf
  // CHECK-PRIMITIVE: arith.divf
  %0 = "stablehlo.logistic"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_logistic
func.func @complex_logistic(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG:.*]]: complex<f32>, %{{.*}}: complex<f32>):
  // CHECK: %[[NEG_ARG:.*]] = complex.neg %[[ARG]]
  // CHECK: %[[EXP_NEG_ARG:.*]] = complex.exp %[[NEG_ARG]]
  // CHECK: %[[CC1:.*]] = complex.create %[[C1]], %[[C0]] : complex<f32>
  // CHECK: %[[ONE_ADD_EXP_NEG_ARG:.*]] = complex.add %[[EXP_NEG_ARG]], %[[CC1]]
  // CHECK: %[[RESULT:.*]] = complex.div %[[CC1]], %[[ONE_ADD_EXP_NEG_ARG]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "stablehlo.logistic"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_ceil
// CHECK-PRIMITIVE-LABEL: func @float_ceil
func.func @float_ceil(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.ceil
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ceil
  %0 = "stablehlo.ceil"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @floor
// CHECK-PRIMITIVE-LABEL: func @floor
func.func @floor(%input: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.floor
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.floor
  %0 = "stablehlo.floor"(%input) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_neg
// CHECK-PRIMITIVE-LABEL: func @float_neg
func.func @float_neg(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: negf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: negf
  %0 = "stablehlo.negate"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_neg
// CHECK-PRIMITIVE-LABEL: func @complex_neg
func.func @complex_neg(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.neg
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.neg
  %0 = "stablehlo.negate"(%arg0) : (tensor<2x2xcomplex<f32>>)
                            -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_sign
// CHECK-PRIMITIVE-LABEL: func @complex_sign
func.func @complex_sign(
    %arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sign
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sign
  %0 = "stablehlo.sign"(%arg0) : (tensor<2x2xcomplex<f32>>)
                          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_tanh
// CHECK-PRIMITIVE-LABEL: func @float_tanh
func.func @float_tanh(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: tanh
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: tanh
  %0 = "stablehlo.tanh"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_tanh
// CHECK-PRIMITIVE-LABEL: func @complex_tanh
func.func @complex_tanh(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "stablehlo.tanh"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.tanh
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.tanh
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @integer_and
// CHECK-PRIMITIVE-LABEL: func @integer_and
func.func @integer_and(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: and
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: and
  %0 = "stablehlo.and"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_or
// CHECK-PRIMITIVE-LABEL: func @integer_or
func.func @integer_or(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: or
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: or
  %0 = "stablehlo.or"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_xor
// CHECK-PRIMITIVE-LABEL: func @integer_xor
func.func @integer_xor(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: xor
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: xor
  %0 = "stablehlo.xor"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @count_leading_zeros
// CHECK-PRIMITIVE-LABEL: func @count_leading_zeros
func.func @count_leading_zeros(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: math.ctlz
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ctlz
  %0 = "stablehlo.count_leading_zeros"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @float_cmp
// CHECK-PRIMITIVE-LABEL: func @float_cmp
func.func @float_cmp(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction EQ>}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf oeq, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.cmpf

// -----

// CHECK-LABEL: func @float_cmp_ne
// CHECK-PRIMITIVE-LABEL: func @float_cmp_ne
func.func @float_cmp_ne(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction NE>}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf une, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.cmpf

// -----

// CHECK-LABEL: func @float_cmp_totalorder
// CHECK-PRIMITIVE-LABEL: func @float_cmp_totalorder
func.func @float_cmp_totalorder(%lhs: tensor<2x2xbf16>,
                %rhs: tensor<2x2xbf16>) -> (tensor<2x2xi1>) {
  %0 = "stablehlo.compare"(%lhs, %rhs) {
    comparison_direction = #stablehlo<comparison_direction LT>,
    compare_type = #stablehlo<comparison_type TOTALORDER>
  } : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i16
// CHECK-DAG: %[[C32767:.*]] = arith.constant 32767 : i16
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: bf16, %[[RHS_IN:.*]]: bf16, %{{.*}}: i1):
// CHECK-NEXT:   %[[LHS_INT:.*]] = arith.bitcast %[[LHS_IN]] : bf16 to i16
// CHECK-NEXT:   %[[LHS_CMP:.*]] = arith.cmpi slt, %[[LHS_INT]], %[[C0]] : i16
// CHECK-NEXT:   %[[LHS_SUB:.*]] = arith.subi %[[C32767]], %[[LHS_INT]] : i16
// CHECK-NEXT:   %[[LHS_SELECT:.*]] = arith.select %[[LHS_CMP]], %[[LHS_SUB]], %[[LHS_INT]] : i16
// CHECK-NEXT:   %[[RHS_INT:.*]] = arith.bitcast %[[RHS_IN]] : bf16 to i16
// CHECK-NEXT:   %[[RHS_CMP:.*]] = arith.cmpi slt, %[[RHS_INT]], %[[C0]] : i16
// CHECK-NEXT:   %[[RHS_SUB:.*]] = arith.subi %[[C32767]], %[[RHS_INT]] : i16
// CHECK-NEXT:   %[[RHS_SELECT:.*]] = arith.select %[[RHS_CMP]], %[[RHS_SUB]], %[[RHS_INT]] : i16
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_SELECT]], %[[RHS_SELECT]] : i16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// CHECK-PRIMITIVE-DAG: %[[C0:.*]] = arith.constant 0 : i16
// CHECK-PRIMITIVE-DAG: %[[C32767:.*]] = arith.constant 32767 : i16
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE-SAME: ins(
// CHECK-PRIMITIVE-SAME: outs(
// CHECK-PRIMITIVE-NEXT: (%[[LHS_IN:[a-zA-Z0-9]*]]: bf16, %[[RHS_IN:.*]]: bf16) {
// CHECK-PRIMITIVE-NEXT:   %[[LHS_INT:.*]] = arith.bitcast %[[LHS_IN]] : bf16 to i16
// CHECK-PRIMITIVE-NEXT:   %[[LHS_CMP:.*]] = arith.cmpi slt, %[[LHS_INT]], %[[C0]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[LHS_SUB:.*]] = arith.subi %[[C32767]], %[[LHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[LHS_SELECT:.*]] = arith.select %[[LHS_CMP]], %[[LHS_SUB]], %[[LHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_INT:.*]] = arith.bitcast %[[RHS_IN]] : bf16 to i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_CMP:.*]] = arith.cmpi slt, %[[RHS_INT]], %[[C0]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_SUB:.*]] = arith.subi %[[C32767]], %[[RHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_SELECT:.*]] = arith.select %[[RHS_CMP]], %[[RHS_SUB]], %[[RHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_SELECT]], %[[RHS_SELECT]] : i16
// CHECK-PRIMITIVE-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @int_cmp
// CHECK-PRIMITIVE-LABEL: func @int_cmp
func.func @int_cmp(%lhs: tensor<2x2xi32>,
              %rhs: tensor<2x2xi32>) -> tensor<2x2xi1> {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction LT>}
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi1>)
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.cmpi

// -----

// CHECK-LABEL: func @complex_cmp_eq
// CHECK-PRIMITIVE-LABEL: func @complex_cmp_eq
func.func @complex_cmp_eq(%lhs: tensor<2xcomplex<f32>>,
                     %rhs: tensor<2xcomplex<f32>>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction EQ>}
          : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: tensor.empty() : tensor<2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f32>, %[[RHS_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.eq %[[LHS_IN]], %[[RHS_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: complex.eq

// -----

// CHECK-LABEL: func @complex_cmp_neq
// CHECK-PRIMITIVE-LABEL: func @complex_cmp_neq
func.func @complex_cmp_neq(%lhs: tensor<2xcomplex<f64>>,
                      %rhs: tensor<2xcomplex<f64>>) -> tensor<2xi1> {
  %0 = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction NE>}
          : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: tensor.empty() : tensor<2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f64>, %[[RHS_IN:.*]]: complex<f64>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.neq %[[LHS_IN]], %[[RHS_IN]] : complex<f64>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: complex.neq

// -----

// CHECK-LABEL: func @float_cos
// CHECK-PRIMITIVE-LABEL: func @float_cos
func.func @float_cos(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.cos
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.cos
  %0 = "stablehlo.cosine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_cos
// CHECK-PRIMITIVE-LABEL: func @complex_cos
func.func @complex_cos(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.cos
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.cos
  %0 = "stablehlo.cosine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_sin
// CHECK-PRIMITIVE-LABEL: func @float_sin
func.func @float_sin(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.sin
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.sin
  %0 = "stablehlo.sine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_sin
// CHECK-PRIMITIVE-LABEL: func @complex_sin
func.func @complex_sin(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sin
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sin
  %0 = "stablehlo.sine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @is_finte
// CHECK-PRIMITIVE-LABEL: func @is_finte
func.func @is_finte(%input: tensor<2x2xf32>) -> tensor<2x2xi1> {
  %0 = "stablehlo.is_finite"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: %[[POS_INF:.+]] = arith.constant 0x7F800000 : f32
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32
// CHECK-NEXT:   %[[ABS_X:.+]] = math.absf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.+]] = arith.cmpf one, %[[ABS_X]], %[[POS_INF]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: math.absf
// CHECK-PRIMITIVE: arith.cmpf

// -----

// CHECK-LABEL: func @round_nearest_even
// CHECK-PRIMITIVE-LABEL: func @round_nearest_even
func.func @round_nearest_even(%val: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[ROUND:.+]] = math.roundeven %[[IN]]
  // CHECK: linalg.yield %[[ROUND]]
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.roundeven
  %0 = "stablehlo.round_nearest_even"(%val) : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @round
// CHECK-PRIMITIVE-LABEL: func @round
func.func @round(%val: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[ROUND:.+]] = math.round %[[IN]]
  // CHECK: linalg.yield %[[ROUND]]
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.round
  %0 = "stablehlo.round_nearest_afz"(%val) : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @select
func.func @select(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>,
             %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.select"(%pred, %lhs, %rhs)
         : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}
// CHECK: tensor.empty() : tensor<2x2xf32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[PRED_IN:.*]]: i1, %[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[PRED_IN]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE-LABEL: func @select
// CHECK-PRIMITIVE: tensor.empty() : tensor<2x2xf32>
// CHECK-PRIMITIVE: linalg.map { arith.select }
// CHECK-PRIMITIVE-SAME: ins(
// CHECK-PRIMITIVE-SAME: outs(

// -----

// CHECK-DAG:   #[[SCALAR_MAP:.*]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG:   #[[ID_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @select_scalar_pred_dyn
// CHECK-SAME:  (%[[PRED:.*]]: tensor<i1>, %[[LHS:.*]]: tensor<2x?xf32>, %[[RHS:.*]]: tensor<2x?xf32>)
func.func @select_scalar_pred_dyn(%pred : tensor<i1>, %lhs: tensor<2x?xf32>, %rhs: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %0 = "stablehlo.select"(%pred, %lhs, %rhs) {someattr} : (tensor<i1>, tensor<2x?xf32>, tensor<2x?xf32>) -> (tensor<2x?xf32>)
  func.return %0 : tensor<2x?xf32>
}
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-DAG:  %[[DIM:.*]] = tensor.dim %[[LHS]], %[[C1]]
// CHECK-DAG:  %[[DST:.*]] = tensor.empty(%[[DIM]])
// CHECK:      linalg.generic
// CHECK-SAME:   indexing_maps = [#[[SCALAR_MAP]], #[[ID_MAP]], #[[ID_MAP]], #[[ID_MAP]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel"]
// CHECK-SAME:   ins(%[[PRED]], %[[LHS]], %[[RHS]] : tensor<i1>, tensor<2x?xf32>, tensor<2x?xf32>)
// CHECK-SAME:   outs(%[[DST]] : tensor<2x?xf32>)
// CHECK-SAME:   {someattr}
// CHECK:      ^bb0(%[[PRED_:.*]]: i1, %[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %{{.*}}: f32):
// CHECK:        %[[RES:.*]] = arith.select %[[PRED_]], %[[LHS_]], %[[RHS_]] : f32
// CHECK:        linalg.yield %[[RES]]

// CHECK-PRIMITIVE-LABEL: func @select_scalar_pred_dyn
// CHECK-PRIMITIVE-SAME:  (%[[PRED:.*]]: tensor<i1>, %[[LHS:.*]]: tensor<2x?xf32>, %[[RHS:.*]]: tensor<2x?xf32>)
// CHECK-PRIMITIVE-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-PRIMITIVE-DAG:  %[[DIM:.*]] = tensor.dim %[[LHS]], %[[C1]]
// CHECK-PRIMITIVE-DAG:  %[[DST:.*]] = tensor.empty(%[[DIM]])
// CHECK-PRIMITIVE-DAG:  %[[PRED_ELEM:.*]] = tensor.extract %[[PRED]]
// CHECK-PRIMITIVE:      linalg.map
// CHECK-PRIMITIVE-SAME:   ins(%[[LHS]], %[[RHS]] : tensor<2x?xf32>, tensor<2x?xf32>)
// CHECK-PRIMITIVE-SAME:   outs(%[[DST]] : tensor<2x?xf32>)
// CHECK-PRIMITIVE-SAME:   {someattr}
// CHECK-PRIMITIVE:      (%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32) {
// CHECK-PRIMITIVE:        %[[RES:.*]] = arith.select %[[PRED_ELEM]], %[[LHS_]], %[[RHS_]] : f32
// CHECK-PRIMITIVE:        linalg.yield %[[RES]]

// -----

// CHECK-LABEL: func @select_mixed
func.func @select_mixed(%pred: tensor<2x?xi1>, %lhs: tensor<?x2xf32>,
             %rhs: tensor<2x2xf32>) -> tensor<?x2xf32> {
  %0 = "stablehlo.select"(%pred, %lhs, %rhs)
         : (tensor<2x?xi1>, tensor<?x2xf32>, tensor<2x2xf32>) -> (tensor<?x2xf32>)
  func.return %0 : tensor<?x2xf32>
}
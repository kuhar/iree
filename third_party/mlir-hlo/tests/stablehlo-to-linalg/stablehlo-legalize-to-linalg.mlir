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
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = arith.addf %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.addf
  %0 = "mhlo.add"(%lhs, %rhs)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_add_dynamic_encoding
// CHECK-PRIMITIVE-LABEL: func @float_add_dynamic_encoding
func.func @float_add_dynamic_encoding(
  %lhs: tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>,
  %rhs: tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>)
    -> tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.yield

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.addf
  %0 = "mhlo.add"(%lhs, %rhs)
      : (tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>,
         tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>)
      -> tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>
  func.return %0 : tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>
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
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<2x2xi32>,
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
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

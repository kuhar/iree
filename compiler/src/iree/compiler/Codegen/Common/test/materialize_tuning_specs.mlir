// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec.mlir \
// RUN:   --iree-codegen-dump-tuning-specs-to=- \
// RUN:   --mlir-disable-threading --no-implicit-module %s | FileCheck %s --check-prefix=USER

// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-dump-tuning-specs-to=- \
// RUN:   --iree-gpu-test-target=gfx942 --mlir-disable-threading \
// RUN:   --no-implicit-module %s | FileCheck %s --check-prefix=DEFAULT

// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec.mlir \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-dump-tuning-specs-to=- \
// RUN:   --iree-gpu-test-target=gfx942 --mlir-disable-threading \
// RUN:   --no-implicit-module %s | FileCheck %s --check-prefix=BOTH

// ============================================================================

// Check that the final tuning spec is as expected when the user tuning spec is provided.

// USER-LABEL: module @iree_linked_tuning_spec attributes {transform.with_named_sequence}
// USER-LABEL:   module @user_spec_0 attributes {transform.with_named_sequence}
// USER-LABEL:     transform.named_sequence @hello
// USER-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
// USER-LABEL:   transform.named_sequence @__kernel_config
// USER:           @user_spec_0::@hello

// Check that the transform spec gets materialized as a module attribute.
// USER:        module attributes
// USER-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
// USER-LABEL:    func.func @main_0

// ============================================================================

// Check that the default tuning spec gets materialized without linking.

// DEFAULT-LABEL: module @iree_default_tuning_spec_gfx942 attributes {transform.with_named_sequence}
// DEFAULT-LABEL:   transform.named_sequence @__kernel_config
// DEFAULT-SAME:      attributes {iree_codegen.tuning_spec_entrypoint}

// Check that the default tuning spec gets materialized as a module attribute.
// DEFAULT:        module attributes
// DEFAULT-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
// DEFAULT-LABEL:    func.func @main_0

// ============================================================================

// Check that both the user tuning spec and the default spec get linked and
// materialized. The user spec should have precedence over the default one.

// BOTH-LABEL: module @iree_linked_tuning_spec attributes {transform.with_named_sequence}
// BOTH-LABEL:   module @user_spec_0 attributes {transform.with_named_sequence}
// BOTH-LABEL:     transform.named_sequence @hello
// BOTH-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
// BOTH-LABEL:   module @iree_default_tuning_spec_gfx942_1 attributes {transform.with_named_sequence}
// BOTH:           transform.named_sequence @__kernel_config
// BOTH-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
// BOTH:         transform.named_sequence @__kernel_config
// BOTH:           @user_spec_0::@hello
// BOTH:           @iree_default_tuning_spec_gfx942_1::@__kernel_config

// BOTH:        module attributes
// BOTH-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
// BOTH-LABEL:    func.func @main_0

module {
  func.func @main_0() {
    return
  }
}

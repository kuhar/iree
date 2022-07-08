// RUN: iree-opt --split-input-file --iree-stream-elide-timepoints %s | FileCheck %s

// Tests that an immediate timepoint passed along a call edge is propagated.

// function call/return

// -----

// Tests that an immediate timepoint passed along a block edge is propagated.

// function with block

// -----

// Tests that back edges with divergent timepoints don't get propagated.

// initial and back edge

// -----

// Tests that mutable global timepoints are properly preserved.

util.global private mutable @global = #stream.timepoint<immediate> : !stream.timepoint

// CHECK-LABEL: func @multiStep
func.func @multiStep() {
  // CHECK: %[[STEP0:.+]]:2 = call @step0
  %0:2 = call @step0() : () -> (!stream.timepoint, !stream.timepoint)
  // CHECK-NEXT: %[[STEP1:.+]] = call @step1(%[[STEP0]]#0, %[[STEP0]]#1)
  %1 = call @step1(%0#0, %0#1) : (!stream.timepoint, !stream.timepoint) -> !stream.timepoint
  // CHECK-NEXT: util.do_not_optimize(%[[STEP1]])
  %2 = util.do_not_optimize(%1) : !stream.timepoint
  return
}
// CHECK-LABEL: func private @step0
func.func private @step0() -> (!stream.timepoint, !stream.timepoint) {
  // CHECK: %[[GLOBAL:.+]] = util.global.load @global
  %global = util.global.load @global : !stream.timepoint
  // CHECK: %[[EXECUTE:.+]] = stream.cmd.execute await(%[[GLOBAL]])
  %0 = stream.cmd.execute await(%global) => with() {} => !stream.timepoint
  // CHECK: %[[IMM0:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[EXECUTE]], %[[IMM0]])
  %1 = stream.timepoint.join max(%0, %global) => !stream.timepoint
  // CHECK: util.global.store %[[JOIN]]
  util.global.store %1, @global : !stream.timepoint
  // CHECK: return %[[EXECUTE]], %[[JOIN]]
  return %0, %1 : !stream.timepoint, !stream.timepoint
}
// CHECK-LABEL: func private @step1
// CHECK-SAME: (%[[WAIT0:.+]]: !stream.timepoint, %[[WAIT1:.+]]: !stream.timepoint)
func.func private @step1(%wait0: !stream.timepoint, %wait1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[GLOBAL_IMM:.+]] = stream.timepoint.immediate
  %global = util.global.load @global : !stream.timepoint
  // CHECK: %[[JOIN0:.+]] = stream.timepoint.join max(%[[WAIT0]], %[[WAIT1]], %[[GLOBAL_IMM]])
  %0 = stream.timepoint.join max(%wait0, %wait1, %global) => !stream.timepoint
  // CHECK: %[[EXECUTE:.+]] = stream.cmd.execute await(%[[JOIN0]])
  %1 = stream.cmd.execute await(%0) => with() {} => !stream.timepoint
  // CHECK-DAG: %[[IMM0:.+]] = stream.timepoint.immediate
  // CHECK-DAG: %[[IMM1:.+]] = stream.timepoint.immediate
  // CHECK-DAG: %[[IMM2:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN1:.+]] = stream.timepoint.join max(%[[IMM2]], %[[IMM1]], %[[EXECUTE]], %[[IMM0]])
  %2 = stream.timepoint.join max(%wait1, %0, %1, %global) => !stream.timepoint
  // CHECK: util.global.store %[[JOIN1]], @global
  util.global.store %2, @global : !stream.timepoint
  // CHECK: return %[[JOIN1]]
  return %2 : !stream.timepoint
}

// -----

// Tests that scf.if regions are handled.

// scf.if timepoint a/b

// -----

// Tests that scf.for regions are handled.

// scf.for timepoint chain

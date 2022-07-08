// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-elide-timepoints"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// Resource usage query/application patterns
//===----------------------------------------------------------------------===//

// Tracks whether a !stream.timepoint is immediately resolved.
// Boolean state will be set to false if any sources are non-immediate.
class IsImmediate
    : public DFX::StateWrapper<DFX::BooleanState, DFX::ValueElement> {
 public:
  using BaseType = DFX::StateWrapper<DFX::BooleanState, DFX::ValueElement>;

  static IsImmediate &createForPosition(const Position &pos,
                                        DFX::Solver &solver) {
    return *(new (solver.getAllocator()) IsImmediate(pos));
  }

  bool isImmediate() const { return isAssumed(); }

  const std::string getName() const override { return "IsImmediate"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return std::string("is_immediate: ") + std::to_string(isAssumed());
  }

 private:
  explicit IsImmediate(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value value, DFX::Solver &solver) override {
    // Immediate timepoints (constant resolved) are always available and cover
    // everything. We check for this as a special case to short-circuit the
    // solver.
    if (isa_and_nonnull<IREE::Stream::TimepointImmediateOp>(
            value.getDefiningOp())) {
      LLVM_DEBUG({
        llvm::dbgs() << "[ElideTimepoints] defined immediate: ";
        value.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      setKnown(true);
      indicateOptimisticFixpoint();
      return;
    }

    // Assume true until proven otherwise.
    setAssumed(true);
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    StateType newState = getState();

    // Scan IR to see if we can identify whether this definitely comes from an
    // immediate op. This will reach across block and call edges and may fan out
    // into many incoming ops - all of them must be immediate for this op to be
    // considered immediate.
    if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
          updateFromDefiningOp(newState, value, result, solver);
          return WalkResult::advance();
        }) == TraversalResult::INCOMPLETE) {
      newState.indicatePessimisticFixpoint();
    }

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  // Updates the usage based on the op defining the value.
  void updateFromDefiningOp(StateType &newState, Value value, OpResult result,
                            DFX::Solver &solver) {
    TypeSwitch<Operation *, void>(result.getOwner())
        .Case([&](IREE::Stream::TimepointImmediateOp op) {
          // Defined by an immediate op; definitely immediate.
        })
        .Case([&](IREE::Stream::TimepointJoinOp op) {
          // Only immediate if all inputs to the join are immediate.
          for (auto operand : op.getAwaitTimepoints()) {
            auto isImmediate = solver.getElementFor<IsImmediate>(
                *this, Position::forValue(operand), DFX::Resolution::REQUIRED);
            LLVM_DEBUG({
              llvm::dbgs() << "[ElideTimepoints] join operand ";
              isImmediate.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            newState ^= isImmediate.getState();
          }
        })
        .Case([&](IREE::Stream::TimelineOpInterface op) {
          // Defined by a timeline operation that ensures it's never immediate.
          LLVM_DEBUG({
            llvm::dbgs() << "[ElideTimepoints] non-immediate timeline op: ";
            value.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << "\n";
          });
          newState.indicatePessimisticFixpoint();
        })
        // Allowed because traversal will take care of things:
        .Case([&](mlir::CallOpInterface) {})
        .Case([&](mlir::BranchOpInterface) {})
        .Case([&](mlir::RegionBranchOpInterface) {})
        .Default([&](Operation *op) {
          // Unknown op defines the value - we can't make any assumptions.
          LLVM_DEBUG({
            llvm::dbgs() << "[ElideTimepoints] unknown usage of ";
            value.printAsOperand(llvm::dbgs(), solver.getAsmState());
            llvm::dbgs() << " by " << op->getName() << "\n";
          });
          newState.indicatePessimisticFixpoint();
        });
  }

  friend class DFX::Solver;
};
const char IsImmediate::ID = 0;

class TimepointCoverage
    : public DFX::StateWrapper<DFX::PotentialValuesState<Value>,
                               DFX::ValueElement> {
 public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialValuesState<Value>, DFX::ValueElement>;

  static TimepointCoverage &createForPosition(const Position &pos,
                                              DFX::Solver &solver) {
    return *(new (solver.getAllocator()) TimepointCoverage(pos));
  }

  const std::string getName() const override { return "TimepointCoverage"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  // Returns true if the given |value| is known to be covered by this value
  // indicating that any time this value is reached |value| must also have been.
  bool covers(Value value) const { return getAssumedSet().contains(value); }

  const std::string getAsStr(AsmState &asmState) const override {
    std::string str;
    llvm::raw_string_ostream sstream(str);
    sstream << "covered: ";
    if (isValidState()) {
      sstream << "[";
      if (isUndefContained()) {
        sstream << "undef, ";
      }
      llvm::interleaveComma(getAssumedSet(), sstream, [&](Value value) {
        value.printAsOperand(sstream, asmState);
      });
      sstream << "]";
    } else {
      sstream << "(invalid)";
    }
    sstream.flush();
    return str;
  }

 private:
  explicit TimepointCoverage(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value value, DFX::Solver &solver) override {
    // Immediate timepoints (constant resolved) are always available and cover
    // everything. We check for this as a special case to short-circuit the
    // solver.
    auto *op = value.getDefiningOp();
    if (isa_and_nonnull<IREE::Stream::TimepointImmediateOp>(op)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[ElideTimepoints] defined immediate: ";
        value.printAsOperand(llvm::dbgs(), solver.getAsmState());
        llvm::dbgs() << "\n";
      });
      indicateOptimisticFixpoint();
      return;
    }
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    StateType newState;

    // Intersect coverage of all incoming block edge operands.
    // This will also step outside the entry block and into callee functions.
    // The intersection prevents back-edges from polluting block arguments.
    auto gatherBlockOperands = [&](BlockArgument blockArg) {
      StateType uniformState;
      bool firstEdge = true;
      if (solver.getExplorer().walkIncomingBlockArgument(
              blockArg, [&](Block *sourceBlock, Value operand) {
                auto operandCoverage = solver.getElementFor<TimepointCoverage>(
                    *this, Position::forValue(operand),
                    DFX::Resolution::REQUIRED);
                LLVM_DEBUG({
                  llvm::dbgs()
                      << "[ElideTimepoints] intersect incoming branch operand ";
                  operandCoverage.print(llvm::dbgs(), solver.getAsmState());
                  llvm::dbgs() << "\n";
                });
                if (firstEdge) {
                  uniformState = operandCoverage.getState();
                  firstEdge = false;
                } else {
                  uniformState.intersectAssumed(operandCoverage.getState());
                }
                return WalkResult::advance();
              }) == TraversalResult::INCOMPLETE) {
        LLVM_DEBUG(llvm::dbgs() << "[ElideTimepoints] incomplete branch arg "
                                   "traversal; assuming unknown");
        uniformState.unionAssumedWithUndef();
      }
      newState ^= uniformState;
      newState.unionAssumed(blockArg);
    };

    // Intersect coverage of all callee/child region return operands.
    // The intersection prevents multiple return sites from interfering.
    auto gatherRegionReturns = [&](Operation *regionOp, unsigned resultIndex) {
      StateType uniformState;
      bool firstEdge = true;
      if (solver.getExplorer().walkReturnOperands(
              regionOp, [&](OperandRange operands) {
                auto operand = operands[resultIndex];
                auto operandCoverage = solver.getElementFor<TimepointCoverage>(
                    *this, Position::forValue(operand),
                    DFX::Resolution::REQUIRED);
                LLVM_DEBUG({
                  llvm::dbgs()
                      << "[ElideTimepoints] intersect incoming return operand ";
                  operandCoverage.print(llvm::dbgs(), solver.getAsmState());
                  llvm::dbgs() << "\n";
                });
                if (firstEdge) {
                  uniformState = operandCoverage.getState();
                  firstEdge = false;
                } else {
                  uniformState.intersectAssumed(operandCoverage.getState());
                }
                return WalkResult::advance();
              }) == TraversalResult::INCOMPLETE) {
        LLVM_DEBUG(llvm::dbgs() << "[ElideTimepoints] incomplete region "
                                   "traversal; assuming unknown");
        uniformState.unionAssumedWithUndef();
      }
      newState ^= uniformState;
    };

    auto *definingOp = value.getDefiningOp();
    if (auto blockArg = value.dyn_cast<BlockArgument>()) {
      // Block arguments need an intersection of all incoming branch/call edges.
      gatherBlockOperands(blockArg);
    } else if (auto timelineOp =
                   dyn_cast<IREE::Stream::TimelineOpInterface>(definingOp)) {
      // Value defined from a timeline op and we can mark all awaits of the op
      // as covered by the result.
      for (auto operand : timelineOp.getAwaitTimepoints()) {
        auto operandCoverage = solver.getElementFor<TimepointCoverage>(
            *this, Position::forValue(operand), DFX::Resolution::REQUIRED);
        LLVM_DEBUG({
          llvm::dbgs() << "[ElideTimepoints] dependent timeline operand ";
          operandCoverage.print(llvm::dbgs(), solver.getAsmState());
          llvm::dbgs() << "\n";
        });
        newState.unionAssumed(operand);
        newState &= operandCoverage;
      }
      // Timepoints cover themselves; this is redundant but simplifies the set
      // logic later on.
      if (auto resultTimepoint = timelineOp.getResultTimepoint()) {
        LLVM_DEBUG({
          llvm::dbgs() << "[ElideTimepoints] produced timeline result ";
          resultTimepoint.printAsOperand(llvm::dbgs(), solver.getAsmState());
          llvm::dbgs() << "\n";
        });
        newState.unionAssumed(resultTimepoint);
      }
    } else if (auto callOp = dyn_cast<mlir::CallOpInterface>(definingOp)) {
      // Step into callees and get a coverage intersection of all return sites.
      auto callableOp =
          callOp.resolveCallable(&solver.getExplorer().getSymbolTables());
      unsigned resultIndex = value.cast<OpResult>().getResultNumber();
      gatherRegionReturns(callableOp, resultIndex);
    } else if (auto regionOp = dyn_cast<RegionBranchOpInterface>(definingOp)) {
      // Step into regions and get a coverage intersection of all return sites.
      unsigned resultIndex = value.cast<OpResult>().getResultNumber();
      gatherRegionReturns(regionOp, resultIndex);
    }

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  friend class DFX::Solver;
};
const char TimepointCoverage::ID = 0;

class TimepointCoverageAnalysis {
 public:
  explicit TimepointCoverageAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpAction<IREE::Util::InitializerOp>(TraversalAction::RECURSE);
    explorer.setOpAction<mlir::func::FuncOp>(TraversalAction::RECURSE);
    explorer.setDialectAction<IREE::Stream::StreamDialect>(
        TraversalAction::RECURSE);
    // Ignore the contents of executables (linalg goo, etc) and execution
    // regions (they don't impact timepoints).
    explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);
    explorer.setOpAction<IREE::Stream::AsyncExecuteOp>(
        TraversalAction::SHALLOW);
    explorer.setOpAction<IREE::Stream::CmdExecuteOp>(TraversalAction::SHALLOW);
    explorer.initialize();

    assert(rootOp->getNumRegions() == 1 && "expected module-like root op");
    topLevelOps = llvm::to_vector<4>(
        rootOp->getRegions().front().getOps<mlir::CallableOpInterface>());
  }

  AsmState &getAsmState() { return solver.getAsmState(); }

  // Runs analysis and populates the state cache.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run() {
    for (auto callableOp : getTopLevelOps()) {
      for (auto &block : *callableOp.getCallableRegion()) {
        // Seed all block arguments.
        for (auto arg : block.getArguments()) {
          if (arg.getType().isa<IREE::Stream::TimepointType>()) {
            solver.getOrCreateElementFor<IsImmediate>(Position::forValue(arg));
          }
        }

        // Seed the timepoints created from any timeline ops.
        for (auto op : block.getOps<IREE::Stream::TimelineOpInterface>()) {
          for (auto operand : op.getAwaitTimepoints()) {
            solver.getOrCreateElementFor<TimepointCoverage>(
                Position::forValue(operand));
            solver.getOrCreateElementFor<IsImmediate>(
                Position::forValue(operand));
          }
          if (auto resultTimepoint = op.getResultTimepoint()) {
            solver.getOrCreateElementFor<TimepointCoverage>(
                Position::forValue(resultTimepoint));
          }
        }
      }
    }

    // Run solver to completion.
    auto result = solver.run();
    LLVM_DEBUG(solver.print(llvm::dbgs()));
    return result;
  }

  // Returns a list of all top-level callable ops in the root op.
  ArrayRef<mlir::CallableOpInterface> getTopLevelOps() const {
    return topLevelOps;
  }

  // Returns true if |value| is known to be immediately resolved.
  bool isImmediate(Value value) {
    auto &isImmediate =
        solver.getOrCreateElementFor<IsImmediate>(Position::forValue(value));
    return isImmediate.isValidState() && isImmediate.isKnown();
  }

  // Union all transitively reached timepoints by the time |value| is reached.
  void unionTransitivelyReachedTimepoints(Value value, SetVector<Value> &set) {
    auto coverage = solver.getOrCreateElementFor<TimepointCoverage>(
        Position::forValue(value));
    if (coverage.isValidState() && !coverage.isUndefContained()) {
      for (auto reached : coverage.getAssumedSet()) {
        set.insert(reached);
      }
    }
  }

 private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;
  SmallVector<mlir::CallableOpInterface> topLevelOps;
};

// Tries to elide timepoints nested within |region| when safe.
// Returns true if any ops were elided.
static bool tryElideTimepointsInRegion(Region &region,
                                       TimepointCoverageAnalysis &analysis) {
  bool didChange = false;
  auto elideTimepoint = [&](Operation *op, Value elidedTimepoint) {
    Value immediateTimepoint =
        OpBuilder(op).create<IREE::Stream::TimepointImmediateOp>(
            elidedTimepoint.getLoc());
    elidedTimepoint.replaceUsesWithIf(
        immediateTimepoint,
        [&](OpOperand &operand) { return operand.getOwner() == op; });
    didChange = true;
  };
  for (auto &block : region) {
    // TODO(benvanik): more aggressively break SSA use-def chains by doing
    // this replacement on all timepoint values for all ops. For now we just
    // check the meaningful users of timepoints and let canonicalization handle
    // the rest.
    for (auto op : block.getOps<IREE::Stream::TimelineOpInterface>()) {
      auto awaitTimepoints = op.getAwaitTimepoints();
      if (awaitTimepoints.empty()) continue;

      LLVM_DEBUG({
        llvm::dbgs() << "[ElideTimepoints] pruning " << op->getName()
                     << " await(";
        llvm::interleaveComma(awaitTimepoints, llvm::dbgs(), [&](Value value) {
          value.printAsOperand(llvm::dbgs(), analysis.getAsmState());
        });
        llvm::dbgs() << ")\n";
      });

      // Prune all immediately reached timepoints.
      // This may let us avoid doing the full pruning pass by getting us down to
      // 0 or 1 timepoints.
      SmallVector<Value> possibleTimepoints;
      for (auto awaitTimepoint : awaitTimepoints) {
        if (analysis.isImmediate(awaitTimepoint)) {
          // Timepoint is definitely immediate and can be pruned.
          LLVM_DEBUG({
            llvm::dbgs() << "  --- eliding use of known-immediate ";
            awaitTimepoint.printAsOperand(llvm::dbgs(), analysis.getAsmState());
            llvm::dbgs() << " in " << op->getName() << "\n";
          });
          elideTimepoint(op, awaitTimepoint);
        } else {
          // May be immediate but not certain; preserve.
          possibleTimepoints.push_back(awaitTimepoint);
        }
      }

      // If there's only one timepoint we don't have to worry with coverage.
      if (possibleTimepoints.size() <= 1) continue;

      // Perform a few passes at pruning the remaining timepoints by removing
      // those covered by others. There's likely a better way of doing this but
      // the back-and-forth scan of the set is easiest (though not very
      // efficient). We could probably use bitmaps to indicate which are covered
      // by which and then some set math to find the minimal required
      // timepoints.
      const int kPruneIterations = 10;
      int lastPrune = -1;
      for (int iteration = 0; iteration < kPruneIterations; ++iteration) {
        LLVM_DEBUG(llvm::dbgs()
                   << " prune pass " << iteration << "/" << kPruneIterations
                   << ": " << possibleTimepoints.size() << " timepoints\n");
        SmallVector<Value> requiredTimepoints;
        SetVector<Value> coveredTimepoints;
        bool didPrune = false;
        for (auto possibleTimepoint : possibleTimepoints) {
          if (coveredTimepoints.count(possibleTimepoint)) {
            // SSA value directly covered by transitive operand values.
            LLVM_DEBUG({
              llvm::dbgs() << "  --- eliding reached ";
              possibleTimepoint.printAsOperand(llvm::dbgs(),
                                               analysis.getAsmState());
              llvm::dbgs() << "\n";
            });
            elideTimepoint(op, possibleTimepoint);
            didPrune = true;
            continue;
          }

          // Build a set of timepoints transitively reached by the timepoint.
          // If all of the timepoints in that set are covered by other operands
          // we can consider this timepoint as reached.
          SetVector<Value> reachedTimepoints;
          analysis.unionTransitivelyReachedTimepoints(possibleTimepoint,
                                                      reachedTimepoints);
          reachedTimepoints.set_subtract(coveredTimepoints);
          if (reachedTimepoints.empty()) {
            LLVM_DEBUG({
              llvm::dbgs() << "  --- eliding reached ";
              possibleTimepoint.printAsOperand(llvm::dbgs(),
                                               analysis.getAsmState());
              llvm::dbgs() << " (transitive coverage)\n";
            });
            elideTimepoint(op, possibleTimepoint);
            didPrune = true;
            continue;
          }

          // This timepoint operand has not yet been reached by the current
          // required set - it may be pruned on the next iteration.
          LLVM_DEBUG({
            llvm::dbgs() << "  +++ preserving ";
            possibleTimepoint.printAsOperand(llvm::dbgs(),
                                             analysis.getAsmState());
            llvm::dbgs() << " missing coverage of: ";
            llvm::interleaveComma(
                reachedTimepoints, llvm::dbgs(), [&](Value value) {
                  value.printAsOperand(llvm::dbgs(), analysis.getAsmState());
                });
            llvm::dbgs() << "\n";
          });
          requiredTimepoints.push_back(possibleTimepoint);
          coveredTimepoints.set_union(reachedTimepoints);
        }
        if (didPrune) {
          lastPrune = iteration;  // changed
        } else if (lastPrune < iteration - 1) {
          break;  // quiesced
        }
        if (requiredTimepoints.size() <= 1) break;  // no more to prune
        possibleTimepoints = requiredTimepoints;
        std::reverse(possibleTimepoints.begin(), possibleTimepoints.end());
      }
    }
  }
  return didChange;
}

//===----------------------------------------------------------------------===//
// -iree-stream-elide-timepoints
//===----------------------------------------------------------------------===//

// Elides waits on timepoints that are known to be reached by a dependent
// timepoint. We err on the side of additional timepoints if we can't guarantee
// that a particular wait is covered.
//
// Example:
//   %timepoint0 = ...
//   %timepoint1 = ... await(%timepoint0)
//   %timepoint2 = stream.timepoint.join max(%timepoint0, %timepoint1)
// ->
//   %timepoint0 = ...
//   %timepoint1 = ... await(%timepoint0)
//   %timepoint2 = stream.timepoint.join max(%timepoint1)
// -> (canonicalization) ->
//   %timepoint0 = ...
//   %timepoint1 = ... await(%timepoint0)
//   %timepoint2 = %timepoint1
class ElideTimepointsPass : public ElideTimepointsBase<ElideTimepointsPass> {
 public:
  ElideTimepointsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) return;

    // Perform whole-program analysis to find for each timepoint what other
    // timepoints are known to be reached.
    TimepointCoverageAnalysis analysis(moduleOp);
    if (failed(analysis.run())) {
      moduleOp.emitError() << "failed to solve for timepoint coverage";
      return signalPassFailure();
    }

    // Apply analysis by replacing known-covered timepoint usage with immediate
    // values. If we change something we'll indicate that so that the parent
    // fixed-point iteration continues.
    bool didChange = false;
    for (auto callableOp : analysis.getTopLevelOps()) {
      didChange = tryElideTimepointsInRegion(*callableOp.getCallableRegion(),
                                             analysis) ||
                  didChange;
    }
    if (didChange) signalFixedPointModified(moduleOp);
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createElideTimepointsPass() {
  return std::make_unique<ElideTimepointsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

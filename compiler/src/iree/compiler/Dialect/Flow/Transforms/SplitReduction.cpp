// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SplitReduction.cpp ----------------------------===//
//
// Split reduction dimension to increase parallelism of a linalg operation.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "iree-flow-split-reduction"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::IREE::Flow {

// TODO(thomasraoux): Move to attributes.
static llvm::cl::opt<int64_t>
    splitReductionRatio("iree-flow-split-matmul-reduction",
                        llvm::cl::desc("split ratio"), llvm::cl::init(1));

static llvm::cl::opt<int64_t> splitMatmulKThreshold(
    "iree-flow-split-matmul-k-threshold",
    llvm::cl::desc("minimal k size to participate in split"),
    llvm::cl::init(5000)); // Min 5000 is selected for SDXL

static llvm::cl::list<int64_t> topkSplitReductionRatio(
    "iree-flow-topk-split-reduction",
    llvm::cl::desc("comma separated list of split ratios"),
    llvm::cl::CommaSeparated);

static llvm::cl::list<std::string> mmtReductionRatio(
    "iree-flow-mmt-split-reduction",
    llvm::cl::desc("comma separated list of matmul split reduction configs in "
                   "the MxNxK_T0_T1=ratio format, e.g., "
                   "'2048x1280x5120_f16_f32=4'"),
    llvm::cl::CommaSeparated);

struct MmtReductionConfig {
  int64_t m;
  int64_t n;
  int64_t k;
  Type operandType;
  Type resultType;
};

static SmallVector<MmtReductionConfig> parseMmtReductionRatios(MLIRContext *ctx, llvm::ArrayRef<std::string> configs) {
  return {};
}

inline static SmallVector<NamedAttribute>
getPrunedAttributeList(linalg::LinalgOp op) {
  auto elidedAttrs = llvm::to_vector(linalg::GenericOp::getAttributeNames());
  elidedAttrs.push_back(linalg::LinalgDialect::kMemoizedIndexingMapsAttrName);
  return getPrunedAttributeList(op, elidedAttrs);
}

static LogicalResult splitReductionOnMatmul(
    RewriterBase &rewriter, linalg::LinalgOp op,
    linalg::ControlSplitReductionFn controlSplitReductionFn) {
  // Since user information about compilation are passed through attributes we
  // need to make sure to propagate those.
  SmallVector<NamedAttribute> prunedAttributeList = getPrunedAttributeList(op);

  FailureOr<linalg::SplitReductionResult> result =
      linalg::splitReduction(rewriter, op, controlSplitReductionFn);
  if (failed(result)) {
    return failure();
  }

  result->splitLinalgOp->setAttrs(prunedAttributeList);
  return result;
}

namespace {
struct SplitReductionPass : public SplitReductionBase<SplitReductionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *rootOp = getOperation();
    LDBG("=== Before Split Reduction ===");

    auto hasLargeK = [&](linalg::LinalgOp op) -> bool {
      SmallVector<unsigned> dims;
      op.getReductionDims(dims);
      if (dims.size() != 1)
        return false;
      unsigned reductionDim = dims[0];
      SmallVector<int64_t, 4> loopRanges = op.getStaticLoopRanges();
      int64_t reductionDimSize = loopRanges[reductionDim];
      if (ShapedType::isDynamic(reductionDimSize) ||
          reductionDimSize < splitMatmulKThreshold)
        return false;
      return true;
    };

    SmallVector<linalg::LinalgOp> matmulCandidates;
    DenseMap<linalg::LinalgOp, Attribute> matmulToConfig;
    rootOp->walk([&](linalg::LinalgOp op) {
      if (!linalg::isaContractionOpInterface(op))
        return;

      if (Attribute config =
              op->getAttr("iree_flow_split_reduction_ratio")) {
        LDBG("Found split reduction ratio override (" << config << ") for:\n"
                                                      << op);
        matmulToConfig[op] = config;
        matmulCandidates.push_back(op);
        return;
      }

      if (hasLargeK(op))
        matmulCandidates.push_back(op);
    });

    if (splitReductionRatio.getValue() <= 1 &&
        topkSplitReductionRatio.empty() && matmulToConfig.empty()) {
        LDBG("Nothing to do, bailing out");
      return;
    }

    auto matmulSplitReductionControlFn =
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
      if (auto requestedRatio =
              dyn_cast_or_null<IntegerAttr>(matmulToConfig.lookup(op))) {
        return {requestedRatio.getInt(), 0, /*innerParallel=*/false};
      }

      // For matmul make the new parallel dimension first so that it looks
      // like a batch_matmul and can follow the same codegen.
      return {int64_t(splitReductionRatio), 0, /*innerParallel=*/false};
    };

    IRRewriter rewriter(context);
    for (auto op : matmulCandidates) {
      (void)splitReductionOnMatmul(rewriter, op, matmulSplitReductionControlFn);
    }

    LinalgExt::TopkSplitReductionControlFn topkSplitReductionControlFn =
        [&](int64_t splitReductionDepth) -> int64_t {
      SmallVector<int64_t> reductionRatios(topkSplitReductionRatio.begin(),
                                           topkSplitReductionRatio.end());
      if (splitReductionDepth >= reductionRatios.size()) {
        return -1;
      } else {
        return reductionRatios[splitReductionDepth];
      }
    };

    SmallVector<LinalgExt::TopkOp> topkCandidates;
    rootOp->walk([&](LinalgExt::TopkOp op) { topkCandidates.push_back(op); });
    for (auto op : topkCandidates) {
      (void)splitReduction(rewriter, op, topkSplitReductionControlFn);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createSplitReductionPass() {
  return std::make_unique<SplitReductionPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow

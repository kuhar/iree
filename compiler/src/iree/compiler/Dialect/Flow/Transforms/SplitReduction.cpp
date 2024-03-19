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

#include <cassert>
#include <cstdint>
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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

static llvm::cl::list<std::string> matmulReductionRatio(
    "iree-flow-matmul-split-reduction",
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
  int64_t reductionRatio;
};

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const MmtReductionConfig &config) {
  return os << "[" << config.m << ", " << config.n << ", " << config.k << "]("
            << config.operandType << " -> " << config.resultType
            << ") ==> ratio " << config.reductionRatio;
}

static MmtReductionConfig parseReductionSpec(MLIRContext *ctx, StringRef spec) {
  MmtReductionConfig config = {};
  auto parts = llvm::to_vector_of<StringRef>(llvm::split(spec, '='));
  assert(parts.size() == 2);
  bool status = llvm::to_integer(parts.back(), config.reductionRatio);
  (void)status;
  assert(status);
  StringRef typedShape = parts.front();
  parts = llvm::to_vector_of<StringRef>(llvm::split(typedShape, '_'));
  assert(parts.size() == 3);

  auto strToType = [ctx](StringRef typeName) {
    assert(typeName == "f16" || typeName == "f32");
    return (typeName == "f16") ? FloatType::getF16(ctx)
                               : FloatType::getF32(ctx);
  };
  config.operandType = strToType(parts[1]);
  config.resultType = strToType(parts[2]);

  StringRef shapeStr = parts.front();
  parts = llvm::to_vector_of<StringRef>(llvm::split(shapeStr, 'x'));
  assert(parts.size() == 3);
  status = llvm::to_integer(parts[0], config.m);
  assert(status);
  status = llvm::to_integer(parts[1], config.n);
  assert(status);
  status = llvm::to_integer(parts[2], config.k);
  assert(status);

  return config;
}

static SmallVector<MmtReductionConfig>
parseMmtReductionRatios(MLIRContext *ctx, llvm::ArrayRef<std::string> specs) {
  return llvm::map_to_vector(
      specs, [ctx](StringRef spec) { return parseReductionSpec(ctx, spec); });
}

static bool matchesReductionConfig(linalg::LinalgOp op, const MmtReductionConfig& config) {
  if (!linalg::isaContractionOpInterface(op) || op.getNumParallelLoops() != 2 ||
      op.getNumReductionLoops() != 1) {
    return false;
  }

  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(op);
  assert(succeeded(contractionDims) && "Could not infer contraction dims");

  int64_t mSize = 1, nSize = 1, kSize = 1, batchSize = 1;
  for (auto mDim : contractionDims->m) {
    if (ShapedType::isDynamic(bounds[mDim]))
      return false;
    mSize *= bounds[mDim];
  }
  if (mSize != config.m)
    return false;

  for (auto nDim : contractionDims->n) {
    if (ShapedType::isDynamic(bounds[nDim]))
      return false;
    nSize *= bounds[nDim];
  }
  if (nSize != config.n)
    return false;

  for (auto kDim : contractionDims->k) {
    if (ShapedType::isDynamic(bounds[kDim]))
      return false;
    kSize *= bounds[kDim];
  }
  if (kSize != config.k)
    return false;

  for (auto bDim : contractionDims->batch) {
    if (ShapedType::isDynamic(bounds[bDim]))
      return false;
    batchSize *= bounds[bDim];
  }
  if (batchSize != 1)
    return false;

  auto lhsType = cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  if (lhsType.getElementType() != config.operandType)
    return false;

  auto rhsType = cast<ShapedType>(op.getDpsInputOperand(1)->get().getType());
  if (rhsType.getElementType() != config.operandType)
    return false;

  auto resultType = cast<ShapedType>(op->getResult(0).getType());
  if (resultType.getElementType() != config.resultType)
    return false;

  return true;
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

    SmallVector<MmtReductionConfig> configs = parseMmtReductionRatios(context, matmulReductionRatio);
    LDBG("Found " << configs.size() << " split reduction configs: ");
    LLVM_DEBUG(for (const auto &config
                    : configs) { llvm::dbgs() << "\t" << config << "\n"; });

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
    DenseMap<linalg::LinalgOp, int64_t> matmulToRatio;
    rootOp->walk([&](linalg::LinalgOp op) {
      if (!linalg::isaContractionOpInterface(op))
        return;

      for (const auto &config : configs) {
        if (matchesReductionConfig(op, config)) {
          LDBG("Matched op to CLI ratio of " << config.reductionRatio << "\n"
                                             << op);
          matmulToRatio[op] = config.reductionRatio;
        }
      }

      if (hasLargeK(op))
        matmulCandidates.push_back(op);
    });

    if (splitReductionRatio.getValue() <= 1 &&
        topkSplitReductionRatio.empty() && matmulToRatio.empty()) {
        LDBG("Nothing to do, bailing out");
      return;
    }

    auto matmulSplitReductionControlFn =
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
      if (int64_t requestedRatio = matmulToRatio.lookup(op)) {
        return {requestedRatio, 0, /*innerParallel=*/false};
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

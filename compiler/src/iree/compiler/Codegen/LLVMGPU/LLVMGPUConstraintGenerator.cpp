// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUConstraintGenerator.h"

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Common/SMTConstraintUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

using AssertOp = IREE::Codegen::AssertOp;
using IntKnobAttr = IREE::Codegen::IntKnobAttr;
using OneOfKnobAttr = IREE::Codegen::OneOfKnobAttr;
using RootOpAttr = IREE::Codegen::RootOpAttr;

namespace {

/// Contraction-like dimension classification used by both matmul and conv.
struct ContractionLikeDims {
  SmallVector<unsigned> b;
  SmallVector<unsigned> m;
  SmallVector<unsigned> n;
  SmallVector<unsigned> k;
};

/// Problem size, loop count, and indexing maps for a root op.
struct RootOpLoopInfo {
  SmallVector<int64_t> staticLoopRanges;
  unsigned numLoops;
  SmallVector<AffineMap> indexingMaps;
};

// Keys for entries in the SMT constraints `knobs` dictionary.
// These names are aligned with lowering config and translation info fields.
constexpr StringLiteral kKnobWorkgroupKey = "workgroup";
constexpr StringLiteral kKnobReductionKey = "reduction";
constexpr StringLiteral kKnobMmaKindKey = "mma_kind";
constexpr StringLiteral kKnobSubgroupBasisKey = "subgroup_basis";
constexpr StringLiteral kKnobWorkgroupSizeKey = "workgroup_size";
constexpr StringLiteral kKnobSubgroupSizeKey = "subgroup_size";

// SMT variable names for knob values used in constraints.
constexpr StringLiteral kKnobMmaIdxName = "mma_idx";
constexpr StringLiteral kKnobSgMCntName = "sg_m_cnt";
constexpr StringLiteral kKnobSgNCntName = "sg_n_cnt";
constexpr StringLiteral kKnobSgSizeName = "sg_size";
constexpr StringLiteral kKnobWgSizeXName = "wg_size_x";
constexpr StringLiteral kKnobWgSizeYName = "wg_size_y";
constexpr StringLiteral kKnobWgSizeZName = "wg_size_z";

// SMT variable name prefixes. The loop dim count varies
// per problem, so names are built at runtime as prefix + dim idx.
constexpr StringLiteral kKnobWgPrefix = "wg_";
constexpr StringLiteral kKnobRedPrefix = "red_";

// Default value for dims that are not knobbed.
constexpr int64_t kNoTileDimVal = 0;
constexpr int64_t kUnitTileDimVal = 1;

} // namespace

/// Assert: lhs % rhs == 0, with format args for diagnostics.
static void assertDivisible(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs, StringRef msg) {
  Value zero = mkIntConst(builder, loc, 0);
  Value rem = smt::IntModOp::create(builder, loc, lhs, rhs);
  Value eq = smt::EqOp::create(builder, loc, rem, zero);
  std::string fmtMsg = (msg + " ({} % {} == 0)").str();
  AssertOp::create(builder, loc, eq, fmtMsg, ValueRange{lhs, rhs});
}

/// Assert: lhs <pred> rhs.
static void assertCmp(OpBuilder &builder, Location loc, smt::IntPredicate pred,
                      Value lhs, Value rhs, StringRef msg) {
  Value cmp = smt::IntCmpOp::create(builder, loc, pred, lhs, rhs);
  AssertOp::create(builder, loc, cmp, msg);
}

/// Assert: val >= lo && val <= hi.
static void assertBounds(OpBuilder &builder, Location loc, Value val,
                         StringRef name, int64_t lo, int64_t hi) {
  assertCmp(builder, loc, smt::IntPredicate::ge, val,
            mkIntConst(builder, loc, lo), (name + " >= " + Twine(lo)).str());
  assertCmp(builder, loc, smt::IntPredicate::le, val,
            mkIntConst(builder, loc, hi), (name + " <= " + Twine(hi)).str());
}

/// Helper to build a knob variable name from a prefix.
/// e.g. ("wg_", 2) -> "wg_2".
static std::string makeVarName(StringRef prefix, unsigned idx) {
  return (prefix + Twine(idx)).str();
}

static std::string makeSubgroupTileCountName(unsigned idx) {
  return ("sg_" + Twine(idx) + "_tcnt").str();
}

/// Helper to create an i64 IntegerAttr with a fixed value.
static IntegerAttr makeIntAttr(MLIRContext *ctx, int64_t value = 0) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), value);
}

/// Emit smt.lookup ops to derive MMA m/n/k shape values from the mma_idx knob.
/// Also asserts bounds on mma_idx: 0 <= mma_idx < compatibleMMAs.size().
struct MMADerivedValues {
  Value mmaM;
  Value mmaN;
  Value mmaK;
};

static MMADerivedValues emitMMADerivedValues(OpBuilder &builder, Location loc,
                                             ArrayRef<Attribute> mmaAttrs) {
  assert(!mmaAttrs.empty() && "expected at least one compatible MMA");

  Value mmaIdx = mkKnob(builder, loc, kKnobMmaIdxName);
  assertBounds(builder, loc, mmaIdx, kKnobMmaIdxName, 0,
               static_cast<int64_t>(mmaAttrs.size()) - 1);

  SmallVector<int64_t> keys;
  SmallVector<int64_t> mVals;
  SmallVector<int64_t> nVals;
  SmallVector<int64_t> kVals;
  for (auto [i, attr] : llvm::enumerate(mmaAttrs)) {
    auto [m, n, k] = cast<IREE::GPU::MmaInterfaceAttr>(attr).getMNKShape();
    keys.push_back(static_cast<int64_t>(i));
    mVals.push_back(m);
    nVals.push_back(n);
    kVals.push_back(k);
  }

  smt::IntType intTy = smt::IntType::get(builder.getContext());
  return {
      IREE::Codegen::LookupOp::create(builder, loc, intTy, mmaIdx, keys, mVals),
      IREE::Codegen::LookupOp::create(builder, loc, intTy, mmaIdx, keys, nVals),
      IREE::Codegen::LookupOp::create(builder, loc, intTy, mmaIdx, keys, kVals),
  };
}

/// Get unique compatible MMA attrs for matmul and conv ops.
static SmallVector<Attribute>
getCompatibleMMAAttrs(linalg::LinalgOp op, IREE::GPU::TargetAttr gpuTarget,
                      const RootOpLoopInfo &loopInfo,
                      const ContractionLikeDims &dims) {
  if (gpuTarget.getWgp().getMma().empty()) {
    return {};
  }

  SmallVector<Attribute> mmaAttrs;
  const int64_t targetSubgroupSize = gpuTarget.getPreferredSubgroupSize();
  Type lhsElemType = getElementTypeOrSelf(op.getDpsInputOperand(0)->get());
  Type rhsElemType = getElementTypeOrSelf(op.getDpsInputOperand(1)->get());
  Type initElemType = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
  int64_t mSize = loopInfo.staticLoopRanges[dims.m.back()];
  int64_t nSize = loopInfo.staticLoopRanges[dims.n.back()];
  int64_t kSize = loopInfo.staticLoopRanges[dims.k.back()];

  // Dynamic shapes are not supported by tuner yet.
  if (ShapedType::isDynamic(mSize) || ShapedType::isDynamic(nSize) ||
      ShapedType::isDynamic(kSize)) {
    return {};
  }

  GPUMatmulShapeType problem{mSize,       nSize,       kSize,
                             lhsElemType, rhsElemType, initElemType};

  auto getIntrinsic = [](IREE::GPU::MMAAttr mma) -> GPUIntrinsicType {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    return GPUIntrinsicType{mSize, nSize, kSize, aType, bType, cType, mma};
  };

  for (IREE::GPU::MMAAttr mma : gpuTarget.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize) {
      continue;
    }
    // VectorDistribute matmul/conv skip block intrinsics.
    if (mma.isBlockIntrinsic()) {
      continue;
    }
    if (!mma.getDistributionMappingKind()) {
      continue;
    }
    // Check if the mma intrinsic supports the problem.
    if (failed(canTargetIntrinsic(problem, getIntrinsic(mma),
                                  targetSubgroupSize, /*canUpcastAcc*/ true,
                                  /*mustBeAligned*/ false))) {
      continue;
    }

    if (!llvm::is_contained(mmaAttrs, mma)) {
      mmaAttrs.push_back(mma);
    }
  }
  return mmaAttrs;
}

/// Get contraction-like (m,n,k) dims for a linalg op.
/// Only supports contraction and convolution today.
static FailureOr<ContractionLikeDims>
inferContractionLikeDims(linalg::LinalgOp linalgOp) {
  if (linalg::isaContractionOpInterface(linalgOp)) {
    FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
        mlir::linalg::inferContractionDims(linalgOp);
    if (failed(contractionDims)) {
      return failure();
    }
    return ContractionLikeDims{llvm::to_vector(contractionDims->batch),
                               llvm::to_vector(contractionDims->m),
                               llvm::to_vector(contractionDims->n),
                               llvm::to_vector(contractionDims->k)};
  }
  if (linalg::isaConvolutionOpInterface(linalgOp)) {
    FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
        mlir::linalg::inferConvolutionDims(linalgOp);
    if (failed(convolutionDims) || convolutionDims->outputImage.empty() ||
        convolutionDims->outputChannel.empty() ||
        convolutionDims->inputChannel.empty()) {
      return failure();
    }
    // TODO(Amily): This mapping aligns with how VectorDistribute
    // sets the dims for convs. It may be too coarse for conv
    // semantics; revisit when plumbing through conv constraint
    // generation.
    return ContractionLikeDims{llvm::to_vector(convolutionDims->batch),
                               llvm::to_vector(convolutionDims->outputImage),
                               llvm::to_vector(convolutionDims->outputChannel),
                               llvm::to_vector(convolutionDims->inputChannel)};
  }
  return failure();
}

/// Returns loop info for supported root ops.
static std::optional<RootOpLoopInfo> getRootOpLoopInfo(Operation *rootOp) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    return RootOpLoopInfo{linalgOp.getStaticLoopRanges(),
                          linalgOp.getNumLoops(),
                          linalgOp.getIndexingMapsArray()};
  }
  return std::nullopt;
}

/// Build the VectorDistribute knobs dict for contraction-like dims.
static DictionaryAttr
buildVectorDistributeKnobsDict(MLIRContext *ctx, const RootOpLoopInfo &loopInfo,
                               const ContractionLikeDims &dims,
                               ArrayRef<Attribute> compatibleMMAs) {
  SmallVector<NamedAttribute> knobsEntries;

  // Build workgroup entries from lowering config semantics: untiled dims get 0,
  // batch and outer M/N dims get unit tile 1, and inner M/N dims are knobbed.
  SmallVector<Attribute> workgroupEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  SmallVector<unsigned> unitWorkgroupDims;
  llvm::append_range(unitWorkgroupDims, dims.b);
  llvm::append_range(unitWorkgroupDims, dims.m);
  llvm::append_range(unitWorkgroupDims, dims.n);
  for (unsigned i : unitWorkgroupDims) {
    workgroupEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  // inferContractionLikeDims guarantees dims.m/n/k are non-empty for both
  // branches (asserted or early-returned for conv).
  workgroupEntries[dims.m.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.m.back()));
  workgroupEntries[dims.n.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.n.back()));
  knobsEntries.emplace_back(kKnobWorkgroupKey,
                            ArrayAttr::get(ctx, workgroupEntries));
  // Build reduction entries from the complement of unit workgroup dims.
  // Innermost K dim gets IntKnobAttr, outer K and filter dims get 1.
  SmallVector<Attribute> reductionEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    if (llvm::is_contained(unitWorkgroupDims, i)) {
      continue;
    }
    reductionEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  reductionEntries[dims.k.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobRedPrefix, dims.k.back()));
  knobsEntries.emplace_back(kKnobReductionKey,
                            ArrayAttr::get(ctx, reductionEntries));

  // Add mma_kind knob.
  knobsEntries.emplace_back(
      kKnobMmaKindKey,
      OneOfKnobAttr::get(ctx, kKnobMmaIdxName, compatibleMMAs));

  // Build subgroup_basis as [[counts], [mapping]]. Mapping is an identity map
  // of const int for VectorDistribute matmul and conv. It is kept as a
  // placeholder so downstream constraint verification can match the
  // subgroup_basis knob template in the lowering config.
  // Only innermost M and N dims get subgroup tiling, others stay 1.
  SmallVector<Attribute> subgroupCounts(loopInfo.numLoops, makeIntAttr(ctx, 1));
  subgroupCounts[dims.m.back()] = IntKnobAttr::get(ctx, kKnobSgMCntName);
  subgroupCounts[dims.n.back()] = IntKnobAttr::get(ctx, kKnobSgNCntName);
  SmallVector<Attribute> subgroupMapping;
  subgroupMapping.reserve(loopInfo.numLoops);
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    subgroupMapping.push_back(makeIntAttr(ctx, i));
  }
  ArrayAttr subgroupBasis =
      ArrayAttr::get(ctx, {ArrayAttr::get(ctx, subgroupCounts),
                           ArrayAttr::get(ctx, subgroupMapping)});
  knobsEntries.emplace_back(kKnobSubgroupBasisKey, subgroupBasis);

  // Add workgroup size and subgroup size at the top level.
  SmallVector<Attribute> wgSizeKnobs = {
      IntKnobAttr::get(ctx, kKnobWgSizeXName),
      IntKnobAttr::get(ctx, kKnobWgSizeYName),
      IntKnobAttr::get(ctx, kKnobWgSizeZName)};
  knobsEntries.emplace_back(kKnobWorkgroupSizeKey,
                            ArrayAttr::get(ctx, wgSizeKnobs));
  knobsEntries.emplace_back(kKnobSubgroupSizeKey,
                            IntKnobAttr::get(ctx, kKnobSgSizeName));

  return DictionaryAttr::get(ctx, knobsEntries);
}

/// Build the TileAndFuse knobs dict for contraction-like dims.
static DictionaryAttr
buildTileAndFuseKnobsDict(MLIRContext *ctx, const RootOpLoopInfo &loopInfo,
                          const ContractionLikeDims &dims,
                          ArrayRef<Attribute> compatibleMMAs) {
  SmallVector<NamedAttribute> knobsEntries;

  // Keep the same workgroup/reduction convention as VectorDistribute:
  // untiled dims get 0, batch and outer M/N dims get unit tiles, and the
  // innermost M/N/K dims are knobbed.
  SmallVector<Attribute> workgroupEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  SmallVector<unsigned> unitWorkgroupDims;
  llvm::append_range(unitWorkgroupDims, dims.b);
  llvm::append_range(unitWorkgroupDims, dims.m);
  llvm::append_range(unitWorkgroupDims, dims.n);
  for (unsigned i : unitWorkgroupDims) {
    workgroupEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  workgroupEntries[dims.m.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.m.back()));
  workgroupEntries[dims.n.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobWgPrefix, dims.n.back()));
  knobsEntries.emplace_back(kKnobWorkgroupKey,
                            ArrayAttr::get(ctx, workgroupEntries));

  SmallVector<Attribute> reductionEntries(loopInfo.numLoops,
                                          makeIntAttr(ctx, kNoTileDimVal));
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    if (llvm::is_contained(unitWorkgroupDims, i)) {
      continue;
    }
    reductionEntries[i] = makeIntAttr(ctx, kUnitTileDimVal);
  }
  reductionEntries[dims.k.back()] =
      IntKnobAttr::get(ctx, makeVarName(kKnobRedPrefix, dims.k.back()));
  knobsEntries.emplace_back(kKnobReductionKey,
                            ArrayAttr::get(ctx, reductionEntries));

  // TileAndFuse materializes subgroup tile counts in the lowering config.
  SmallVector<Attribute> subgroupEntries(loopInfo.numLoops,
                                         makeIntAttr(ctx, kNoTileDimVal));
  subgroupEntries[dims.m.back()] =
      IntKnobAttr::get(ctx, makeSubgroupTileCountName(dims.m.back()));
  subgroupEntries[dims.n.back()] =
      IntKnobAttr::get(ctx, makeSubgroupTileCountName(dims.n.back()));
  knobsEntries.emplace_back("subgroup", ArrayAttr::get(ctx, subgroupEntries));

  knobsEntries.emplace_back(
      kKnobMmaKindKey,
      OneOfKnobAttr::get(ctx, kKnobMmaIdxName, compatibleMMAs));

  SmallVector<Attribute> subgroupCounts(loopInfo.numLoops, makeIntAttr(ctx, 1));
  subgroupCounts[dims.m.back()] = IntKnobAttr::get(ctx, kKnobSgMCntName);
  subgroupCounts[dims.n.back()] = IntKnobAttr::get(ctx, kKnobSgNCntName);
  SmallVector<Attribute> subgroupMapping;
  subgroupMapping.reserve(loopInfo.numLoops);
  for (unsigned i = 0; i < loopInfo.numLoops; ++i) {
    subgroupMapping.push_back(makeIntAttr(ctx, i));
  }
  knobsEntries.emplace_back(
      kKnobSubgroupBasisKey,
      ArrayAttr::get(ctx, {ArrayAttr::get(ctx, subgroupCounts),
                           ArrayAttr::get(ctx, subgroupMapping)}));

  SmallVector<Attribute> wgSizeKnobs = {
      IntKnobAttr::get(ctx, kKnobWgSizeXName),
      IntKnobAttr::get(ctx, kKnobWgSizeYName),
      IntKnobAttr::get(ctx, kKnobWgSizeZName)};
  knobsEntries.emplace_back(kKnobWorkgroupSizeKey,
                            ArrayAttr::get(ctx, wgSizeKnobs));
  knobsEntries.emplace_back(kKnobSubgroupSizeKey,
                            IntKnobAttr::get(ctx, kKnobSgSizeName));

  return DictionaryAttr::get(ctx, knobsEntries);
}

/// Emit VectorDistribute constraints for contraction-like dims (matmul/conv).
static LogicalResult emitVectorDistributeConstraints(
    OpBuilder &builder, linalg::LinalgOp linalgOp,
    const ContractionLikeDims &dims, IREE::GPU::TargetAttr gpuTarget,
    ArrayRef<Value> smtDimArgs, ArrayRef<Attribute> compatibleMMAs) {
  Location loc = linalgOp.getLoc();

  unsigned mDim = dims.m.back();
  std::string mName = makeVarName(kKnobWgPrefix, mDim);
  Value wgM = mkKnob(builder, loc, mName);
  assertDivisible(
      builder, loc, smtDimArgs[mDim], wgM,
      (kLoopRangePrefix + Twine(mDim) + " must be divisible by " + mName)
          .str());

  unsigned nDim = dims.n.back();
  std::string nName = makeVarName(kKnobWgPrefix, nDim);
  Value wgN = mkKnob(builder, loc, nName);
  assertDivisible(
      builder, loc, smtDimArgs[nDim], wgN,
      (kLoopRangePrefix + Twine(nDim) + " must be divisible by " + nName)
          .str());

  unsigned kDim = dims.k.back();
  std::string kName = makeVarName(kKnobRedPrefix, kDim);
  Value redK = mkKnob(builder, loc, kName);
  assertDivisible(
      builder, loc, smtDimArgs[kDim], redK,
      (kLoopRangePrefix + Twine(kDim) + " must be divisible by " + kName)
          .str());

  // Keep convolution coverage conservative for now. The sample equivalence
  // check is contraction-only, while convolution needs more pipeline-specific
  // handling for folded filter/spatial dimensions.
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return success();
  }

  int64_t subgroupSize = gpuTarget.getPreferredSubgroupSize();
  int64_t maxThreads = gpuTarget.getWgp().getMaxThreadCountPerWorkgroup();
  int64_t maxSharedMem = gpuTarget.getWgp().getMaxWorkgroupMemoryBytes();

  MMADerivedValues mma = emitMMADerivedValues(builder, loc, compatibleMMAs);
  Value sgMCnt = mkKnob(builder, loc, kKnobSgMCntName);
  Value sgNCnt = mkKnob(builder, loc, kKnobSgNCntName);
  Value sgSize = mkKnob(builder, loc, kKnobSgSizeName);
  Value subgroupSizeVal = mkIntConst(builder, loc, subgroupSize);

  assertCmp(builder, loc, smt::IntPredicate::ge, wgM, mma.mmaM,
            mName + " >= mma_m");
  assertCmp(builder, loc, smt::IntPredicate::le, wgM,
            mkIntConst(builder, loc, 512), mName + " <= 512");
  assertCmp(builder, loc, smt::IntPredicate::le, wgM, smtDimArgs[mDim],
            mName + " <= " + kLoopRangePrefix.str() + std::to_string(mDim));

  assertCmp(builder, loc, smt::IntPredicate::ge, wgN, mma.mmaN,
            nName + " >= mma_n");
  assertCmp(builder, loc, smt::IntPredicate::le, wgN,
            mkIntConst(builder, loc, 512), nName + " <= 512");
  assertCmp(builder, loc, smt::IntPredicate::le, wgN, smtDimArgs[nDim],
            nName + " <= " + kLoopRangePrefix.str() + std::to_string(nDim));

  assertCmp(builder, loc, smt::IntPredicate::ge, redK, mma.mmaK,
            kName + " >= mma_k");
  assertCmp(builder, loc, smt::IntPredicate::le, redK,
            mkIntConst(builder, loc, 512), kName + " <= 512");
  assertCmp(builder, loc, smt::IntPredicate::le, redK, smtDimArgs[kDim],
            kName + " <= " + kLoopRangePrefix.str() + std::to_string(kDim));

  assertDivisible(builder, loc, wgM, mma.mmaM,
                  mName + " must be a multiple of mma_m");
  assertDivisible(builder, loc, wgN, mma.mmaN,
                  nName + " must be a multiple of mma_n");
  assertDivisible(builder, loc, redK, mma.mmaK,
                  kName + " must be a multiple of mma_k");

  assertBounds(builder, loc, sgMCnt, kKnobSgMCntName, 1, 32);
  assertBounds(builder, loc, sgNCnt, kKnobSgNCntName, 1, 32);

  Value sgMTimesIntrinsic =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, mma.mmaM});
  assertDivisible(builder, loc, wgM, sgMTimesIntrinsic,
                  mName + " must be divisible by sg_m_cnt * mma_m");
  Value sgNTimesIntrinsic =
      smt::IntMulOp::create(builder, loc, ValueRange{sgNCnt, mma.mmaN});
  assertDivisible(builder, loc, wgN, sgNTimesIntrinsic,
                  nName + " must be divisible by sg_n_cnt * mma_n");

  Value totalThreads = smt::IntMulOp::create(
      builder, loc, ValueRange{sgMCnt, sgNCnt, subgroupSizeVal});
  assertCmp(builder, loc, smt::IntPredicate::le, totalThreads,
            mkIntConst(builder, loc, maxThreads),
            "total threads <= max_threads");

  Value kTimesN = smt::IntMulOp::create(builder, loc, ValueRange{redK, wgN});
  assertDivisible(builder, loc, kTimesN, totalThreads,
                  kName + " * " + nName +
                      " must be divisible by total threads");
  Value kTimesM = smt::IntMulOp::create(builder, loc, ValueRange{redK, wgM});
  assertDivisible(builder, loc, kTimesM, totalThreads,
                  kName + " * " + mName +
                      " must be divisible by total threads");

  auto lhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto rhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(1)->get().getType());
  int64_t lhsBytes = lhsType.getElementTypeBitWidth() / 8;
  int64_t rhsBytes = rhsType.getElementTypeBitWidth() / 8;
  Value lhsShared = smt::IntMulOp::create(
      builder, loc, ValueRange{mkIntConst(builder, loc, lhsBytes), wgM, redK});
  Value rhsShared = smt::IntMulOp::create(
      builder, loc, ValueRange{mkIntConst(builder, loc, rhsBytes), wgN, redK});
  Value totalShared =
      smt::IntAddOp::create(builder, loc, ValueRange{lhsShared, rhsShared});
  assertCmp(builder, loc, smt::IntPredicate::le, totalShared,
            mkIntConst(builder, loc, maxSharedMem),
            "shared memory must fit in workgroup memory");

  Value wgSizeX = mkKnob(builder, loc, kKnobWgSizeXName);
  Value wgSizeY = mkKnob(builder, loc, kKnobWgSizeYName);
  Value wgSizeZ = mkKnob(builder, loc, kKnobWgSizeZName);
  Value expectedWgSizeX =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt, sgSize});
  Value wgSizeXEq = smt::EqOp::create(builder, loc, wgSizeX, expectedWgSizeX);
  AssertOp::create(builder, loc, wgSizeXEq,
                   "wg_size_x == sg_m_cnt * sg_n_cnt * sg_size");
  Value one = mkIntConst(builder, loc, 1);
  AssertOp::create(builder, loc, smt::EqOp::create(builder, loc, wgSizeY, one),
                   "wg_size_y == 1");
  AssertOp::create(builder, loc, smt::EqOp::create(builder, loc, wgSizeZ, one),
                   "wg_size_z == 1");
  AssertOp::create(builder, loc,
                   smt::EqOp::create(builder, loc, sgSize, subgroupSizeVal),
                   "sg_size == preferred_subgroup_size");

  // Load alignment constraint from the original tuner-side model.
  assertDivisible(builder, loc, redK, mma.mmaM,
                  kName + " must be a multiple of mma_m");

  Value sgSizeTimesN =
      smt::IntMulOp::create(builder, loc, ValueRange{sgSize, sgNCnt});
  Value leN = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::le,
                                    sgSizeTimesN, wgN);
  Value leM = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::le,
                                    sgSizeTimesN, wgM);
  AssertOp::create(builder, loc,
                   smt::OrOp::create(builder, loc, ValueRange{leN, leM}),
                   "sg_size * sg_n_cnt must not exceed wg_m or wg_n");

  return success();
}

/// Emit TileAndFuse constraints for contraction-like dims (matmul/conv).
static LogicalResult emitTileAndFuseConstraints(
    OpBuilder &builder, linalg::LinalgOp linalgOp,
    const ContractionLikeDims &dims, IREE::GPU::TargetAttr gpuTarget,
    ArrayRef<Value> smtDimArgs, ArrayRef<Attribute> compatibleMMAs) {
  Location loc = linalgOp.getLoc();

  unsigned mDim = dims.m.back();
  unsigned nDim = dims.n.back();
  unsigned kDim = dims.k.back();
  std::string mName = makeVarName(kKnobWgPrefix, mDim);
  std::string nName = makeVarName(kKnobWgPrefix, nDim);
  std::string kName = makeVarName(kKnobRedPrefix, kDim);
  Value wgM = mkKnob(builder, loc, mName);
  Value wgN = mkKnob(builder, loc, nName);
  Value redKTileCount = mkKnob(builder, loc, kName);

  assertDivisible(
      builder, loc, smtDimArgs[mDim], wgM,
      (kLoopRangePrefix + Twine(mDim) + " must be divisible by " + mName)
          .str());
  assertDivisible(
      builder, loc, smtDimArgs[nDim], wgN,
      (kLoopRangePrefix + Twine(nDim) + " must be divisible by " + nName)
          .str());

  if (!linalg::isaContractionOpInterface(linalgOp)) {
    assertDivisible(
        builder, loc, smtDimArgs[kDim], redKTileCount,
        (kLoopRangePrefix + Twine(kDim) + " must be divisible by " + kName)
            .str());
    return success();
  }

  int64_t subgroupSize = gpuTarget.getPreferredSubgroupSize();
  int64_t maxThreads = gpuTarget.getWgp().getMaxThreadCountPerWorkgroup();
  int64_t maxSharedMem = gpuTarget.getWgp().getMaxWorkgroupMemoryBytes();

  MMADerivedValues mma = emitMMADerivedValues(builder, loc, compatibleMMAs);
  Value redK =
      smt::IntMulOp::create(builder, loc, ValueRange{redKTileCount, mma.mmaK});
  assertDivisible(builder, loc, smtDimArgs[kDim], redK,
                  (kLoopRangePrefix + Twine(kDim) + " must be divisible by " +
                   kName + " * mma_k")
                      .str());

  Value sgMCnt = mkKnob(builder, loc, kKnobSgMCntName);
  Value sgNCnt = mkKnob(builder, loc, kKnobSgNCntName);
  Value sgMTcnt = mkKnob(builder, loc, makeSubgroupTileCountName(mDim));
  Value sgNTcnt = mkKnob(builder, loc, makeSubgroupTileCountName(nDim));
  Value sgSize = mkKnob(builder, loc, kKnobSgSizeName);
  Value subgroupSizeVal = mkIntConst(builder, loc, subgroupSize);

  assertCmp(builder, loc, smt::IntPredicate::ge, wgM, mma.mmaM,
            mName + " >= mma_m");
  assertCmp(builder, loc, smt::IntPredicate::le, wgM,
            mkIntConst(builder, loc, 512), mName + " <= 512");
  assertCmp(builder, loc, smt::IntPredicate::le, wgM, smtDimArgs[mDim],
            mName + " <= " + kLoopRangePrefix.str() + std::to_string(mDim));

  assertCmp(builder, loc, smt::IntPredicate::ge, wgN, mma.mmaN,
            nName + " >= mma_n");
  assertCmp(builder, loc, smt::IntPredicate::le, wgN,
            mkIntConst(builder, loc, 512), nName + " <= 512");
  assertCmp(builder, loc, smt::IntPredicate::le, wgN, smtDimArgs[nDim],
            nName + " <= " + kLoopRangePrefix.str() + std::to_string(nDim));

  assertCmp(builder, loc, smt::IntPredicate::ge, redKTileCount,
            mkIntConst(builder, loc, 1), kName + " >= 1");
  assertCmp(builder, loc, smt::IntPredicate::le, redK,
            mkIntConst(builder, loc, 512), kName + " * mma_k <= 512");
  assertCmp(builder, loc, smt::IntPredicate::le, redK, smtDimArgs[kDim],
            kName + " * mma_k <= " + kLoopRangePrefix.str() +
                std::to_string(kDim));

  assertDivisible(builder, loc, wgM, mma.mmaM,
                  mName + " must be a multiple of mma_m");
  assertDivisible(builder, loc, wgN, mma.mmaN,
                  nName + " must be a multiple of mma_n");

  assertBounds(builder, loc, sgMCnt, kKnobSgMCntName, 1, 32);
  assertBounds(builder, loc, sgNCnt, kKnobSgNCntName, 1, 32);
  assertBounds(builder, loc, sgMTcnt, makeSubgroupTileCountName(mDim), 1, 32);
  assertBounds(builder, loc, sgNTcnt, makeSubgroupTileCountName(nDim), 1, 32);

  Value expectedM = smt::IntMulOp::create(
      builder, loc, ValueRange{sgMCnt, sgMTcnt, mma.mmaM});
  AssertOp::create(builder, loc,
                   smt::EqOp::create(builder, loc, wgM, expectedM),
                   mName + " == sg_m_cnt * sg_m_tcnt * mma_m");
  Value expectedN = smt::IntMulOp::create(
      builder, loc, ValueRange{sgNCnt, sgNTcnt, mma.mmaN});
  AssertOp::create(builder, loc,
                   smt::EqOp::create(builder, loc, wgN, expectedN),
                   nName + " == sg_n_cnt * sg_n_tcnt * mma_n");

  Value wgSizeX = mkKnob(builder, loc, kKnobWgSizeXName);
  Value wgSizeY = mkKnob(builder, loc, kKnobWgSizeYName);
  Value wgSizeZ = mkKnob(builder, loc, kKnobWgSizeZName);
  Value totalThreads =
      smt::IntMulOp::create(builder, loc, ValueRange{sgMCnt, sgNCnt, sgSize});
  AssertOp::create(builder, loc,
                   smt::EqOp::create(builder, loc, wgSizeX, totalThreads),
                   "wg_size_x == sg_m_cnt * sg_n_cnt * sg_size");
  assertCmp(builder, loc, smt::IntPredicate::le, totalThreads,
            mkIntConst(builder, loc, maxThreads),
            "total threads <= max_threads");

  Value one = mkIntConst(builder, loc, 1);
  AssertOp::create(builder, loc, smt::EqOp::create(builder, loc, wgSizeY, one),
                   "wg_size_y == 1");
  AssertOp::create(builder, loc, smt::EqOp::create(builder, loc, wgSizeZ, one),
                   "wg_size_z == 1");
  AssertOp::create(builder, loc,
                   smt::EqOp::create(builder, loc, sgSize, subgroupSizeVal),
                   "sg_size == preferred_subgroup_size");

  auto lhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto rhsType =
      cast<ShapedType>(linalgOp.getDpsInputOperand(1)->get().getType());
  int64_t lhsBytes = lhsType.getElementTypeBitWidth() / 8;
  int64_t rhsBytes = rhsType.getElementTypeBitWidth() / 8;
  Value lhsShared = smt::IntMulOp::create(
      builder, loc, ValueRange{mkIntConst(builder, loc, lhsBytes), wgM, redK});
  Value rhsShared = smt::IntMulOp::create(
      builder, loc, ValueRange{mkIntConst(builder, loc, rhsBytes), wgN, redK});
  Value totalShared =
      smt::IntAddOp::create(builder, loc, ValueRange{lhsShared, rhsShared});
  assertCmp(builder, loc, smt::IntPredicate::le, totalShared,
            mkIntConst(builder, loc, maxSharedMem),
            "shared memory must fit in workgroup memory");

  return success();
}

/// Emit constraints for a single root op under the VectorDistribute pipeline.
/// Only supports linalg contraction and convolution today.
static LogicalResult
emitVectorDistributeConstraintsForOp(Operation *rootOp, RootOpAttr rootOpAttr) {
  // Gate on contraction-like linalg ops.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp || (!linalg::isaContractionOpInterface(linalgOp) &&
                    !linalg::isaConvolutionOpInterface(linalgOp))) {
    return success();
  }

  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(rootOp);
  if (!gpuTarget) {
    return success();
  }

  std::optional<RootOpLoopInfo> loopInfo = getRootOpLoopInfo(rootOp);
  if (!loopInfo) {
    return success();
  }

  FailureOr<ContractionLikeDims> dims = inferContractionLikeDims(linalgOp);
  if (failed(dims)) {
    return success();
  }

  SmallVector<Attribute> compatibleMMAs =
      getCompatibleMMAAttrs(linalgOp, gpuTarget, *loopInfo, *dims);
  if (compatibleMMAs.empty()) {
    return success();
  }

  MLIRContext *ctx = rootOp->getContext();
  OpBuilder builder(ctx);
  DictionaryAttr knobs =
      buildVectorDistributeKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
  auto pipelineAttr = IREE::GPU::PipelineAttr::get(
      ctx, IREE::GPU::LoweringPipeline::VectorDistribute);
  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               loopInfo->numLoops, loopInfo->indexingMaps);

  return emitVectorDistributeConstraints(builder, linalgOp, *dims, gpuTarget,
                                         shell.smtDimArgs, compatibleMMAs);
}

/// Emit constraints for a single root op under the TileAndFuse pipeline.
/// Only supports linalg contraction and convolution today.
static LogicalResult emitTileAndFuseConstraintsForOp(Operation *rootOp,
                                                     RootOpAttr rootOpAttr) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp || (!linalg::isaContractionOpInterface(linalgOp) &&
                    !linalg::isaConvolutionOpInterface(linalgOp))) {
    return success();
  }

  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(rootOp);
  if (!gpuTarget) {
    return success();
  }

  std::optional<RootOpLoopInfo> loopInfo = getRootOpLoopInfo(rootOp);
  if (!loopInfo) {
    return success();
  }

  FailureOr<ContractionLikeDims> dims = inferContractionLikeDims(linalgOp);
  if (failed(dims)) {
    return success();
  }

  SmallVector<Attribute> compatibleMMAs =
      getCompatibleMMAAttrs(linalgOp, gpuTarget, *loopInfo, *dims);
  if (compatibleMMAs.empty()) {
    return success();
  }

  MLIRContext *ctx = rootOp->getContext();
  OpBuilder builder(ctx);
  DictionaryAttr knobs =
      buildTileAndFuseKnobsDict(ctx, *loopInfo, *dims, compatibleMMAs);
  auto pipelineAttr = IREE::GPU::PipelineAttr::get(
      ctx, IREE::GPU::LoweringPipeline::TileAndFuse);
  ConstraintsOpShell shell =
      createConstraintsOpShell(builder, rootOp, rootOpAttr, pipelineAttr, knobs,
                               loopInfo->numLoops, loopInfo->indexingMaps);

  return emitTileAndFuseConstraints(builder, linalgOp, *dims, gpuTarget,
                                    shell.smtDimArgs, compatibleMMAs);
}

/// Multiple root ops may be present in a set, e.g. <set = 0>:
/// [linalg.fill, linalg.matmul]. This function will choose the matmul op
/// over the fill op.
static Operation *getTunableOp(ArrayRef<Operation *> rootOps) {
  if (rootOps.empty()) {
    return nullptr;
  }
  for (Operation *rootOp : rootOps) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
    if (linalgOp && (linalg::isaContractionOpInterface(linalgOp) ||
                     linalg::isaConvolutionOpInterface(linalgOp))) {
      return rootOp;
    }
    if (auto attnOp = dyn_cast<IREE::LinalgExt::OnlineAttentionOp>(rootOp)) {
      return rootOp;
    }
  }
  return nullptr;
}

LogicalResult emitLLVMGPUConstraints(Attribute attr,
                                     ArrayRef<Operation *> rootOps) {
  Operation *tunableOp = getTunableOp(rootOps);
  if (!tunableOp) {
    return success();
  }
  RootOpAttr opAttr = getRootOpInfo(tunableOp);
  if (!opAttr) {
    return success();
  }

  auto gpuPipelineAttr = cast<IREE::GPU::PipelineAttr>(attr);

  if (gpuPipelineAttr.getValue() ==
      IREE::GPU::LoweringPipeline::VectorDistribute) {
    return emitVectorDistributeConstraintsForOp(tunableOp, opAttr);
  }
  if (gpuPipelineAttr.getValue() == IREE::GPU::LoweringPipeline::TileAndFuse) {
    return emitTileAndFuseConstraintsForOp(tunableOp, opAttr);
  }

  return success();
}

} // namespace mlir::iree_compiler

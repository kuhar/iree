// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PadToAllocation.cpp ----- Pass to increase cache bandwidth ---------===//
//
// Inserts tensor padding to pad the underlying allocations and increase the
// L1 cache bandwidth.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_PADTOALLOCATIONPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {
struct PadMmt final : OpRewritePattern<linalg::MatmulTransposeBOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulTransposeBOp op,
                                PatternRewriter &rewriter) const override {
    llvm::outs() << "JAKUB: match\n";
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    if (lhs.getDefiningOp<tensor::ExtractSliceOp>() ||
        rhs.getDefiningOp<tensor::ExtractSliceOp>())
      return failure();

    llvm::outs() << "JAKUB: match\n";

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType)
      return failure();

    if (lhsType.isDynamicDim(0) || lhsType.isDynamicDim(1))
      return failure();
    if (rhsType.isDynamicDim(0) || rhsType.isDynamicDim(1))
      return failure();

    int64_t mDim = lhsType.getDimSize(0);
    int64_t nDim = rhsType.getDimSize(0);
    int64_t kDim = rhsType.getDimSize(1);
    int64_t elementTypeBytes = lhsType.getElementTypeBitWidth() / 8;

    int64_t kSize = kDim * elementTypeBytes;
    if (kSize % (128 * 4) != 0)
      return failure();

    int64_t newKDim = kDim + 128 / elementTypeBytes;
    auto newLhsShape = llvm::to_vector<2>(lhsType.getShape());
    newLhsShape.back() = newKDim;
    auto newRhsShape = llvm::to_vector<2>(rhsType.getShape());
    newRhsShape.back() = newKDim;

    Location loc = op.getLoc();
    auto paddedLhs = rewriter.create<tensor::EmptyOp>(
        loc, newLhsShape, lhsType.getElementType(), lhsType.getEncoding());
    auto paddedRhs = rewriter.create<tensor::EmptyOp>(
        loc, newRhsShape, rhsType.getElementType(), rhsType.getEncoding());

    Attribute zeroIdx = rewriter.getI64IntegerAttr(0);
    Attribute oneIdx = rewriter.getI64IntegerAttr(1);
    OpFoldResult offsets[2] = {zeroIdx, zeroIdx};
    OpFoldResult lhsSizes[2] = {rewriter.getI64IntegerAttr(mDim),
                                rewriter.getI64IntegerAttr(kDim)};
    OpFoldResult rhsSizes[2] = {rewriter.getI64IntegerAttr(nDim),
                                rewriter.getI64IntegerAttr(kDim)};
    OpFoldResult strides[2] = {oneIdx, oneIdx};
    auto insLhs = rewriter.create<tensor::InsertSliceOp>(
        loc, lhs, paddedLhs, offsets, lhsSizes, strides);
    auto insRhs = rewriter.create<tensor::InsertSliceOp>(
        loc, rhs, paddedRhs, offsets, rhsSizes, strides);

    auto extLhs = rewriter.create<tensor::ExtractSliceOp>(loc, insLhs, offsets,
                                                          lhsSizes, strides);
    extLhs->setAttr("__no_fold", rewriter.getUnitAttr());
    auto extRhs = rewriter.create<tensor::ExtractSliceOp>(loc, insRhs, offsets,
                                                          rhsSizes, strides);
    extRhs->setAttr("__no_fold", rewriter.getUnitAttr());
    rewriter.modifyOpInPlace(op, [&extLhs, &extRhs, &op] {
      op.setOperand(0, extLhs);
      op.setOperand(1, extRhs);
    });
    llvm::outs() << "JAKUB: done\n";

    return success();
  }
};

struct PadContract final : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(op))
      return failure();

    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    if (lhs.getDefiningOp<tensor::ExtractSliceOp>() ||
        rhs.getDefiningOp<tensor::ExtractSliceOp>())
      return failure();

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType)
      return failure();
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
      return failure();

    if (lhsType.isDynamicDim(0) || lhsType.isDynamicDim(1))
      return failure();
    if (rhsType.isDynamicDim(0) || rhsType.isDynamicDim(1))
      return failure();

    SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
    FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
        mlir::linalg::inferContractionDims(op);
    assert(succeeded(contractionDims) && "Could not infer contraction dims");
    if (contractionDims->k.size() != 1 || contractionDims->m.size() != 1 ||
        contractionDims->n.size() != 1 || !contractionDims->batch.empty())
      return failure();

    llvm::outs() << "JAKUB: match 5\n";
    int64_t nDim = contractionDims->n.back();
    int64_t kDim = contractionDims->k.back();

    // Infer if lhs or rhs is transposed to help generate better schedule.
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    bool transposedLhs =
        kDim !=
        llvm::cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
    bool transposedRhs =
        nDim !=
        llvm::cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();
    if (transposedLhs || !transposedRhs)
      return failure();

    int64_t mSize = lhsType.getDimSize(0);
    int64_t nSize = rhsType.getDimSize(0);
    int64_t kSize = rhsType.getDimSize(1);
    llvm::outs() << "JAKUB: MNK: " << mSize << "x" << nSize << "x" << kSize << "\n";
    int64_t elementTypeBytes = lhsType.getElementTypeBitWidth() / 8;

    if (mSize < 256 || nSize < 256 || kSize < 256)
      return failure();

    int64_t kSizeB = kSize * elementTypeBytes;
    if (kSize % (128 * 4) != 0)
      return failure();

    int64_t newKSize = kSizeB + 128 / elementTypeBytes;
    auto newLhsShape = llvm::to_vector<2>(lhsType.getShape());
    newLhsShape.back() = newKSize;
    auto newRhsShape = llvm::to_vector<2>(rhsType.getShape());
    newRhsShape.back() = newKSize;

    Location loc = op.getLoc();
    auto paddedLhs = rewriter.create<tensor::EmptyOp>(
        loc, newLhsShape, lhsType.getElementType(), lhsType.getEncoding());
    auto paddedRhs = rewriter.create<tensor::EmptyOp>(
        loc, newRhsShape, rhsType.getElementType(), rhsType.getEncoding());

    Attribute zeroIdx = rewriter.getI64IntegerAttr(0);
    Attribute oneIdx = rewriter.getI64IntegerAttr(1);
    OpFoldResult offsets[2] = {zeroIdx, zeroIdx};
    OpFoldResult lhsSizes[2] = {rewriter.getI64IntegerAttr(mSize),
                                rewriter.getI64IntegerAttr(kSize)};
    OpFoldResult rhsSizes[2] = {rewriter.getI64IntegerAttr(nSize),
                                rewriter.getI64IntegerAttr(kSize)};
    OpFoldResult strides[2] = {oneIdx, oneIdx};
    auto insLhs = rewriter.create<tensor::InsertSliceOp>(
        loc, lhs, paddedLhs, offsets, lhsSizes, strides);
    auto insRhs = rewriter.create<tensor::InsertSliceOp>(
        loc, rhs, paddedRhs, offsets, rhsSizes, strides);

    auto extLhs = rewriter.create<tensor::ExtractSliceOp>(loc, insLhs, offsets,
                                                          lhsSizes, strides);
    extLhs->setAttr("__no_fold", rewriter.getUnitAttr());
    auto extRhs = rewriter.create<tensor::ExtractSliceOp>(loc, insRhs, offsets,
                                                          rhsSizes, strides);
    extRhs->setAttr("__no_fold", rewriter.getUnitAttr());
    rewriter.modifyOpInPlace(op, [&extLhs, &extRhs, &op] {
      op.setOperand(0, extLhs);
      op.setOperand(1, extRhs);
    });
    llvm::outs() << "JAKUB: done\n";

    return success();
  }
};

struct PaddToAllocationPass final
    : impl::PadToAllocationPassBase<PaddToAllocationPass> {
  using impl::PadToAllocationPassBase<
      PaddToAllocationPass>::PadToAllocationPassBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<PadMmt, PadContract>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow

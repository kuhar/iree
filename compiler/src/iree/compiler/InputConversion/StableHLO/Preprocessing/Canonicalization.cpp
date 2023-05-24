// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements optional canonicalization patterns for StableHLO ops.

#include <cassert>
#include <functional>
#include <numeric>

#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h"
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Rewriters.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_STABLEHLOCANONICALIZE
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h.inc"

namespace {

// This is an upper limit on how many elements canonicalization patterns are
// allowed to materialize as new constants.
constexpr int64_t kFoldOpEltLimit = 65536;

static bool isIotaRange(ElementsAttr attr) {
  auto elems = attr.tryGetValues<APInt>();
  if (!elems) return false;

  for (auto [idx, value] : llvm::enumerate(*elems)) {
    if (idx != value) {
      return false;
    }
  }

  return true;
}

/// Matches when either of the submatchers match.
template <typename MatcherA, typename MatcherB>
struct m_AnyOf {
  m_AnyOf(MatcherA a, MatcherB b) : matcherA(a), matcherB(b) {}

  bool match(Operation *op) { return matcherA.match(op) || matcherB.match(op); }

  MatcherA matcherA;
  MatcherB matcherB;
};

template <typename MatcherA, typename MatcherB>
m_AnyOf(MatcherA, MatcherB) -> m_AnyOf<MatcherA, MatcherB>;

/// Binary constant folder that used a generic folder function to handle both
/// ints and floats.
template <typename Fn>
static TypedAttr foldBinaryOpIntOrFloat(TypedAttr lhs, TypedAttr rhs,
                                        Fn &&folder) {
  Attribute operands[2] = {lhs, rhs};
  Type elemTy = getElementTypeOrSelf(cast<TypedAttr>(lhs).getType());

  if (isa<IntegerType>(elemTy)) {
    if (Attribute res = constFoldBinaryOp<IntegerAttr>(
            operands, [&folder](const APInt &lhs, const APInt &rhs) {
              return folder(lhs, rhs);
            })) {
      return cast<TypedAttr>(res);
    }
    return nullptr;
  }

  if (isa<FloatType>(elemTy)) {
    if (Attribute res = constFoldBinaryOp<FloatAttr>(
            operands, [&folder](const APFloat &lhs, const APFloat &rhs) {
              return folder(lhs, rhs);
            })) {
      return cast<TypedAttr>(res);
    }
    return nullptr;
  }

  return nullptr;
}

struct AddOpCanon final : OpRewritePattern<mlir::stablehlo::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type) return failure();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (matchPattern(lhs, m_Zero())) {
      rewriter.replaceOp(op, rhs);
      return success();
    }

    if (matchPattern(rhs, m_AnyOf(m_Zero(), m_NegZeroFloat()))) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    TypedAttr lhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(rhs, m_Constant(&rhsAttr));

    // The canonical form has the constant operand as the RHS.
    if (isa<IntegerType>(type.getElementType()) && lhsAttr && !rhsAttr) {
      rewriter.updateRootInPlace(op, [op, lhs, rhs] {
        op->setOperands(ValueRange{rhs, lhs});
      });
      return success();
    }

    if (lhsAttr && rhsAttr) {
      if (TypedAttr res =
              foldBinaryOpIntOrFloat(lhsAttr, rhsAttr, std::plus<>{})) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
        return success();
      }
    }

    return failure();
  }
};

struct SubtractOpCanon final : OpRewritePattern<mlir::stablehlo::SubtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type) return failure();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (isa<IntegerType>(type.getElementType()) && lhs == rhs) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, rewriter.getZeroAttr(op.getType()));
      return success();
    }

    // Subtraction of 0.
    if (matchPattern(rhs, m_AnyOf(m_Zero(), m_PosZeroFloat()))) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    TypedAttr lhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(rhs, m_Constant(&rhsAttr));

    if (lhsAttr && rhsAttr) {
      if (TypedAttr res =
              foldBinaryOpIntOrFloat(lhsAttr, rhsAttr, std::minus<>{})) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
        return success();
      }
    }

    return failure();
  }
};

struct MulOpCanon final : OpRewritePattern<mlir::stablehlo::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type) return failure();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Multiplication by 0. This fold is not trivial for floats in presence of
    // NaN values.
    if (matchPattern(lhs, m_Zero())) {
      rewriter.replaceOp(op, lhs);
      return success();
    }
    if (matchPattern(rhs, m_Zero())) {
      rewriter.replaceOp(op, rhs);
      return success();
    }

    // Multiplication by 1.
    if (matchPattern(rhs, m_One())) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    TypedAttr lhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(rhs, m_Constant(&rhsAttr));

    // The canonical form has the constant operand as the RHS.
    if (isa<IntegerType>(type.getElementType()) && lhsAttr && !rhsAttr) {
      rewriter.updateRootInPlace(op, [op, lhs, rhs] {
        op->setOperands(ValueRange{rhs, lhs});
      });
      return success();
    }

    if (lhsAttr && rhsAttr) {
      if (TypedAttr res =
              foldBinaryOpIntOrFloat(lhsAttr, rhsAttr, std::multiplies<>{})) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
        return success();
      }
    }

    return failure();
  }
};

static mlir::stablehlo::ComparisonDirection invertDirection(
    mlir::stablehlo::ComparisonDirection direction) {
  using mlir::stablehlo::ComparisonDirection;

  switch (direction) {
    case ComparisonDirection::EQ:
      return ComparisonDirection::EQ;
    case ComparisonDirection::GE:
      return ComparisonDirection::LE;
    case ComparisonDirection::LE:
      return ComparisonDirection::GE;
    case ComparisonDirection::GT:
      return ComparisonDirection::LT;
    case ComparisonDirection::LT:
      return ComparisonDirection::GT;
    case ComparisonDirection::NE:
      return ComparisonDirection::NE;
  }

  llvm_unreachable("Unhandled case");
}

static APInt calculateComp(mlir::stablehlo::ComparisonType kind,
                           mlir::stablehlo::ComparisonDirection direction,
                           const APInt &lhs, const APInt &rhs) {
  using mlir::stablehlo::ComparisonDirection;
  using mlir::stablehlo::ComparisonType;
  assert(llvm::is_contained({ComparisonType::SIGNED, ComparisonType::UNSIGNED},
                            kind) &&
         "Not an integer comparison");

  auto asBit = [](bool value) {
    return value ? APInt::getAllOnes(1) : APInt::getZero(1);
  };

  // Signed comparison.
  if (kind == ComparisonType::SIGNED) {
    switch (direction) {
      case ComparisonDirection::EQ:
        return asBit(lhs == rhs);
      case ComparisonDirection::GE:
        return asBit(lhs.sge(rhs));
      case ComparisonDirection::GT:
        return asBit(lhs.sgt(rhs));
      case ComparisonDirection::LE:
        return asBit(lhs.sle(rhs));
      case ComparisonDirection::LT:
        return asBit(lhs.slt(rhs));
      case ComparisonDirection::NE:
        return asBit(lhs != rhs);
    }
  }

  // Unsigned comparison.
  switch (direction) {
    case ComparisonDirection::EQ:
      return asBit(lhs == rhs);
    case ComparisonDirection::GE:
      return asBit(lhs.uge(rhs));
    case ComparisonDirection::GT:
      return asBit(lhs.ugt(rhs));
    case ComparisonDirection::LE:
      return asBit(lhs.ule(rhs));
    case ComparisonDirection::LT:
      return asBit(lhs.ult(rhs));
    case ComparisonDirection::NE:
      return asBit(lhs != rhs);
  }

  llvm_unreachable("Unhandled case");
}

struct CompareOpCanon final : OpRewritePattern<mlir::stablehlo::CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type) return failure();

    // Bail out on non-integer comparison.
    // TODO: Support more comparison types.
    using mlir::stablehlo::ComparisonType;
    std::optional<ComparisonType> compType = op.getCompareType();
    if (!compType ||
        !llvm::is_contained({ComparisonType::SIGNED, ComparisonType::UNSIGNED},
                            *compType)) {
      return failure();
    }

    using mlir::stablehlo::ComparisonDirection;
    ComparisonDirection direction = op.getComparisonDirection();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (lhs == rhs) {
      switch (direction) {
        case ComparisonDirection::EQ:
        case ComparisonDirection::GE:
        case ComparisonDirection::LE: {
          rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
              op, SplatElementsAttr::get(type, rewriter.getBoolAttr(true)));
          return success();
        }
        case ComparisonDirection::GT:
        case ComparisonDirection::LT:
        case ComparisonDirection::NE: {
          rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
              op, rewriter.getZeroAttr(type));
          return success();
        }
      }
      llvm_unreachable("Unhandled case");
    }

    TypedAttr lhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(rhs, m_Constant(&rhsAttr));

    // The canonical form has the constant operand as the RHS.
    if (isa<IntegerType>(type.getElementType()) && lhsAttr && !rhsAttr) {
      rewriter.updateRootInPlace(op, [&op, direction, lhs, rhs] {
        op.setComparisonDirection(invertDirection(direction));
        op->setOperands(ValueRange{rhs, lhs});
      });
      return success();
    }

    if (lhsAttr && rhsAttr) {
      if (Attribute res = constFoldBinaryOp<IntegerAttr>(
              ArrayRef<Attribute>({lhsAttr, rhsAttr}), op.getType(),
              [direction, kind = *compType](const APInt &a, const APInt &b) {
                return calculateComp(kind, direction, a, b);
              })) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
        return success();
      }
    }

    return failure();
  }
};

struct BroadcastInDimOpCanon final
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type) return failure();

    Value operand = op.getOperand();
    auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandTy) return failure();

    // Fold when broadcast is a noop.
    DenseIntElementsAttr dims = op.getBroadcastDimensions();
    bool isDimsIota = isIotaRange(dims);
    if (type == operandTy && isDimsIota) {
      rewriter.replaceOp(op, operand);
      return success();
    }

    // Handle splat broadcasts.
    if (SplatElementsAttr cstAttr;
        matchPattern(operand, m_Constant(&cstAttr))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, SplatElementsAttr::get(op.getType(),
                                     cstAttr.getSplatValue<Attribute>()));
      return success();
    }

    auto bsDimIndices = dims.getValues<int64_t>();
    if (operandTy.hasStaticShape() && type.hasStaticShape() &&
        type.getNumElements() == operandTy.getNumElements()) {
      // BroadcastInDim equivalent to reshape.
      if (isDimsIota) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, type,
                                                                operand);
        return success();
      }
      // BroadcastInDim equivalent to transpose.
      if (type.getRank() == operandTy.getRank()) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::TransposeOp>(
            op, type, operand, dims);
        return success();
      }
    }

    // Eliminate redundant nested BroadcastInDim.
    if (auto broadcastInDimOp =
            operand.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto newIndices = cast<DenseIntElementsAttr>(
          broadcastInDimOp.getBroadcastDimensions().mapValues(
              dims.getElementType(), [&bsDimIndices](const APInt &dim) {
                return APInt(dim.getBitWidth(),
                             bsDimIndices[dim.getSExtValue()], true);
              }));
      rewriter.replaceOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
          op, type, broadcastInDimOp.getOperand(), newIndices);
      return success();
    }

    return failure();
  }
};

struct ConcatenateOpCanon final
    : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type || !type.hasStaticShape()) return failure();

    size_t numElems = type.getNumElements();
    if (numElems > kFoldOpEltLimit) return failure();

    // Fold concatenate when all inputs are constants.
    OperandRange inputs = op.getInputs();
    SmallVector<DenseElementsAttr> constants(inputs.size());
    for (auto [input, constant] : llvm::zip_equal(inputs, constants)) {
      if (!matchPattern(input, m_Constant(&constant))) {
        return failure();
      }
    }

    uint64_t axis = op.getDimension();
    ArrayRef<int64_t> shape = type.getShape();
    int64_t topSize = std::accumulate(shape.begin(), shape.begin() + axis,
                                      int64_t{1}, std::multiplies<>{});

    SmallVector<Attribute> newElems;
    newElems.reserve(numElems);

    for (int64_t i = 0; i != topSize; ++i) {
      for (ElementsAttr attr : constants) {
        size_t bottomSize = attr.getNumElements() / topSize;
        auto begin = attr.value_begin<Attribute>() + (i * bottomSize);
        newElems.append(begin, begin + bottomSize);
      }
    }

    assert(newElems.size() == numElems);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), newElems));
    return success();
  }
};

struct ConvertOpCanon final : OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this convert is a noop.
    if (op.getOperand().getType() != op.getType()) return failure();

    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

struct DynamicReshapeOpCanon final
    : OpRewritePattern<mlir::stablehlo::DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // This is a noop when the output type is already a static shape.
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type || !type.hasStaticShape()) return failure();

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, type,
                                                            op.getOperand());
    return success();
  }
};

struct GetTupleElementOpCanon final
    : OpRewritePattern<mlir::stablehlo::GetTupleElementOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GetTupleElementOp op,
                                PatternRewriter &rewriter) const override {
    auto constructor =
        op.getOperand().getDefiningOp<mlir::stablehlo::TupleOp>();
    if (!constructor) return failure();

    Value result = constructor.getOperand(op.getIndex());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct RealOpCanon final : OpRewritePattern<mlir::stablehlo::RealOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::RealOp op,
                                PatternRewriter &rewriter) const override {
    auto complex = op.getOperand().getDefiningOp<mlir::stablehlo::ComplexOp>();
    if (!complex) return failure();

    rewriter.replaceOp(op, complex.getLhs());
    return success();
  }
};

struct ImagOpCanon final : OpRewritePattern<mlir::stablehlo::ImagOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ImagOp op,
                                PatternRewriter &rewriter) const override {
    auto complex = op.getOperand().getDefiningOp<mlir::stablehlo::ComplexOp>();
    if (!complex) return failure();

    rewriter.replaceOp(op, complex.getRhs());
    return success();
  }
};

struct GetDimensionSizeOpCanon final
    : OpRewritePattern<mlir::stablehlo::GetDimensionSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GetDimensionSizeOp op,
                                PatternRewriter &rewriter) const override {
    // Fold get_dimension_size when the queried dim is statically known.
    auto tensorTy = dyn_cast<RankedTensorType>(op.getOperand().getType());
    if (!tensorTy) return failure();

    int64_t dimSize = tensorTy.getDimSize(op.getDimension());
    if (dimSize < 0) return failure();

    auto elemTy = cast<IntegerType>(op.getType().getElementType());
    IntegerAttr elemVal = rewriter.getIntegerAttr(elemTy, dimSize);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), elemVal));
    return success();
  }
};

struct ReshapeOpCanon final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Fold noop reshape.
    if (op.getType() == op.getOperand().getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    // Fold reshape of a constant.
    ElementsAttr cstAttr;
    if (!matchPattern(op.getOperand(), m_Constant(&cstAttr))) {
      return failure();
    }

    if (auto splat = dyn_cast<SplatElementsAttr>(cstAttr)) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, SplatElementsAttr::get(op.getType(),
                                     splat.getSplatValue<Attribute>()));
      return success();
    }

    auto elements =
        llvm::to_vector_of<Attribute>(cstAttr.getValues<Attribute>());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), elements));
    return success();
  }
};

struct TransposeOpCanon final : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this transpose is a noop and use the operand instead.
    if (!isIotaRange(op.getPermutation())) return failure();

    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

struct StableHLOCanonicalize final
    : impl::StableHLOCanonicalizeBase<StableHLOCanonicalize> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateCanonicalizationPatterns(ctx, &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
void populateCanonicalizationPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns,
                                      PatternBenefit benefit) {
  patterns->add<
      // Arithmetic ops.
      AddOpCanon, SubtractOpCanon, MulOpCanon, CompareOpCanon,
      // Complex ops.
      RealOpCanon, ImagOpCanon,
      // Query ops.
      GetDimensionSizeOpCanon, GetTupleElementOpCanon,
      // Shape manipulation(-ish) ops.
      BroadcastInDimOpCanon, ConcatenateOpCanon, ConvertOpCanon,
      DynamicReshapeOpCanon, ReshapeOpCanon, TransposeOpCanon>(context,
                                                               benefit);
}
}  // namespace mlir::iree_compiler::stablehlo

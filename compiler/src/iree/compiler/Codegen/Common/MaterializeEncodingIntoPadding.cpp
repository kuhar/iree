// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZEENCODINGINTOPADDINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using namespace IREE::Encoding;
using IREE::Codegen::MaterializeEncodingInfo;

namespace {

class MaterializePadEncodingTypeConverter
    : public MaterializeEncodingTypeConverter {
public:
  MaterializePadEncodingTypeConverter(
      IREE::HAL::ExecutableTargetAttr targetAttr)
      : MaterializeEncodingTypeConverter(
            IREE::Codegen::EncodingNopLayoutAttr::get(
                targetAttr.getContext())) {
    if (auto attr = targetAttr.getConfiguration().getNamed("encoding")) {
      if (auto encodingLayoutAttr =
              dyn_cast<IREE::Encoding::EncodingLayoutAttrInterface>(
                  attr->getValue())) {
        encodingAttr = encodingLayoutAttr.cloneWithSimplifiedConfig(
            targetAttr.getConfiguration());
        llvm::errs() << "Encoding attr: " << encodingAttr << "\n";
      }
    }
  }

  IREE::GPU::GPUPadLayoutAttr getPadLayout(RankedTensorType type) const {
    auto iface = dyn_cast_or_null<IREE::Encoding::EncodingLayoutAttrInterface>(
        encodingAttr);
    if (!iface) {
      return nullptr;
    }

    auto layout =
        dyn_cast_or_null<IREE::GPU::GPUPadLayoutAttr>(iface.getLayout(type));
    llvm::errs() << "Layout for " << type << "\n" << layout << "\n";
    return layout;
  }

  std::optional<int64_t> getPadK(RankedTensorType type) const {
    auto layout = getPadLayout(type);
    if (!layout) {
      return std::nullopt;
    }

    auto padK = layout.getConfiguration().getAs<IntegerAttr>("pad_k");
    if (!padK) {
      return std::nullopt;
    }
    return padK.getInt();
  }

  RankedTensorType getPaddedType(RankedTensorType type) const {
    std::optional<int64_t> padK = getPadK(type);
    if (!padK) {
      return type;
    }

    auto newShape = llvm::to_vector_of<int64_t>(type.getShape());
    newShape.back() += *padK;
    return RankedTensorType::get(newShape, type.getElementType());
  }

private:
  Attribute encodingAttr;
};

struct MaterializeSubspanOp final
    : OpMaterializeEncodingPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpMaterializeEncodingPattern::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
        subspanOp.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "expected result type to be !flow.dispatch.tensor");
    }
    auto boundTensorType =
        llvm::dyn_cast<RankedTensorType>(resultType.getBoundType());
    if (!boundTensorType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "bound type is not a RankedTensorType");
    }

    auto &typeConverter =
        *static_cast<const MaterializePadEncodingTypeConverter *>(
            getTypeConverter());
    auto convertedBoundType =
        typeConverter.convertType<RankedTensorType>(boundTensorType);
    if (!convertedBoundType || convertedBoundType == boundTensorType) {
      return rewriter.notifyMatchFailure(subspanOp, "bound type already valid");
    }

    llvm::errs() << "materializeSubspanOp:\n" << subspanOp << "\n";
    llvm::errs() << "converted bound type:\n" << convertedBoundType << "\n";
    RankedTensorType paddedType = typeConverter.getPaddedType(boundTensorType);
    llvm::errs() << "padded bound type:\n" << paddedType << "\n";
    if (paddedType == convertedBoundType) {
      return failure();
    }

    auto newResultType =
        IREE::Flow::DispatchTensorType::get(resultType.getAccess(), paddedType);
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, newResultType, subspanOp.getLayout(), subspanOp.getBinding(),
        subspanOp.getByteOffset(), subspanOp.getDynamicDims(),
        subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());
    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorLoadOp
    : public OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorLoadOp> {
  using OpMaterializeEncodingPattern::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the load covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial loads.
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    auto sourceType = loadOp.getSourceType();
    auto boundTensorType = cast<RankedTensorType>(sourceType.getBoundType());
    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
    if (typeConverter.convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(loadOp, "bound type already valid");
    }

    llvm::errs() << "materializeTensorLoad:\n" << loadOp << "\n";
    RankedTensorType paddedType = typeConverter.getPaddedType(boundTensorType);
    llvm::errs() << "bound type: " << boundTensorType << "\n";
    llvm::errs() << "padded bound type: " << paddedType << "\n";
    if (paddedType == boundTensorType) {
      return failure();
    }
    return failure();

    SmallVector<OpFoldResult> newMixedSizes = getMixedValues(
        boundTensorType.getShape(), loadOp.getSourceDims(), rewriter);

    SmallVector<OpFoldResult> newOffsets(newMixedSizes.size(),
                                         rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> newStrides(newMixedSizes.size(),
                                         rewriter.getIndexAttr(1));
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        loadOp, adaptor.getSource(), newDynamicDims, newOffsets, newMixedSizes,
        newStrides);

    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorStoreOp
    : public OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpMaterializeEncodingPattern::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the store covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial stores.
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    auto targetType = storeOp.getTargetType();
    auto boundTensorType = cast<RankedTensorType>(targetType.getBoundType());
    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
    if (typeConverter.convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(storeOp, "bound type already valid");
    }

    llvm::errs() << "materializeTensorStore:\n" << storeOp << "\n";
    RankedTensorType paddedType = typeConverter.getPaddedType(boundTensorType);
    llvm::errs() << "bound type: " << boundTensorType << "\n";
    llvm::errs() << "padded bound type: " << paddedType << "\n";
    if (paddedType == boundTensorType) {
      return failure();
    }

    SmallVector<OpFoldResult> newMixedSizes = getMixedValues(
        paddedType.getShape(), storeOp.getTargetDims(), rewriter);
    SmallVector<OpFoldResult> newOffsets(newMixedSizes.size(),
                                         rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> newStrides(newMixedSizes.size(),
                                         rewriter.getIndexAttr(1));
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);

    auto rawValue = adaptor.getValue()
                        .getDefiningOp<UnrealizedConversionCastOp>()
                        ->getOperand(0);
    auto rawTarget = adaptor.getTarget()
                         .getDefiningOp<UnrealizedConversionCastOp>()
                         ->getOperand(0);

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, rawValue, rawTarget, newDynamicDims, newOffsets, newMixedSizes,
        newStrides);
    return success();
  }
};

struct SetEncodingOpLoweringConversion
    : public OpMaterializeEncodingPattern<IREE::Encoding::SetEncodingOp> {
  using OpMaterializeEncodingPattern::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter =
        *getTypeConverter<MaterializePadEncodingTypeConverter>();
    RankedTensorType resultType = encodingOp.getResultType();
    RankedTensorType sourceType = encodingOp.getSourceType();
    RankedTensorType paddedType = typeConverter.getPaddedType(resultType);
    llvm::errs() << "set encoding op: " << encodingOp << "\n";
    llvm::errs() << "source type: " << sourceType << "\n";
    llvm::errs() << "padded result type: " << paddedType << "\n";
    if (resultType == paddedType) {
      return failure();
    }

    Location loc = encodingOp.getLoc();
    SmallVector<Value> dynamicResultSizes;
    for (size_t dim = 0, e = sourceType.getNumDynamicDims(); dim != e; ++dim) {
      dynamicResultSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, adaptor.getSource(), dim));
    }
    Value empty =
        rewriter.create<tensor::EmptyOp>(loc, paddedType, dynamicResultSizes);
    llvm::errs() << "empty: " << empty << "\n";

    SmallVector<OpFoldResult> offsets(paddedType.getRank(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(paddedType.getRank(),
                                      rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, adaptor.getSource());
    auto insertOp = rewriter.create<tensor::InsertSliceOp>(
        loc, adaptor.getSource(), empty, offsets, sizes, strides);
    if (failed(insertOp.verify())) {
      llvm::errs() << "insert failed to verify\n";
    }
    llvm::errs() << "insert: " << insertOp << "\n";
    rewriter.replaceOp(encodingOp, insertOp);
    return success();
  }
};

struct MaterializeEncodingIntoPaddingPass final
    : impl::MaterializeEncodingIntoPaddingPassBase<
          MaterializeEncodingIntoPaddingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface operation = getOperation();

    auto materializeEncodingValueFn =
        [](RankedTensorType, OpBuilder &,
           Location) -> FailureOr<MaterializeEncodingValueInfo> {
      return failure();
    };

    RewritePatternSet materializeEncodingPattern(context);
    MaterializePadEncodingTypeConverter typeConverter(
        IREE::HAL::ExecutableTargetAttr::lookup(operation));
    MaterializeEncodingConversionTarget target(*context);
    populateMaterializeEncodingPatterns(materializeEncodingPattern, target,
                                        typeConverter,
                                        materializeEncodingValueFn);
    materializeEncodingPattern.add<
        MaterializeSubspanOp, MaterializeFlowDispatchTensorLoadOp,
        MaterializeFlowDispatchTensorStoreOp, SetEncodingOpLoweringConversion>(
        context, typeConverter, materializeEncodingValueFn,
        PatternBenefit{100});

    if (failed(applyPartialConversion(operation, target,
                                      std::move(materializeEncodingPattern)))) {
      operation.emitOpError("materialization failed");
      return signalPassFailure();
    }

    // Add patterns to resolve dims ops and cleanups.
    {
      RewritePatternSet patterns(context);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
      context->getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(operation, std::move(patterns)))) {
        operation.emitOpError("folding patterns failed");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

void addEncodingToPaddingPasses(FunctionLikeNest &passManager) {
  passManager.addPass(createMaterializeEncodingIntoPaddingPass)
      .addPass(createBufferizeCopyOnlyDispatchesPass)
      .addPass(createCanonicalizerPass);
}

} // namespace mlir::iree_compiler

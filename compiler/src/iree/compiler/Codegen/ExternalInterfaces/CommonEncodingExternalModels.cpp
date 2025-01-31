// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- CommonEncodingExternalModels.cpp
//--------------------------------------===//
//
// Implements target-agnostic encoding interfaces. These have target-specific
// emission but common lowering.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/ExternalInterfaces/CommonEncodingExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#include <numeric>

#define DEBUG_TYPE "iree-common-encoding-external-models"

namespace mlir::iree_compiler::IREE::Codegen {

namespace {

struct PadSerializedEncodingLayoutAttrInterface final
    : Encoding::SerializedEncodingLayoutAttrInterface::ExternalModel<
          PadSerializedEncodingLayoutAttrInterface, PadEncodingLayoutAttr> {
  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    int64_t padValue = 0;
    if (auto padAttr = cast<PadEncodingLayoutAttr>(attr)
                           .getConfiguration()
                           .getAs<IntegerAttr>("pad_k")) {
      padValue = padAttr.getInt();
    }

    llvm::errs() << "attr: " << attr << "\n";
    llvm::errs() << "pad value: " << padValue << "\n";
    llvm::errs() << "type: " << type << "\n";
    llvm::errs() << "dynamic dims: ";
    llvm::interleaveComma(dynamicDims, llvm::errs());
    llvm::errs() << "\n";

    auto newStaticShape =
        llvm::filter_to_vector<4>(type.getShape(), [](int64_t dim) {
          return !ShapedType::isDynamic(dim);
        });
    newStaticShape.back() += padValue;
    // Account for the element type.
    newStaticShape.push_back(
        llvm::divideCeil(type.getElementTypeBitWidth(), 8));
    llvm::errs() << "new static shape: ";
    llvm::interleaveComma(newStaticShape, llvm::errs());
    llvm::errs() << "\n";

    int64_t totalStaticSize = std::accumulate(
        newStaticShape.begin(), newStaticShape.end(), 1, std::multiplies<>{});
    llvm::errs() << "total static size: " << totalStaticSize << "\n";
    Value totalSize =
        builder.create<arith::ConstantIndexOp>(loc, totalStaticSize);

    for (Value dim : dynamicDims) {
      totalSize = builder.create<arith::MulIOp>(loc, totalSize, dim);
    }
    llvm::errs() << "total size: " << totalSize << "\n";
    return totalSize;
  }
};

} // namespace

void registerCommonEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Codegen::IREECodegenDialect *dialect) {
        IREE::Codegen::PadEncodingLayoutAttr::attachInterface<
            PadSerializedEncodingLayoutAttrInterface>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::Codegen

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <optional>
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-vector-reduction-to-gpu"

namespace mlir {
namespace iree_compiler {

void debugPrint(func::FuncOp funcOp, const char *message) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << message << " ---//\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

/// Emit shared local memory allocation in case it is needed when lowering the
/// warp operations.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  MemRefType memrefType;
  auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  if (auto vectorType = llvm::dyn_cast<VectorType>(type)) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(),
                        MemRefLayoutAttrInterface{}, addressSpaceAttr);
  } else {
    memrefType = MemRefType::get({1}, type, MemRefLayoutAttrInterface{},
                                 addressSpaceAttr);
  }
  return builder.create<memref::AllocOp>(loc, memrefType);
}

/// Returns true if the given op is a memref.load from a uniform buffer or
/// read-only storage buffer.
static bool isUniformLoad(Operation *op) {
  using namespace IREE::HAL;

  auto loadOp = dyn_cast<memref::LoadOp>(op);
  if (!loadOp)
    return false;
  auto space = loadOp.getMemRefType().getMemorySpace();
  auto attr = llvm::dyn_cast_if_present<DescriptorTypeAttr>(space);
  if (!attr)
    return false;

  if (attr.getValue() == DescriptorType::UniformBuffer)
    return true;

  auto subspan = loadOp.getMemRef().getDefiningOp<InterfaceBindingSubspanOp>();
  if (!subspan)
    return false;
  if (auto flags = subspan.getDescriptorFlags()) {
    if (bitEnumContainsAll(*flags, IREE::HAL::DescriptorFlags::ReadOnly))
      return true;
  }
  return false;
}

/// Hoist uniform operations as well as special hal operations that have side
/// effect but are safe to move out of the warp single lane region.
static void
moveScalarAndBindingUniformCode(vector::WarpExecuteOnLane0Op warpOp) {
  /// Hoist ops without side effect as well as special binding ops.
  auto canBeHoisted = [](Operation *op,
                         function_ref<bool(Value)> definedOutside) {
    if (op->getNumRegions() != 0)
      return false;
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;
    if (isMemoryEffectFree(op))
      return true;

    if (isa<IREE::HAL::InterfaceBindingSubspanOp,
            IREE::HAL::InterfaceConstantLoadOp, memref::AssumeAlignmentOp>(op))
      return true;
    if (isUniformLoad(op))
      return true;
    // Shared memory is already scoped to the workgroup and can safely be
    // hoisted out of the the warp op.
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      if (hasSharedMemoryAddressSpace(allocOp.getType())) {
        return true;
      }
    }

    return false;
  };
  Block *body = warpOp.getBody();

  // Keep track of the ops we want to hoist.
  llvm::SmallSetVector<Operation *, 8> opsToMove;

  // Helper to check if a value is or will be defined outside of the region.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto *definingOp = value.getDefiningOp();
    return (definingOp && opsToMove.count(definingOp)) ||
           warpOp.isDefinedOutsideOfRegion(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there.
  for (auto &op : body->without_terminator()) {
    bool hasVectorResult = llvm::any_of(op.getResults(), [](Value result) {
      return llvm::isa<VectorType>(result.getType());
    });
    if ((!hasVectorResult || isUniformLoad(&op)) &&
        canBeHoisted(&op, isDefinedOutsideOfBody)) {
      opsToMove.insert(&op);
    }
  }

  // Move all the ops marked as uniform outside of the region.
  for (Operation *op : opsToMove)
    op->moveBefore(warpOp);
}

namespace {

/// Pattern to convert InsertElement to broadcast, this is a workaround until
/// MultiDimReduction distribution is supported.
class InsertElementToBroadcast final
    : public OpRewritePattern<vector::InsertElementOp> {
public:
  using OpRewritePattern<vector::InsertElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertElementOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (insertOp.getDestVectorType().getNumElements() != 1)
      return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.getSource());
    return success();
  }
};

/// Pattern to sink `gpu.barrier` ops out of a `warp_execute_on_lane_0` op.
class WarpOpBarrier : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<vector::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    Operation *lastNode = yield->getPrevNode();
    auto barrierOp = dyn_cast_or_null<gpu::BarrierOp>(lastNode);
    if (!barrierOp)
      return failure();

    rewriter.setInsertionPointAfter(warpOp);
    (void)rewriter.create<gpu::BarrierOp>(barrierOp.getLoc());
    rewriter.eraseOp(barrierOp);
    return success();
  }
};

static Value simpleWarpShuffleFunction(Location loc, OpBuilder &builder,
                                       Value val, Value srcIdx,
                                       int64_t warpSz) {
  assert((val.getType().isF32() || val.getType().isInteger(32)) &&
         "unsupported shuffle type");
  Type i32Type = builder.getIntegerType(32);
  Value srcIdxI32 = builder.create<arith::IndexCastOp>(loc, i32Type, srcIdx);
  Value warpSzI32 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(i32Type, warpSz));
  Value result = builder
                     .create<gpu::ShuffleOp>(loc, val, srcIdxI32, warpSzI32,
                                             gpu::ShuffleMode::IDX)
                     .getResult(0);
  return result;
}

static std::optional<SmallVector<int64_t>>
getNativeVectorShapeImpl(VectorType type) {
  if (type.isScalable())
    return std::nullopt;
  if (type.getRank() != 2)
    return std::nullopt;

  auto shape = llvm::to_vector(type.getShape());
  shape[0] = 1;
  return shape;
}

static std::optional<SmallVector<int64_t>> getNativeVectorShape(Operation *op) {
  llvm::errs() << "Get native vector sharpe for: " << *op << "\n";
  // return std::nullopt;
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      return getNativeVectorShapeImpl(vecType);
    }
  }

  auto res =
      TypeSwitch<Operation *, std::optional<SmallVector<int64_t>>>(op)
          .Case<VectorTransferOpInterface>(
              [](VectorTransferOpInterface op)
                  -> std::optional<SmallVector<int64_t>> {
                if (isa<vector::TransferWriteOp>(op))
                  return SmallVector<int64_t>{1};

                return getNativeVectorShapeImpl(op.getVectorType());
              })
          .Case<vector::MultiDimReductionOp>(
              [](vector::MultiDimReductionOp op) {
                return getNativeVectorShapeImpl(op.getSourceVectorType());
              })
          .Case<vector::BroadcastOp>([](vector::BroadcastOp op) {
            return getNativeVectorShapeImpl(op.getResultVectorType());
          })
          .Default([](Operation *) { return std::nullopt; });

  if (res) {
    llvm::errs() << "\tshape: [";
    llvm::interleaveComma(*res, llvm::errs());
    llvm::errs() << "]\n";
  } else {
    llvm::errs() << "\tshape: <nullopt>\n";
  }

  return res;
}

/// Pattern to sink `gpu.barrier` ops out of a `warp_execute_on_lane_0` op.
struct ScfForSplit final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return failure();

    auto type = dyn_cast<VectorType>(op.getResultTypes().front());
    if (!type || type.isScalable() || type.getRank() <= 1)
      return failure();

    if (op.getNumRegionIterArgs() != 1)
      return failure();

    auto shape = llvm::to_vector(type.getShape());
    int64_t numResults = shape.front();
    if (numResults == 1)
      return failure();

    llvm::errs() << "JAKUB: scf.for loop\n";
    Location loc = op.getLoc();
    llvm::SmallVector<Value> extractedInputs(numResults);
    Value arg = op.getInitArgs().front();
    for (auto [idx, input] : llvm::enumerate(extractedInputs)) {
      input = rewriter.create<vector::ExtractOp>(loc, arg, idx);
      llvm::errs() << "\tNew input: " << input << "\n";
    }

    auto newLoop =
        op.replaceWithAdditionalIterOperands(rewriter, extractedInputs, true);
    if (failed(newLoop))
      return failure();

    op = cast<scf::ForOp>(*newLoop);

    rewriter.startRootUpdate(op);
    Block &block = op.getRegion().getBlocks().front();
    rewriter.setInsertionPointToStart(&block);
    BlockArgument origIterArg = op.getRegionIterArg(0);
    Value newIterArg = rewriter.create<arith::ConstantOp>(
        origIterArg.getLoc(), origIterArg.getType(),
        rewriter.getZeroAttr(origIterArg.getType()));
    for (auto [idx, iterArg] :
         llvm::enumerate(op.getRegionIterArgs().drop_front())) {
      newIterArg = rewriter.create<vector::InsertOp>(origIterArg.getLoc(),
                                                     iterArg, newIterArg, idx);
    }
    origIterArg.replaceAllUsesWith(newIterArg);

    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());
    Value origYieldValue = yieldOp->getOperand(0);
    rewriter.setInsertionPoint(yieldOp);
    for (auto [idx, operand] :
         llvm::enumerate(yieldOp->getOpOperands().drop_front())) {
      operand.assign(rewriter.create<vector::ExtractOp>(yieldOp.getLoc(),
                                                        origYieldValue, idx));
    }
    rewriter.finalizeRootUpdate(op);

    rewriter.setInsertionPointAfter(op);
    Value newResult = rewriter.create<arith::ConstantOp>(
        op.getLoc(), type, rewriter.getZeroAttr(type));
    for (auto [idx, result] : llvm::enumerate(op->getResults().drop_front())) {
      newResult = rewriter.create<vector::InsertOp>(op.getLoc(), result,
                                                    newResult, idx);
    }
    rewriter.replaceAllUsesWith(op->getResult(0), newResult);

    llvm::errs() << "With replace operands: " << op << "\n";
    return success();
  }
};

/// Adds patterns to unroll vector ops to SPIR-V native vector size.
static void populateVectorUnrollPatterns(RewritePatternSet &patterns) {
  auto options =
      vector::UnrollVectorOptions().setNativeShapeFn(getNativeVectorShape);
  vector::populateVectorUnrollPatterns(patterns, options);
}

class VectorReductionToGPUPass
    : public VectorReductionToGPUBase<VectorReductionToGPUPass> {
public:
  explicit VectorReductionToGPUPass(
      bool expandSubgroupReduction,
      std::function<int(func::FuncOp)> getWarpSize)
      : expandSubgroupReduction(expandSubgroupReduction),
        getWarpSize(getWarpSize) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect, gpu::GPUDialect,
                    affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    debugPrint(funcOp, "after step #0: before vector reduction to gpu");

    // 1. Pre-process multiDimReductions.
    // TODO: Remove once MultiDimReduce is supported by distribute patterns.
    {
      RewritePatternSet patterns(ctx);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerReduction);
      // Add clean up patterns after lowering of multidimreduce lowering.
      patterns.add<InsertElementToBroadcast>(ctx);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    debugPrint(funcOp, "after step #1: preprocessing reduction ops");

    {
      RewritePatternSet patterns(ctx);
      populateVectorUnrollPatterns(patterns);
      patterns.add<ScfForSplit>(ctx);
      scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ReductionOp::getCanonicalizationPatterns(patterns, ctx);
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after unrolling vector ops");

    auto workgroupSize = llvm::map_to_vector(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return llvm::cast<IntegerAttr>(attr).getInt(); });
    assert(workgroupSize[1] == 1 && workgroupSize[2] == 1);
    // 2. Create the warp op and move the function body into it.
    const int groupSize = workgroupSize[0];
    Location loc = funcOp.getLoc();
    OpBuilder builder(funcOp);
    auto threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                   gpu::Dimension::x);
    auto cstGroupSize = builder.create<arith::ConstantIndexOp>(loc, groupSize);
    auto warpOp = builder.create<vector::WarpExecuteOnLane0Op>(
        loc, TypeRange(), threadX.getResult(), groupSize);
    warpOp.getWarpRegion().takeBody(funcOp.getFunctionBody());
    Block &newBlock = funcOp.getFunctionBody().emplaceBlock();
    threadX->moveBefore(&newBlock, newBlock.end());
    cstGroupSize->moveBefore(&newBlock, newBlock.end());
    warpOp->moveBefore(&newBlock, newBlock.end());
    warpOp.getWarpRegion().getBlocks().back().back().moveBefore(&newBlock,
                                                                newBlock.end());
    builder.setInsertionPointToEnd(&warpOp.getWarpRegion().getBlocks().back());
    builder.create<vector::YieldOp>(loc);

    debugPrint(funcOp, "after step #2: wrapping code with the warp execute op");

    // 3. Hoist the scalar code outside of the warp region.
    moveScalarAndBindingUniformCode(warpOp);

    debugPrint(funcOp, "after step #3: hosting uniform code");

    // 4. Distribute transfer write operations and propagate vector
    // distribution.
    {
      int warpSize = this->getWarpSize ? this->getWarpSize(funcOp) : 32;
      auto groupReductionFn = [=](Location loc, OpBuilder &builder, Value input,
                                  vector::CombiningKind kind,
                                  uint32_t size) -> Value {
        return emitGPUGroupReduction(loc, builder, input, kind, size, warpSize,
                                     expandSubgroupReduction);
      };
      auto distributionFn = [](Value val) {
        auto vecType = llvm::dyn_cast<VectorType>(val.getType());
        if (!vecType)
          return AffineMap::get(val.getContext());
        // Create a map (d0, d1) -> (d1) to distribute along the inner
        // dimension. Once we support n-d distribution we can add more
        // complex cases.
        int64_t vecRank = vecType.getRank();
        OpBuilder builder(val.getContext());
        return AffineMap::get(vecRank, 0,
                              builder.getAffineDimExpr(vecRank - 1));
      };
      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(
          patterns, distributionFn, simpleWarpShuffleFunction);
      vector::populateDistributeReduction(patterns, groupReductionFn);
      vector::populateDistributeTransferWriteOpPatterns(patterns,
                                                        distributionFn);
      patterns.add<WarpOpBarrier>(patterns.getContext(), 3);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    debugPrint(funcOp, "after step #4: propagating distribution");

    // 5. Lower the remaining WarpExecuteOnLane0 ops.
    {
      RewritePatternSet patterns(ctx);
      vector::WarpExecuteOnLane0LoweringOptions options;
      options.warpAllocationFn = allocateGlobalSharedMemory;
      options.warpSyncronizationFn = [](Location loc, OpBuilder &builder,
                                        vector::WarpExecuteOnLane0Op warpOp) {
        builder.create<gpu::BarrierOp>(loc);
      };
      vector::populateWarpExecuteOnLane0OpToScfForPattern(patterns, options);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    debugPrint(funcOp, "after step #5: lowering remaining ops");
  }

private:
  bool expandSubgroupReduction;
  std::function<int(func::FuncOp)> getWarpSize;
};

} // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertVectorReductionToGPUPass(
    bool expandSubgroupReduction,
    std::function<int(func::FuncOp)> getWarpSize) {
  return std::make_unique<VectorReductionToGPUPass>(expandSubgroupReduction,
                                                    getWarpSize);
}

} // namespace iree_compiler
} // namespace mlir

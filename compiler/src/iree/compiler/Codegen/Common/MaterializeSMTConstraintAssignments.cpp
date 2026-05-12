// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTMATERIALIZESMTCONSTRAINTASSIGNMENTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

using IREE::Codegen::CompilationInfoAttr;
using IREE::Codegen::ConstraintsOp;
using IREE::Codegen::IntKnobAttr;
using IREE::Codegen::OneOfKnobAttr;
using IREE::Codegen::TranslationInfoAttr;
using IREE::GPU::LoweringConfigAttr;

static InFlightDiagnostic emitMaterializationError(ConstraintsOp op) {
  return op.emitError(
      "failed to materialize compilation_info from constraints: ");
}

static FailureOr<Attribute>
materializeKnobAttribute(ConstraintsOp op, Attribute attr,
                         const llvm::StringMap<int64_t> &assignments) {
  MLIRContext *ctx = op.getContext();
  return llvm::TypeSwitch<Attribute, FailureOr<Attribute>>(attr)
      .Case([&](IntKnobAttr knob) -> FailureOr<Attribute> {
        StringRef name = knob.getName().getValue();
        auto it = assignments.find(name);
        if (it == assignments.end()) {
          emitMaterializationError(op)
              << "missing assignment for knob '" << name << "'";
          return failure();
        }
        return IntegerAttr::get(IntegerType::get(ctx, 64), it->second);
      })
      .Case([&](OneOfKnobAttr knob) -> FailureOr<Attribute> {
        StringRef name = knob.getName().getValue();
        auto it = assignments.find(name);
        if (it == assignments.end()) {
          emitMaterializationError(op)
              << "missing assignment for knob '" << name << "'";
          return failure();
        }
        ArrayAttr options = knob.getOptions();
        int64_t index = it->second;
        if (index < 0 || index >= static_cast<int64_t>(options.size())) {
          emitMaterializationError(op)
              << "assignment for knob '" << name
              << "' is out of range: " << index << " is not in [0, "
              << options.size() << ")";
          return failure();
        }
        return options[index];
      })
      .Case([&](ArrayAttr array) -> FailureOr<Attribute> {
        SmallVector<Attribute> materialized;
        materialized.reserve(array.size());
        for (Attribute element : array) {
          FailureOr<Attribute> materializedElement =
              materializeKnobAttribute(op, element, assignments);
          if (failed(materializedElement)) {
            return failure();
          }
          materialized.push_back(*materializedElement);
        }
        return ArrayAttr::get(ctx, materialized);
      })
      .Case([&](DictionaryAttr dict) -> FailureOr<Attribute> {
        SmallVector<NamedAttribute> materialized;
        materialized.reserve(dict.size());
        for (NamedAttribute entry : dict) {
          FailureOr<Attribute> materializedValue =
              materializeKnobAttribute(op, entry.getValue(), assignments);
          if (failed(materializedValue)) {
            return failure();
          }
          materialized.emplace_back(entry.getName(), *materializedValue);
        }
        return DictionaryAttr::get(ctx, materialized);
      })
      .Default([&](Attribute attr) -> FailureOr<Attribute> { return attr; });
}

static FailureOr<DictionaryAttr>
materializeKnobsDictionary(ConstraintsOp op,
                           const llvm::StringMap<int64_t> &assignments) {
  FailureOr<Attribute> materialized =
      materializeKnobAttribute(op, op.getKnobsAttr(), assignments);
  if (failed(materialized)) {
    return failure();
  }
  auto dict = dyn_cast<DictionaryAttr>(*materialized);
  if (!dict) {
    emitMaterializationError(op) << "materialized knobs must be a dictionary";
    return failure();
  }
  return dict;
}

static FailureOr<SmallVector<int64_t>>
extractI64Array(ConstraintsOp op, Attribute attr, StringRef fieldName) {
  auto array = dyn_cast_if_present<ArrayAttr>(attr);
  if (!array) {
    emitMaterializationError(op)
        << "expected '" << fieldName << "' to be an integer array";
    return failure();
  }

  SmallVector<int64_t> values;
  values.reserve(array.size());
  for (Attribute element : array) {
    auto intAttr = dyn_cast<IntegerAttr>(element);
    if (!intAttr) {
      emitMaterializationError(op)
          << "expected '" << fieldName << "' to contain only integers";
      return failure();
    }
    values.push_back(intAttr.getInt());
  }
  return values;
}

static FailureOr<int64_t> extractI64(ConstraintsOp op, Attribute attr,
                                     StringRef fieldName) {
  auto intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (!intAttr) {
    emitMaterializationError(op)
        << "expected '" << fieldName << "' to be an integer";
    return failure();
  }
  return intAttr.getInt();
}

static void addIfPresent(OpBuilder &builder,
                         SmallVectorImpl<NamedAttribute> &entries,
                         DictionaryAttr source, StringRef key) {
  if (Attribute attr = source.get(key)) {
    entries.emplace_back(builder.getStringAttr(key), attr);
  }
}

static FailureOr<CompilationInfoAttr>
materializeCompilationInfo(ConstraintsOp op, DictionaryAttr materializedKnobs) {
  OpBuilder builder(op.getContext());

  DictionaryAttr loweringSource = materializedKnobs;
  if (auto nestedLowering = dyn_cast_if_present<DictionaryAttr>(
          materializedKnobs.get("lowering_config"))) {
    loweringSource = nestedLowering;
  }

  SmallVector<NamedAttribute> loweringEntries;
  addIfPresent(builder, loweringEntries, loweringSource, "workgroup");
  addIfPresent(builder, loweringEntries, loweringSource, "reduction");
  addIfPresent(builder, loweringEntries, loweringSource, "subgroup");
  addIfPresent(builder, loweringEntries, loweringSource, "subgroup_basis");
  addIfPresent(builder, loweringEntries, loweringSource, "mma_kind");

  LoweringConfigAttr loweringConfig = LoweringConfigAttr::get(
      op.getContext(), DictionaryAttr::get(op.getContext(), loweringEntries));

  DictionaryAttr translationSource = materializedKnobs;
  if (auto nestedTranslation = dyn_cast_if_present<DictionaryAttr>(
          materializedKnobs.get("translation_info"))) {
    translationSource = nestedTranslation;
  }

  SmallVector<int64_t> workgroupSize = {1, 1, 1};
  if (Attribute workgroupSizeAttr = translationSource.get("workgroup_size")) {
    FailureOr<SmallVector<int64_t>> maybeWorkgroupSize =
        extractI64Array(op, workgroupSizeAttr, "workgroup_size");
    if (failed(maybeWorkgroupSize)) {
      return failure();
    }
    workgroupSize = *maybeWorkgroupSize;
  }

  int64_t subgroupSize = 0;
  if (Attribute subgroupSizeAttr = translationSource.get("subgroup_size")) {
    FailureOr<int64_t> maybeSubgroupSize =
        extractI64(op, subgroupSizeAttr, "subgroup_size");
    if (failed(maybeSubgroupSize)) {
      return failure();
    }
    subgroupSize = *maybeSubgroupSize;
  }

  TranslationInfoAttr translationInfo = TranslationInfoAttr::get(
      op.getContext(), op.getPipeline(), /*codegenSpec=*/SymbolRefAttr(),
      workgroupSize, subgroupSize, /*configuration=*/DictionaryAttr());
  return CompilationInfoAttr::get(op.getContext(), loweringConfig,
                                  translationInfo);
}

static LogicalResult parseAssignments(StringRef assignmentList,
                                      llvm::StringMap<int64_t> &assignments,
                                      Operation *op) {
  SmallVector<StringRef> pairs;
  assignmentList.split(pairs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef pair : pairs) {
    auto [name, valueStr] = pair.split('=');
    name = name.trim();
    valueStr = valueStr.trim();
    if (name.empty() || valueStr.empty()) {
      op->emitError("expected assignments as comma-separated name=value pairs");
      return failure();
    }
    int64_t value = 0;
    if (valueStr.getAsInteger(10, value)) {
      op->emitError("expected integer assignment value in '") << pair << "'";
      return failure();
    }
    assignments[name] = value;
  }
  return success();
}

struct TestMaterializeSMTConstraintAssignmentsPass final
    : impl::TestMaterializeSMTConstraintAssignmentsPassBase<
          TestMaterializeSMTConstraintAssignmentsPass> {
  using Base::Base;

  void runOnOperation() override {
    llvm::StringMap<int64_t> assignmentMap;
    if (failed(parseAssignments(assignments.getValue(), assignmentMap,
                                getOperation()))) {
      signalPassFailure();
      return;
    }

    bool failedMaterialization = false;
    getOperation()->walk([&](ConstraintsOp op) {
      FailureOr<CompilationInfoAttr> compilationInfo =
          materializeCompilationInfoFromConstraints(op, assignmentMap);
      if (failed(compilationInfo)) {
        failedMaterialization = true;
        return WalkResult::interrupt();
      }
      op->setAttr("test.materialized_compilation_info", *compilationInfo);
      return WalkResult::advance();
    });
    if (failedMaterialization) {
      signalPassFailure();
    }
  }
};

} // namespace

FailureOr<CompilationInfoAttr> materializeCompilationInfoFromConstraints(
    ConstraintsOp op, const llvm::StringMap<int64_t> &assignments) {
  FailureOr<DictionaryAttr> materializedKnobs =
      materializeKnobsDictionary(op, assignments);
  if (failed(materializedKnobs)) {
    return failure();
  }
  return materializeCompilationInfo(op, *materializedKnobs);
}

} // namespace mlir::iree_compiler

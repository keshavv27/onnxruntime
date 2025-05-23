// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

namespace onnxruntime {
namespace coreml {

class BatchNormalizationOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  // BatchNormalization opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 7; }

 public:
  bool SupportsMLProgram() const override { return true; }
};

void BatchNormalizationOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // skip everything except input0 for BatchNormalization
  const auto& input_defs = node.InputDefs();
  model_builder.AddInitializerToSkip(input_defs[1]->Name());  // scale
  model_builder.AddInitializerToSkip(input_defs[2]->Name());  // B
  model_builder.AddInitializerToSkip(input_defs[3]->Name());  // mean
  model_builder.AddInitializerToSkip(input_defs[4]->Name());  // var
}

Status BatchNormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                          const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);

  const auto& scale_tensor = *initializers.at(input_defs[1]->Name());
  const auto& bias_tensor = *initializers.at(input_defs[2]->Name());
  const auto& mean_tensor = *initializers.at(input_defs[3]->Name());
  const auto& var_tensor = *initializers.at(input_defs[4]->Name());
  const auto eps = helper.Get("epsilon", 1e-5f);
  const auto channels = scale_tensor.dims()[0];

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.normalization.batch_norm

    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "batch_norm");
    AddOperationInput(*op, "x", input_defs[0]->Name());
    AddOperationInput(*op, "mean", model_builder.AddConstant(op->type(), input_defs[3]->Name() + "mean", mean_tensor));
    AddOperationInput(*op, "variance", model_builder.AddConstant(op->type(), input_defs[4]->Name() + "variance", var_tensor));
    AddOperationInput(*op, "gamma", model_builder.AddConstant(op->type(), input_defs[1]->Name(), scale_tensor));
    AddOperationInput(*op, "beta", model_builder.AddConstant(op->type(), input_defs[2]->Name(), bias_tensor));
    auto input_dtype = input_defs[0]->TypeAsProto()->tensor_type().elem_type();
    if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      MLFloat16 epsilon_fp16(eps);
      AddOperationInput(*op, "epsilon", model_builder.AddScalarConstant(op->type(), "epsilon", epsilon_fp16));
    } else {
      AddOperationInput(*op, "epsilon", model_builder.AddScalarConstant(op->type(), "epsilon", eps));
    }

    AddOperationOutput(*op, *node.OutputDefs()[0]);
    model_builder.AddOperation(std::move(op));
  } else {
    auto* coreml_batch_norm = layer->mutable_batchnorm();
    coreml_batch_norm->set_channels(channels);
    coreml_batch_norm->set_epsilon(eps);
    coreml_batch_norm->set_computemeanvar(false);
    coreml_batch_norm->set_instancenormalization(false);

    ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*coreml_batch_norm->mutable_gamma(), scale_tensor));   // scale
    ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*coreml_batch_norm->mutable_beta(), bias_tensor));     // B
    ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*coreml_batch_norm->mutable_mean(), mean_tensor));     // mean
    ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*coreml_batch_norm->mutable_variance(), var_tensor));  // var

    *layer->mutable_input()->Add() = input_defs[0]->Name();
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool BatchNormalizationOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                                    const logging::Logger& logger) const {
  if (node.OutputDefs().size() != 1) {
    LOGS(logger, VERBOSE) << "Your onnx model may be in training mode, please export "
                             "it in test mode.";
    return false;
  }

  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto input_size = input_shape.size();
  // TODO, support 1d batch normalization (input is 3d)
  // To map 1d input {N,C,H} to 2d {N,C,H,1} first and then squeeze back after
  if (input_size != 4) {
    LOGS(logger, VERBOSE) << "BN only support 4d shape for now, input is "
                          << input_size << "d shape";
    return false;
  }

  NodeAttrHelper helper(node);
  const auto spatial = helper.Get("spatial", 1);
  if (spatial != 1) {
    LOGS(logger, VERBOSE) << "Non-spatial BN is not supported";
    return false;
  }

  const auto& scale_name = input_defs[1]->Name();
  const auto& b_name = input_defs[2]->Name();
  const auto& mean_name = input_defs[3]->Name();
  const auto& var_name = input_defs[4]->Name();
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
  if (!Contains(initializers, scale_name)) {
    LOGS(logger, VERBOSE) << "Scale of BN must be a constant initializer";
    return false;
  }
  if (!Contains(initializers, b_name)) {
    LOGS(logger, VERBOSE) << "B of BN must be a constant initializer";
    return false;
  }
  if (!Contains(initializers, mean_name)) {
    LOGS(logger, VERBOSE) << "Mean of BN must be a constant initializer";
    return false;
  }
  if (!Contains(initializers, var_name)) {
    LOGS(logger, VERBOSE) << "Var of BN must be a constant initializer";
    return false;
  }

#if defined(TARGET_OS_IOS) && defined(TARGET_CPU_X86_64) && TARGET_OS_IOS && TARGET_CPU_X86_64
  // To Pass IOS pipeline https://dev.azure.com/onnxruntime/onnxruntime/_build?definitionId=134&_a=summary
  auto input_dtype = input_defs[0]->TypeAsProto()->tensor_type().elem_type();
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 && input_params.coreml_version < 7) {
    LOGS(logger, VERBOSE) << "float16 input is not supported on the iOS x86_64 simulator"
                          << " due to CoreML producing invalid output.";
    return false;
  }
#endif
  return true;
}

void CreateBatchNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<BatchNormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime

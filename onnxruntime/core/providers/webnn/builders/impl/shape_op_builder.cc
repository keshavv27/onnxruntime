// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ShapeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

Status ShapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                             const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const auto rank = static_cast<int32_t>(input_shape.size());

  emscripten::val desc = emscripten::val::object();
  emscripten::val dims = emscripten::val::array();
  dims.call<void>("push", rank);
  desc.set("dimensions", dims);
  desc.set("shape", dims);
  int data_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
  std::string typed_array_name = "BigInt64Array";
  if (!model_builder.IsInt64Supported()) {
    // Int64 is not supported by current context, use int32 instead.
    data_type = ONNX_NAMESPACE::TensorProto_DataType_INT32;
    typed_array_name = "Int32Array";
  }
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc, data_type), "WebNN backend does not support data type: ", data_type);
  emscripten::val shape_buffer =
      emscripten::val::global(typed_array_name.c_str()).new_(emscripten::val::array(input_shape));
  emscripten::val shape_constant = model_builder.GetBuilder().call<emscripten::val>("constant", desc, shape_buffer);

  NodeAttrHelper helper(node);
  auto true_start = helper.Get("start", 0);
  auto true_end = helper.Get("end", rank);

  // Deal with negative(s) and clamp.
  true_start = std::clamp(true_start + (true_start < 0 ? rank : 0), 0, rank);
  true_end = std::clamp(true_end + (true_end < 0 ? rank : 0), true_start, rank);
  auto slice_length = true_end - true_start;

  emscripten::val starts = emscripten::val::array();
  starts.call<void>("push", true_start);
  emscripten::val sizes = emscripten::val::array();
  sizes.call<void>("push", slice_length);

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  // Since WebNN doesn't support Shape op, we use constant + slice ops as workaround.
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("slice",
                                                                            shape_constant,
                                                                            starts,
                                                                            sizes,
                                                                            options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateShapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ShapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime

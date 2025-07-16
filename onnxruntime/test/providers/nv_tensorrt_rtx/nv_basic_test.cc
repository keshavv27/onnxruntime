// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/common/trt_op_test_utils.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>
#include <string>
#include <thread>
#include <filesystem>
#include <chrono>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {

template <typename T>
class NvExecutionProviderTest : public ::testing::Test {
 protected:
  std::string getTypeAsName() {
    std::string dtype_name = "";
    if constexpr (std::is_same<T, double>::value) {
      dtype_name = "fp64";
    } else if constexpr (std::is_same<T, float>::value) {
      dtype_name = "fp32";
    } else if constexpr (std::is_same<T, BFloat16>::value) {
      dtype_name = "bf16";
    } else if constexpr (std::is_same<T, MLFloat16>::value) {
      dtype_name = "fp16";
    } else if constexpr (std::is_same<T, int8_t>::value) {
      dtype_name = "int8";
    } else if constexpr (std::is_same<T, uint8_t>::value) {
      dtype_name = "uint8";
    } else if constexpr (std::is_same<T, int32_t>::value) {
      dtype_name = "int32";
    } else if constexpr (std::is_same<T, int64_t>::value) {
      dtype_name = "int64";
    }
    return dtype_name;
  }
};

using NvExecutionProviderTestTypes = ::testing::Types<double, float, MLFloat16, BFloat16, uint8_t, int8_t, int32_t, int64_t>;  // double,
TYPED_TEST_SUITE(NvExecutionProviderTest, NvExecutionProviderTestTypes);

std::string PathToUTF8(const PathString& path) {
#ifdef WIN32
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(path);
#else
  return path.c_str();
#endif
}

void clearFileIfExists(PathString path) {
  if (std::filesystem::exists(path)) {
    std::filesystem::remove(path);
  }
}

template <typename T>
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<T> found(rtensor.Data<T>(), rtensor.Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

/**
 * Create a simple model with dynamic or non-dynamic input shape.
 * \param model_name - model name
 * \param graph_name - graph name
 * \param dims - input dimensions
 * \param add_fast_gelu - add FastGelu node which makes the whole model partition into TRT EP and CUDA EP subgraphs.
 *
 * input: "X", "Y" and "Z"
 *        you can specify input dimensions, for example (1, 3, 2), (1, 2) or (1, -1, -1)). Note: -1 means the dimension is dynamic.
 *        All three inputs have the same dimensions.
 * output: "M"
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *       Add (+ float scalar "S")
 *       /
 *     "O"
 *
 *     or
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *    FastGelu (This node will be placed on CUDA EP)
 *     /
 *     *       Add (+ float scalar "S")
 *    /
 *   "O"
 */
static void CreateBaseModel(const PathString& model_name,
                            std::string graph_name,
                            std::vector<int> dims,
                            bool add_fast_gelu = false,
                            ONNX_NAMESPACE::TensorProto_DataType dtype = ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(dtype);

  for (auto dim : dims) {
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  ONNX_NAMESPACE::TypeProto dyn_float_tensor;
  dyn_float_tensor.mutable_tensor_type()->set_elem_type(dtype);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);

  auto& output_arg_2 = graph.GetOrCreateNodeArg("node_2_out_1", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg_2);

  if (add_fast_gelu) {
    auto& output_arg_3 = graph.GetOrCreateNodeArg("node_3_out_1", &dyn_float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_3);

    graph.AddNode("node_3", "FastGelu", "node 3.", inputs, outputs,
                  /* attributes */ nullptr, kMSDomain);

    inputs.clear();
    inputs.push_back(&output_arg_3);
  }

  ONNX_NAMESPACE::TypeProto float_scalar;
  float_scalar.mutable_tensor_type()->set_elem_type(dtype);
  float_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& input_scalar = graph.GetOrCreateNodeArg("S", &float_scalar);
  inputs.push_back(&input_scalar);

  auto& output_arg_4 = graph.GetOrCreateNodeArg("O", &dyn_float_tensor);

  outputs.clear();
  outputs.push_back(&output_arg_4);
  graph.AddNode("node_5", "Add", "node 5.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  status = onnxruntime::Model::Save(model, model_name);
  ASSERT_TRUE(status.IsOK());
}

/**
 * Create a simple model with a single convolution operator.
 * \param model_name - model name
 * \param graph_name - graph name
 * \param input_dims - input dimensions (batch, channel, height, width)
 * \param weight_dims - weight dimensions (output_channels, input_channels/groups, kernel_height, kernel_width)
 * \param bias_dims - bias dimensions (output_channels) - can be empty for no bias
 * \param strides - convolution strides
 * \param dilations - convolution dilations
 * \param padding - convolution padding
 * \param group_count - number of groups for grouped convolution
 *
 * input: "X" (input tensor), "W" (weight tensor), "B" (bias tensor - optional)
 * output: "Y"
 *
 *      "X"    "W"    "B" (optional)
 *        \     |     /
 *         \    |    /
 *          \   |   /
 *           \  |  /
 *            Conv
 *             |
 *            "Y"
 */
static void CreateConvModel(const PathString& model_name,
                           std::string graph_name,
                           std::vector<int64_t> input_dims,
                           std::vector<int64_t> weight_dims,
                           std::vector<int64_t> bias_dims,
                           std::vector<int64_t> strides,
                           std::vector<int64_t> dilations,
                           std::vector<int64_t> padding,
                           int64_t group_count,
                           ONNX_NAMESPACE::TensorProto_DataType dtype = ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // Create input tensor type
  ONNX_NAMESPACE::TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(dtype);
  for (auto dim : input_dims) {
    input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  // Create weight tensor type
  ONNX_NAMESPACE::TypeProto weight_tensor_type;
  weight_tensor_type.mutable_tensor_type()->set_elem_type(dtype);
  for (auto dim : weight_dims) {
    weight_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  // Create input and weight node args
  auto& input_arg = graph.GetOrCreateNodeArg("X", &input_tensor_type);
  auto& weight_arg = graph.GetOrCreateNodeArg("W", &weight_tensor_type);
  inputs.push_back(&input_arg);
  inputs.push_back(&weight_arg);

  // Add bias if specified
  if (!bias_dims.empty()) {
    ONNX_NAMESPACE::TypeProto bias_tensor_type;
    bias_tensor_type.mutable_tensor_type()->set_elem_type(dtype);
    for (auto dim : bias_dims) {
      bias_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }
    auto& bias_arg = graph.GetOrCreateNodeArg("B", &bias_tensor_type);
    inputs.push_back(&bias_arg);
  }

  // Create output tensor type (will be inferred)
  ONNX_NAMESPACE::TypeProto output_tensor_type;
  output_tensor_type.mutable_tensor_type()->set_elem_type(dtype);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_tensor_type);
  outputs.push_back(&output_arg);

  // Create convolution node with attributes
  auto& conv_node = graph.AddNode("conv_node", "Conv", "Convolution node", inputs, outputs);

  // Set attributes
  conv_node.AddAttribute("strides", strides);
  conv_node.AddAttribute("dilations", dilations);
  conv_node.AddAttribute("pads", padding);
  conv_node.AddAttribute("group", group_count);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  status = onnxruntime::Model::Save(model, model_name);
  ASSERT_TRUE(status.IsOK());
}

/**
 * Create a simple model with a single GEMM (General Matrix Multiplication) operator.
 * \param model_name - model name
 * \param graph_name - graph name
 * \param input_a_dims - dimensions for input matrix A
 * \param input_b_dims - dimensions for input matrix B
 * \param bias_dims - dimensions for bias matrix C (can be empty for no bias)
 * \param alpha - scalar multiplier for A*B (default 1.0)
 * \param beta - scalar multiplier for bias C (default 1.0)
 * \param trans_a - whether to transpose A (default 0)
 * \param trans_b - whether to transpose B (default 0)
 *
 * input: "A" (input matrix A), "B" (input matrix B), "C" (bias matrix - optional)
 * output: "Y"
 *
 * Operation: Y = alpha * A * B + beta * C
 *
 *      "A"    "B"    "C" (optional)
 *        \     |     /
 *         \    |    /
 *          \   |   /
 *           \  |  /
 *            GEMM
 *             |
 *            "Y"
 */
static void CreateGemmModel(const PathString& model_name,
                           std::string graph_name,
                           std::vector<int64_t> input_a_dims,
                           std::vector<int64_t> input_b_dims,
                           std::vector<int64_t> bias_dims,
                           float alpha = 1.0f,
                           float beta = 1.0f,
                           int64_t trans_a = 0,
                           int64_t trans_b = 0,
                           ONNX_NAMESPACE::TensorProto_DataType dtype = ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // Create input A tensor type
  ONNX_NAMESPACE::TypeProto input_a_tensor_type;
  input_a_tensor_type.mutable_tensor_type()->set_elem_type(dtype);
  for (auto dim : input_a_dims) {
    input_a_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  // Create input B tensor type
  ONNX_NAMESPACE::TypeProto input_b_tensor_type;
  input_b_tensor_type.mutable_tensor_type()->set_elem_type(dtype);
  for (auto dim : input_b_dims) {
    input_b_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  // Create input A and B node args
  auto& input_a_arg = graph.GetOrCreateNodeArg("A", &input_a_tensor_type);
  auto& input_b_arg = graph.GetOrCreateNodeArg("B", &input_b_tensor_type);
  inputs.push_back(&input_a_arg);
  inputs.push_back(&input_b_arg);

  // Add bias if specified
  if (!bias_dims.empty()) {
    ONNX_NAMESPACE::TypeProto bias_tensor_type;
    bias_tensor_type.mutable_tensor_type()->set_elem_type(dtype);
    for (auto dim : bias_dims) {
      bias_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }
    auto& bias_arg = graph.GetOrCreateNodeArg("C", &bias_tensor_type);
    inputs.push_back(&bias_arg);
  }

  // Create output tensor type (will be inferred)
  ONNX_NAMESPACE::TypeProto output_tensor_type;
  output_tensor_type.mutable_tensor_type()->set_elem_type(dtype);
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &output_tensor_type);
  outputs.push_back(&output_arg);

  // Create GEMM node with attributes
  auto& gemm_node = graph.AddNode("gemm_node", "Gemm", "GEMM node", inputs, outputs);

  // Set attributes
  gemm_node.AddAttribute("alpha", alpha);
  gemm_node.AddAttribute("beta", beta);
  gemm_node.AddAttribute("transA", trans_a);
  gemm_node.AddAttribute("transB", trans_b);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  status = onnxruntime::Model::Save(model, model_name);
  ASSERT_TRUE(status.IsOK());
}

static Ort::IoBinding generate_io_binding(Ort::Session& session, std::map<std::string, std::vector<int64_t>> shape_overwrites = {}) {
  Ort::IoBinding binding(session);
  auto allocator = Ort::AllocatorWithDefaultOptions();
  for (int input_idx = 0; input_idx < int(session.GetInputCount()); ++input_idx) {
    auto input_name = session.GetInputNameAllocated(input_idx, Ort::AllocatorWithDefaultOptions());
    auto full_tensor_info = session.GetInputTypeInfo(input_idx);
    auto tensor_info = full_tensor_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    auto type = tensor_info.GetElementType();
    if (shape_overwrites.find(input_name.get()) == shape_overwrites.end()) {
      for (auto& v : shape) {
        if (v == -1) {
          v = 1;
        }
      }
    } else {
      shape = shape_overwrites[input_name.get()];
    }
    auto input_value = Ort::Value::CreateTensor(allocator,
                                                shape.data(),
                                                shape.size(),
                                                type);
    binding.BindInput(input_name.get(), input_value);
  }

  for (int output_idx = 0; output_idx < int(session.GetOutputCount()); ++output_idx) {
    auto output_name = session.GetOutputNameAllocated(output_idx, Ort::AllocatorWithDefaultOptions());
    binding.BindOutput(output_name.get(), allocator.GetInfo());
  }
  return binding;
}

TEST(NvExecutionProviderTest, ContextEmbedAndReload) {
  PathString model_name = ORT_TSTR("nv_execution_provider_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_test_ctx.onnx");
  auto model_name_ctx_str = PathToUTF8(model_name_ctx);
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model_name, graph_name, dims);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // JIT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name_ctx.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }
}

TEST(NvExecutionProviderTest, ContextEmbedAndReloadDynamic) {
  PathString model_name = ORT_TSTR("nv_execution_provider_dyn_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_dyn_test_ctx.onnx");
  auto model_name_ctx_str = PathToUTF8(model_name_ctx);
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, -1, -1};

  CreateBaseModel(model_name, graph_name, dims);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // JIT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name_ctx.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    std::map<std::string, std::vector<int64_t>> shape_overwrites;
    shape_overwrites["X"] = {1, 5, 5};
    shape_overwrites["Y"] = {1, 5, 1};
    auto io_binding = generate_io_binding(session_object, shape_overwrites);
    session_object.Run(run_options, io_binding);
  }
}

TEST(NvExecutionProviderTest, ContextEmbedAndReloadDataDynamic) {
  PathString model_name = ORT_TSTR("nv_execution_provider_data_dyn_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_data_dyn_test_ctx.onnx");
  auto model_name_ctx_str = PathToUTF8(model_name_ctx);
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, -1, -1};

  CreateBaseModel(model_name, graph_name, dims, true);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // JIT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name_ctx.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    std::map<std::string, std::vector<int64_t>> shape_overwrites;
    shape_overwrites["X"] = {1, 5, 5};
    shape_overwrites["Y"] = {1, 5, 5};
    auto io_binding = generate_io_binding(session_object, shape_overwrites);
    session_object.Run(run_options, io_binding);
  }
}

TYPED_TEST(NvExecutionProviderTest, IOTypeTests) {
  std::string dtype_name = this->getTypeAsName();
  ASSERT_FALSE(dtype_name.empty());
  const std::string model_name_str = "nv_execution_provider_" + dtype_name + ".onnx";
  const PathString model_name = ToPathString(model_name_str);
  std::string graph_name = "test" + dtype_name;
  std::vector<int> dims = {1, -1, -1};

  CreateBaseModel(model_name, graph_name, dims, true);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name.c_str(), so);

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }
}

TEST(NvExecutionProviderTest, ConvolutionModelTest) {
  PathString model_name = ORT_TSTR("nv_execution_provider_conv_test.onnx");
  std::string graph_name = "conv_test";

  // Define convolution parameters
  std::vector<int64_t> input_dims = {1, 3, 4, 4};      // batch=1, channels=3, height=4, width=4
  std::vector<int64_t> weight_dims = {2, 3, 3, 3};     // output_channels=2, input_channels=3, kernel_height=3, kernel_width=3
  std::vector<int64_t> bias_dims = {2};                // output_channels=2
  std::vector<int64_t> strides = {1, 1};               // stride_height=1, stride_width=1
  std::vector<int64_t> dilations = {1, 1};             // dilation_height=1, dilation_width=1
  std::vector<int64_t> padding = {0, 0, 0, 0};         // pad_top=0, pad_left=0, pad_bottom=0, pad_right=0
  int64_t group_count = 1;                             // standard convolution

  CreateConvModel(model_name, graph_name, input_dims, weight_dims, bias_dims,
                  strides, dilations, padding, group_count);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  Ort::SessionOptions so;
  so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
  Ort::Session session_object(env, model_name.c_str(), so);

  // Create input data
  std::vector<float> input_data(1 * 3 * 4 * 4, float(2.00000f));
  std::vector<float> weight_data(2 * 3 * 3 * 3, float(0.50000f));
  std::vector<float> bias_data(2, float(0.10000f));


  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_dims.data(), input_dims.size());
  auto weight_tensor = Ort::Value::CreateTensor<float>(memory_info, weight_data.data(), weight_data.size(), weight_dims.data(), weight_dims.size());
  auto bias_tensor = Ort::Value::CreateTensor<float>(memory_info, bias_data.data(), bias_data.size(), bias_dims.data(), bias_dims.size());


  std::vector<const char*> input_names = {"X", "W", "B"};
  std::vector<Ort::Value> input_values;
  input_values.push_back(std::move(input_tensor));
  input_values.push_back(std::move(weight_tensor));
  input_values.push_back(std::move(bias_tensor));


  std::vector<const char*> output_names = {"Y"};


  auto output_tensors = session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(), input_names.size(), output_names.data(), output_names.size());


  ASSERT_EQ(output_tensors.size(), 1);
  auto& output_tensor = output_tensors[0];

  // Expected output dimensions: batch=1, output_channels=2, height=2, width=2
  // (4x4 input with 3x3 kernel and stride 1, no padding gives 2x2 output)
  std::vector<int64_t> expected_dims = {1, 2, 2, 2};

  // For this test case with all inputs=2.0, weights=0.5, bias=0.1
  // Each output value should be: (3*3*3 * 2.0 * 0.5) + 0.1 = 27 + 0.1 = 27.1
  std::vector<float> expected_values(1 * 2 * 2 * 2, float(27.1f));

 // Verify output dimensions
  auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
  auto output_shape = tensor_info.GetShape();
  ASSERT_EQ(output_shape.size(), expected_dims.size());
  for (size_t i = 0; i < expected_dims.size(); ++i) {
    ASSERT_EQ(output_shape[i], expected_dims[i]);
  }

  // Verify output values
  const float* output_data = output_tensor.GetTensorData<float>();
  size_t output_size = expected_values.size();
  for (size_t i = 0; i < output_size; ++i) {
    ASSERT_NEAR(output_data[i], expected_values[i], float(0.00001f));
  }
}

TEST(NvExecutionProviderTest, GemmModelTest) {
  PathString model_name = ORT_TSTR("nv_execution_provider_gemm_test.onnx");
  std::string graph_name = "gemm_test";

  // Define GEMM parameters
  std::vector<int64_t> input_a_dims = {2, 3};        // 2x3 matrix A
  std::vector<int64_t> input_b_dims = {3, 2};        // 3x2 matrix B
  std::vector<int64_t> bias_dims = {2, 2};           // 2x2 bias matrix C
  float alpha = 1.0f;                                // multiplier for A*B
  float beta = 1.0f;                                 // multiplier for bias C
  int64_t trans_a = 0;                               // don't transpose A
  int64_t trans_b = 0;                               // don't transpose B

  CreateGemmModel(model_name, graph_name, input_a_dims, input_b_dims, bias_dims,
                  alpha, beta, trans_a, trans_b);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  Ort::SessionOptions so;
  so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
  Ort::Session session_object(env, model_name.c_str(), so);


  std::vector<float> input_a_data(2 * 3, 1.0f);      // Fill A with 1.0f
  std::vector<float> input_b_data(3 * 2, 0.5f);      // Fill B with 0.5f
  std::vector<float> bias_data(2 * 2, 0.1f);         // Fill C with 0.1f


  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_a_tensor = Ort::Value::CreateTensor<float>(memory_info, input_a_data.data(), input_a_data.size(), input_a_dims.data(), input_a_dims.size());
  auto input_b_tensor = Ort::Value::CreateTensor<float>(memory_info, input_b_data.data(), input_b_data.size(), input_b_dims.data(), input_b_dims.size());
  auto bias_tensor = Ort::Value::CreateTensor<float>(memory_info, bias_data.data(), bias_data.size(), bias_dims.data(), bias_dims.size());


  std::vector<const char*> input_names = {"A", "B", "C"};
  std::vector<Ort::Value> input_values;
  input_values.push_back(std::move(input_a_tensor));
  input_values.push_back(std::move(input_b_tensor));
  input_values.push_back(std::move(bias_tensor));


  std::vector<const char*> output_names = {"Y"};


  auto output_tensors = session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(), input_names.size(), output_names.data(), output_names.size());


  ASSERT_EQ(output_tensors.size(), 1);
  auto& output_tensor = output_tensors[0];


  std::vector<int64_t> expected_dims = {2, 2};

  // For this test case:
  // A (2x3) filled with 1.0, B (3x2) filled with 0.5, C (2x2) filled with 0.1
  // A * B = [[1.5, 1.5], [1.5, 1.5]] (each element is 1.0*0.5 + 1.0*0.5 + 1.0*0.5 = 1.5)
  // Y = alpha * A * B + beta * C = 1.0 * 1.5 + 1.0 * 0.1 = 1.6
  std::vector<float> expected_values(2 * 2, 1.6f);

  // Verify output dimensions
  auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
  auto output_shape = tensor_info.GetShape();
  ASSERT_EQ(output_shape.size(), expected_dims.size());
  for (size_t i = 0; i < expected_dims.size(); ++i) {
    ASSERT_EQ(output_shape[i], expected_dims[i]);
  }

  const float* output_data = output_tensor.GetTensorData<float>();
  size_t output_size = expected_values.size();
  for (size_t i = 0; i < output_size; ++i) {
    ASSERT_NEAR(output_data[i], expected_values[i], 0.00001f);
  }
}

}  // namespace test
}  // namespace onnxruntime

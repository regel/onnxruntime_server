#include "serving.h"

#include <assert.h>
#include <fmt/format.h>
#include <grpcpp/grpcpp.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "inference/inference.grpc.pb.h"

namespace serving {

static bool CheckStatus(const OrtApi* g_ort, OrtStatus* status) {
  if (status != nullptr) {
    const char* msg = g_ort->GetErrorMessage(status);
    std::cerr << msg << std::endl;
    g_ort->ReleaseStatus(status);
    throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
  }
  return true;
}

const OrtApi* Serving::g_ort;
OrtEnv* Serving::g_env;
OrtSession* Serving::g_session;
OrtAllocator* Serving::g_allocator;

void Serving::global_init(const std::string& model_path) {
  OrtSessionOptions* session_options;
  Serving::g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  CheckStatus(Serving::g_ort,
              Serving::g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "main",
                                        &Serving::g_env));
  CheckStatus(Serving::g_ort,
              Serving::g_ort->CreateSessionOptions(&session_options));
  CheckStatus(Serving::g_ort,
              Serving::g_ort->SetIntraOpNumThreads(session_options, 0));
  CheckStatus(Serving::g_ort, Serving::g_ort->SetSessionGraphOptimizationLevel(
                                  session_options, ORT_ENABLE_BASIC));
  CheckStatus(Serving::g_ort, Serving::g_ort->CreateSession(
                                  Serving::g_env, model_path.c_str(),
                                  session_options, &Serving::g_session));
  CheckStatus(Serving::g_ort, Serving::g_ort->GetAllocatorWithDefaultOptions(
                                  &Serving::g_allocator));
}

Serving::Serving(const std::string& model_path) : model_path_(model_path) {
  std::ifstream file(model_path);
  if (!file) {
    throw std::runtime_error("File does not exist");
  }
  global_init(model_path);
  CheckStatus(Serving::g_ort, Serving::g_ort->SessionGetInputCount(
                                  Serving::g_session, &this->num_input_nodes));
  CheckStatus(Serving::g_ort, Serving::g_ort->SessionGetOutputCount(
                                  Serving::g_session, &this->num_output_nodes));
  this->input_node_names.resize(this->num_input_nodes);
  for (size_t i = 0; i < this->num_input_nodes; i++) {
    // Get input node names
    char* input_name = nullptr;
    CheckStatus(Serving::g_ort,
                Serving::g_ort->SessionGetInputName(
                    Serving::g_session, i, Serving::g_allocator, &input_name));
    this->input_node_names[i] = input_name;
  }
  this->output_node_names.resize(this->num_output_nodes);
  for (size_t j = 0; j < this->num_output_nodes; j++) {
    char* output_name = nullptr;
    CheckStatus(Serving::g_ort,
                Serving::g_ort->SessionGetOutputName(
                    Serving::g_session, j, Serving::g_allocator, &output_name));
    this->output_node_names[j] = output_name;
  }
}

grpc::Status Serving::Run(grpc::ServerContext* context,
                          const inference::SessionRequest* request,
                          inference::SessionResponse* reply) {
  std::vector<OrtValue*> input_tensors;
  std::vector<OrtValue*> output_tensors;

  input_tensors.resize(this->num_input_nodes);
  output_tensors.resize(this->num_output_nodes);

  for (size_t i = 0; i < this->num_input_nodes; i++) {
    // Get input node names
    char* input_name = nullptr;
    CheckStatus(Serving::g_ort,
                Serving::g_ort->SessionGetInputName(
                    Serving::g_session, i, Serving::g_allocator, &input_name));
    bool found = false;
    const inference::Tuple* found_tuple = nullptr;
    for (const auto& tuple : request->array_map()) {
      if (strcmp(tuple.name().c_str(), input_name) == 0) {
        found_tuple = &tuple;
        found = true;
        break;
      }
    }
    if (!found) {
      return grpc::Status(
          grpc::StatusCode::INVALID_ARGUMENT,
          fmt::format("Input name '{}' not found in request", input_name));
    }

    OrtTypeInfo* type_info = nullptr;
    const OrtTensorTypeAndShapeInfo* tensor_info;
    ONNXTensorElementDataType input_type;
    size_t tensor_size;
    CheckStatus(Serving::g_ort, Serving::g_ort->SessionGetInputTypeInfo(
                                    Serving::g_session, i, &type_info));
    CheckStatus(Serving::g_ort, Serving::g_ort->CastTypeInfoToTensorInfo(
                                    type_info, &tensor_info));
    CheckStatus(Serving::g_ort, Serving::g_ort->GetTensorShapeElementCount(
                                    tensor_info, &tensor_size));
    CheckStatus(Serving::g_ort,
                Serving::g_ort->GetTensorElementType(tensor_info, &input_type));
    if (type_info) Serving::g_ort->ReleaseTypeInfo(type_info);

    if (found_tuple->values_size() != tensor_size) {
      return grpc::Status(
          grpc::StatusCode::INVALID_ARGUMENT,
          fmt::format("Invalid input size. Expected: '{}' Got: '{}'",
                      tensor_size, found_tuple->values_size()));
    }
    const ::google::protobuf::RepeatedField<float>& data_repeated =
        found_tuple->values();
    std::vector<float> data(data_repeated.begin(), data_repeated.end());
    int64_t shape[2] = {1, static_cast<int64_t>(tensor_size)};
    size_t shape_len = 2;
    OrtMemoryInfo* memory_info;
    CheckStatus(Serving::g_ort,
                Serving::g_ort->CreateCpuMemoryInfo(
                    OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    CheckStatus(Serving::g_ort,
                Serving::g_ort->CreateTensorWithDataAsOrtValue(
                    memory_info, reinterpret_cast<void*>(data.data()),
                    data.size() * sizeof(float), shape, shape_len, input_type,
                    &input_tensors[i]));
    Serving::g_ort->ReleaseMemoryInfo(memory_info);
  }
  CheckStatus(Serving::g_ort,
              Serving::g_ort->Run(
                  Serving::g_session, nullptr, this->input_node_names.data(),
                  (const OrtValue* const*)input_tensors.data(),
                  input_tensors.size(), this->output_node_names.data(),
                  this->output_node_names.size(), output_tensors.data()));

  for (size_t i = 0; i < this->num_input_nodes; i++) {
    if (input_tensors[i]) Serving::g_ort->ReleaseValue(input_tensors[i]);
  }
  for (size_t j = 0; j < this->num_output_nodes; j++) {
    inference::Tuple new_tuple;
    OrtTypeInfo* type_info = nullptr;
    const OrtTensorTypeAndShapeInfo* tensor_info;
    size_t tensor_size;
    CheckStatus(Serving::g_ort, Serving::g_ort->SessionGetOutputTypeInfo(
                                    Serving::g_session, j, &type_info));
    CheckStatus(Serving::g_ort, Serving::g_ort->CastTypeInfoToTensorInfo(
                                    type_info, &tensor_info));
    CheckStatus(Serving::g_ort, Serving::g_ort->GetTensorShapeElementCount(
                                    tensor_info, &tensor_size));

    void* output_buffer;
    CheckStatus(Serving::g_ort, Serving::g_ort->GetTensorMutableData(
                                    output_tensors[j], &output_buffer));
    float* float_buffer = reinterpret_cast<float*>(output_buffer);

    for (size_t k = 0; k < tensor_size; k++) {
      new_tuple.add_values(float_buffer[k]);
    }
    new_tuple.set_name(this->output_node_names[j]);
    reply->mutable_array_map()->Add()->CopyFrom(new_tuple);
    if (output_tensors[j]) Serving::g_ort->ReleaseValue(output_tensors[j]);
  }
  return grpc::Status::OK;
}
}  // namespace serving

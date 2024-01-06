#include "serving.h"

#include <grpcpp/grpcpp.h>
#include <onnxruntime_cxx_api.h>

#include <fstream>
#include <iostream>

#include "inference/inference.grpc.pb.h"

namespace serving {
Serving::Serving(const std::string& model_path) : model_path_(model_path) {
  std::ifstream file(model_path);
  if (!file) {
    throw std::runtime_error("File does not exist");
  }
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  this->model_content_ = content;
}

grpc::Status Serving::Run(grpc::ServerContext* context,
                          const inference::SessionRequest* request,
                          inference::SessionResponse* reply) {
  for (const auto& tuple : request->array_map()) {
    std::cout << "Tuple name: " << tuple.name() << std::endl;
    std::cout << "Length of values: " << tuple.values_size() << std::endl;
    reply->mutable_array_map()->Add()->CopyFrom(tuple);
  }
  return grpc::Status::OK;
}
}  // namespace serving

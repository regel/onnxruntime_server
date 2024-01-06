#ifndef ONNXRUNTIME_SERVING_H_
#define ONNXRUNTIME_SERVING_H_

#include <grpcpp/grpcpp.h>

#include "inference/inference.grpc.pb.h"

namespace serving {
class Serving final : public inference::Runtime::Service {
  std::string model_path_;  // Private class member for model_path
  std::string model_content_;

 public:
  explicit Serving(const std::string& model_path);
  grpc::Status Run(grpc::ServerContext* context,
                   const inference::SessionRequest* request,
                   inference::SessionResponse* reply);
};
}  // namespace serving

#endif

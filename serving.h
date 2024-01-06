#ifndef ONNXRUNTIME_SERVING_H_
#define ONNXRUNTIME_SERVING_H_

#include <grpcpp/grpcpp.h>
#include <onnxruntime_cxx_api.h>

#include "inference/inference.grpc.pb.h"

namespace serving {

class Serving final : public inference::Runtime::Service {
 private:
  std::string model_path_;  // Private class member for model_path
  static const OrtApi* g_ort;
  static OrtEnv* g_env;
  static OrtSession* g_session;
  static OrtAllocator* g_allocator;

  static void global_init(const std::string& model_path);

 public:
  explicit Serving(const std::string& model_path);
  grpc::Status Run(grpc::ServerContext* context,
                   const inference::SessionRequest* request,
                   inference::SessionResponse* reply);
};
}  // namespace serving

#endif

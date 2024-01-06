#include <absl/base/log_severity.h>
#include <absl/flags/flag.h>
#include <absl/flags/marshalling.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/strings/string_view.h>
#include <fmt/format.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "serving.h"

struct PortNumber {
  explicit PortNumber(int p = 0) : port(p) {}

  int port;  // Valid range is [0..65535]
};

// Returns a textual flag value corresponding to the PortNumber `p`.
std::string AbslUnparseFlag(PortNumber p) {
  // Delegate to the usual unparsing for int.
  return absl::UnparseFlag(p.port);
}

// Parses a PortNumber from the command line flag value `text`.
// Returns true and sets `*p` on success; returns false and sets `*error`
// on failure.
bool AbslParseFlag(absl::string_view text, PortNumber* p, std::string* error) {
  // Convert from text to int using the int-flag parser.
  if (!absl::ParseFlag(text, &p->port, error)) {
    return false;
  }
  if (p->port < 0 || p->port > 65535) {
    *error = "not in range [0,65535]";
    return false;
  }
  return true;
}

struct RequiredFile {
  explicit RequiredFile(std::string s = "") : path(s) {}

  std::string path;
};

std::string AbslUnparseFlag(RequiredFile s) {
  return absl::UnparseFlag(s.path);
}

bool AbslParseFlag(absl::string_view text, RequiredFile* s,
                   std::string* error) {
  if (!absl::ParseFlag(text, &s->path, error)) {
    return false;
  }
  if (s->path.empty()) {
    *error = "option is required but missing";
    return false;
  }
  if (!std::ifstream(s->path)) {
    *error = fmt::format("File at '{}' does not exist", s->path);
    return false;
  }
  return true;
}

ABSL_FLAG(absl::optional<RequiredFile>, model_path, absl::nullopt,
          "Path to ONNX model");
ABSL_FLAG(absl::LogSeverity, log_level, absl::LogSeverity::kInfo,
          "Logging level. Allowed options (case sensitive): info, warning, "
          "error, fatal");
ABSL_FLAG(std::string, address, "0.0.0.0", "The base server address");
ABSL_FLAG(int, num_threads, 0, "Number of server threads");
ABSL_FLAG(PortNumber, grpc_port, PortNumber(50051),
          "GRPC port to listen to requests");

void RunServer(const std::string& address, const int port,
               const int num_threads, const std::string& model_path) {
  std::string server_address = fmt::format("{}:{}", address, port);
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();

  serving::Serving service(model_path);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  grpc::ResourceQuota quota;
  quota.SetMaxThreads(num_threads);
  builder.SetResourceQuota(quota);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << fmt::format("Server listening on {}:{}", server_address, port)
            << std::endl;

  server->Wait();
}

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("./onnxruntime_server --model_path trained.pt");
  absl::ParseCommandLine(argc, argv);
  QCHECK(absl::GetFlag(FLAGS_model_path).has_value())
      << "the option '--model_path' is required but missing";

  const std::string server_address =
      absl::GetFlag(FLAGS_address);  // Get server address using absl flag
  const PortNumber port_number =
      absl::GetFlag(FLAGS_grpc_port);  // Get port value using absl flag
  const std::string model_path = absl::GetFlag(FLAGS_model_path).value().path;

  int num_threads = absl::GetFlag(FLAGS_num_threads);
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }
  RunServer(server_address, port_number.port, num_threads, model_path);
  return 0;
}

find_package(Threads)
find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

#
# Protobuf/Grpc source files
#
set(PROTO_FILES
    inference/inference.proto
)

#
# Add Library target with protobuf sources
#
add_library(inference_grpc_proto ${PROTO_FILES})
target_link_libraries(inference_grpc_proto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc
        gRPC::grpc++
        gRPC::grpc++_reflection
)
target_include_directories(inference_grpc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

#
# Compile protobuf and grpc files in inference target to cpp
#
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET inference_grpc_proto LANGUAGE cpp)
protobuf_generate(TARGET inference_grpc_proto LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")

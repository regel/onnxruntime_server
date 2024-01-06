cmake_minimum_required(VERSION 3.10)
project(onnxruntime_server LANGUAGES C CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (APPLE AND NOT DEFINED CMAKE_CXX_STANDARD)
  message("CMAKE_CXX_STANDARD was undefined, defaulting to C++14.")
  set(CMAKE_CXX_STANDARD 14)
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(./third_party/fmt EXCLUDE_FROM_ALL)
add_subdirectory(proto)  # builds inference_grpc_proto library

find_package(Threads REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(ONNXRuntime REQUIRED)

message(STATUS "Using protobuf ${Protobuf_VERSION}")
message(STATUS "Using gRPC ${gRPC_VERSION}")

include_directories(
  "${ONNXRuntime_INCLUDE_DIR}"
)

add_library(onnxruntime_serving
    serving.cc
    serving.h
)
target_link_libraries(onnxruntime_serving
        inference_grpc_proto
        fmt::fmt-header-only
        protobuf::libprotobuf
        gRPC::grpc
        gRPC::grpc++
        gRPC::grpc++_reflection
        ${ONNXRuntime_LIBRARIES}
)

set(SOURCES
    onnxruntime_server.cc
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_executable(onnxruntime_server ${SOURCES})
target_link_libraries(onnxruntime_server
        inference_grpc_proto
        onnxruntime_serving
        absl::base
        absl::flags
        absl::flags_parse
        absl::strings
        fmt::fmt-header-only
        protobuf::libprotobuf
        gRPC::grpc
        gRPC::grpc++
        gRPC::grpc++_reflection
        ${ONNXRuntime_LIBRARIES}
)

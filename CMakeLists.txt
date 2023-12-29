cmake_minimum_required(VERSION 3.10)
project(onnxruntime_server)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(./third_party/fmt EXCLUDE_FROM_ALL)
add_subdirectory(proto)
# Add onnxruntime library
# add_subdirectory(./onnxruntime/cmake)

#include_directories(/usr/local/include)

find_package(Threads REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

message(STATUS "Using protobuf ${Protobuf_VERSION}")
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(SOURCES
    onnxruntime_server.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_executable(onnxruntime_server ${SOURCES})
target_link_libraries(onnxruntime_server
    PRIVATE
        inference
        absl::base absl::flags
        absl::flags_parse
        absl::strings
	#        onnxruntime
        fmt::fmt-header-only
        protobuf::libprotobuf
        gRPC::grpc
        gRPC::grpc++
        gRPC::grpc++_reflection
)

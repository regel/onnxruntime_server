cmake_minimum_required(VERSION 3.10)
project(onnxruntime_server)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(proto)
add_subdirectory(./abseil-cpp)
add_subdirectory(./fmt EXCLUDE_FROM_ALL)
# Add onnxruntime library
# add_subdirectory(./onnxruntime/cmake)

include_directories(/usr/local/include)


set(SOURCES
    onnxruntime_server.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_executable(onnxruntime_server ${SOURCES})
target_link_directories(onnxruntime_server
    PUBLIC
        /usr/local/lib
)
target_link_libraries(onnxruntime_server
    PRIVATE
        inference
    PUBLIC
        absl::base absl::flags
        absl::flags_parse
        absl::strings
        onnxruntime
        fmt::fmt-header-only
)

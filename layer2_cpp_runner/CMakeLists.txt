cmake_minimum_required(VERSION 3.22)

# Expects ONNX Runtime to be provided via ONNXRUNTIME_ROOT cache variable.
# Example: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
find_library(ORT_LIB onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib")
find_path(ORT_INCLUDE onnxruntime_cxx_api.h HINTS "${ONNXRUNTIME_ROOT}/include")

add_executable(runner main.cpp)
target_include_directories(runner PRIVATE ${ORT_INCLUDE})
target_link_libraries(runner PRIVATE ${ORT_LIB})

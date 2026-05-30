# Layer 5: Android cross-compilation target.
#
# Usage (from repo root, with Android NDK installed):
#   cmake -B build-android \
#         -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#         -DANDROID_ABI=arm64-v8a \
#         -DANDROID_PLATFORM=android-29 \
#         -DONNXRUNTIME_ROOT=/path/to/onnxruntime-android \
#         layer5_android
#   cmake --build build-android

cmake_minimum_required(VERSION 3.22)
project(edge-ai-bench-android CXX)

set(CMAKE_CXX_STANDARD 17)

find_library(ORT_LIB onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib")
find_path(ORT_INCLUDE onnxruntime_cxx_api.h HINTS "${ONNXRUNTIME_ROOT}/include")

add_executable(runner_android ../layer2_cpp_runner/main.cpp)
target_include_directories(runner_android PRIVATE ${ORT_INCLUDE})
target_link_libraries(runner_android PRIVATE ${ORT_LIB} log)

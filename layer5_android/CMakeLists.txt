# Layer 5: Android ARM64 cross-compilation
#
# Build with the Android NDK toolchain file.
# Usage (from repo root):
#
#   cmake -B build-android \
#         -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#         -DANDROID_ABI=arm64-v8a \
#         -DANDROID_PLATFORM=android-29 \
#         -DONNXRUNTIME_ROOT=/path/to/onnxruntime-android \
#         layer5_android
#   cmake --build build-android
#
# Or compile directly with the NDK clang++ (no CMake required):
#
#   $CLANG = "$NDK\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android29-clang++"
#   & $CLANG -std=c++17 -O2 -o runner_android layer2_cpp_runner/main.cpp \
#       -I$ORT_ROOT/headers \
#       -L$ORT_ROOT/libs/arm64-v8a -lonnxruntime -llog \
#       -Wl,-rpath,'$ORIGIN'

cmake_minimum_required(VERSION 3.22)
project(edge-ai-bench-android CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(ORT_LIB onnxruntime HINTS "${ONNXRUNTIME_ROOT}/libs/arm64-v8a")
find_path(ORT_INCLUDE onnxruntime_cxx_api.h HINTS "${ONNXRUNTIME_ROOT}/headers")

add_executable(runner_android ../layer2_cpp_runner/main.cpp)
target_include_directories(runner_android PRIVATE ${ORT_INCLUDE})
target_link_libraries(runner_android PRIVATE ${ORT_LIB} log)

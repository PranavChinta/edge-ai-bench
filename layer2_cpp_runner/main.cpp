#include <onnxruntime_cxx_api.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

int main(int argc, char* argv[]) {
    const char* model_path = argc > 1 ? argv[1] : "model.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge-ai-bench");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);

    Ort::Session session(env, model_path, opts);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name  = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);

    constexpr int64_t batch = 1, features = 128;
    std::array<int64_t, 2> input_shape{batch, features};
    std::vector<float> input_data(batch * features, 1.0f);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[]  = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    // Warm-up
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    constexpr int RUNS = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; ++i)
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;
    std::printf("Model: %s\n", model_path);
    std::printf("Avg latency over %d runs: %.3f ms\n", RUNS, ms);
    return 0;
}

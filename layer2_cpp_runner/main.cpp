#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#  include <psapi.h>
static double peak_rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    return -1.0;
}
#else
#  include <sys/resource.h>
static double peak_rss_mb() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#  ifdef __APPLE__
    return ru.ru_maxrss / (1024.0 * 1024.0);
#  else
    return ru.ru_maxrss / 1024.0;
#  endif
}
#endif

int main(int argc, char* argv[]) {
    const std::string model_path = argc > 1 ? argv[1] : "model.onnx";
    constexpr int WARMUP = 1;
    constexpr int RUNS   = 100;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge-ai-bench");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    printf("Loading model: %s\n", model_path.c_str());
    Ort::Session session(env, model_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc  = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[]  = {input_name_alloc.get()};
    const char* output_names[] = {output_name_alloc.get()};

    // Build input tensor: batch=1, features=128
    constexpr int64_t BATCH = 1, FEATURES = 128;
    std::array<int64_t, 2> input_shape{BATCH, FEATURES};
    std::vector<float> input_data(BATCH * FEATURES, 1.0f);

    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto make_tensor = [&]() {
        return Ort::Value::CreateTensor<float>(
            mem_info,
            input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());
    };

    // Warm-up passes (not timed)
    for (int i = 0; i < WARMUP; ++i) {
        auto t = make_tensor();
        session.Run(Ort::RunOptions{nullptr}, input_names, &t, 1, output_names, 1);
    }

    // Timed passes
    std::vector<double> latencies(RUNS);
    for (int i = 0; i < RUNS; ++i) {
        auto t  = make_tensor();
        auto t0 = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, input_names, &t, 1, output_names, 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double sum  = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / RUNS;
    double mn   = *std::min_element(latencies.begin(), latencies.end());
    double mx   = *std::max_element(latencies.begin(), latencies.end());

    // P50 / P95 / P99
    std::vector<double> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) { return sorted[static_cast<int>(p / 100.0 * (RUNS - 1))]; };

    double rss = peak_rss_mb();

    printf("\n=== Inference Benchmark ===\n");
    printf("Model         : %s\n", model_path.c_str());
    printf("Input shape   : [%lld, %lld]\n", (long long)BATCH, (long long)FEATURES);
    printf("Runs          : %d (+ %d warm-up)\n", RUNS, WARMUP);
    printf("\nLatency (ms):\n");
    printf("  Mean        : %.3f\n", mean);
    printf("  Min         : %.3f\n", mn);
    printf("  Max         : %.3f\n", mx);
    printf("  P50         : %.3f\n", pct(50));
    printf("  P95         : %.3f\n", pct(95));
    printf("  P99         : %.3f\n", pct(99));
    if (rss > 0)
        printf("\nPeak RSS (MB) : %.2f\n", rss);

    return 0;
}

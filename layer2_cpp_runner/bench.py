"""
Layer 2 Python benchmark — functional mirror of main.cpp.

Uses the ONNX Runtime Python API (same underlying C++ engine) to measure
inference latency and peak RSS. Run this on any platform; build main.cpp
with MSVC or the Android NDK for native deployment.

Usage:
    python layer2_cpp_runner/bench.py [model.onnx]
"""

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import psutil


WARMUP = 1
RUNS   = 100


def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    idx = int(p / 100.0 * (len(s) - 1))
    return s[idx]


def peak_rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    info = proc.memory_info()
    return info.rss / (1024 ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="model.onnx")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # suppress warnings

    print(f"Loading model: {args.model}")
    sess = ort.InferenceSession(args.model, sess_options=so)

    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    # Shape is inferred from the model's input metadata
    input_meta = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_meta.shape]
    dummy = np.ones(shape, dtype=np.float32)
    batch, features = shape[0], shape[-1]

    # Warm-up
    for _ in range(WARMUP):
        sess.run([output_name], {input_name: dummy})

    # Timed runs
    rss_before = peak_rss_mb()
    latencies: list[float] = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run([output_name], {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1e3)
    rss_after = peak_rss_mb()

    mean_ms = sum(latencies) / RUNS
    min_ms  = min(latencies)
    max_ms  = max(latencies)

    print(f"\n=== Inference Benchmark ===")
    print(f"Model         : {args.model}")
    print(f"ORT version   : {ort.__version__}")
    print(f"Input shape   : {shape}")
    print(f"Runs          : {RUNS} (+ {WARMUP} warm-up)")
    print(f"\nLatency (ms):")
    print(f"  Mean        : {mean_ms:.3f}")
    print(f"  Min         : {min_ms:.3f}")
    print(f"  Max         : {max_ms:.3f}")
    print(f"  P50         : {percentile(latencies, 50):.3f}")
    print(f"  P95         : {percentile(latencies, 95):.3f}")
    print(f"  P99         : {percentile(latencies, 99):.3f}")
    print(f"\nPeak RSS (MB) : {rss_after:.2f}  (delta: +{rss_after - rss_before:.2f})")


if __name__ == "__main__":
    main()

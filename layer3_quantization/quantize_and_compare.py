"""
Layer 3: INT8 dynamic quantization of MobileNetV2 ONNX model + full benchmark comparison.
"""

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import psutil
from onnxruntime.quantization import quantize_dynamic, QuantType

WARMUP = 100
RUNS   = 100


def percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    return s[int(p / 100.0 * (len(s) - 1))]


def peak_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def make_session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=so)


def infer_input_shape(sess: ort.InferenceSession) -> list[int]:
    return [d if isinstance(d, int) and d > 0 else 1
            for d in sess.get_inputs()[0].shape]


def benchmark(sess: ort.InferenceSession) -> dict:
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    dummy = np.ones(infer_input_shape(sess), dtype=np.float32)

    for _ in range(WARMUP):
        sess.run([output_name], {input_name: dummy})

    rss_before = peak_rss_mb()
    latencies: list[float] = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run([output_name], {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1e3)
    rss_peak = peak_rss_mb()

    return {
        "mean": sum(latencies) / RUNS,
        "min":  min(latencies),
        "max":  max(latencies),
        "p95":  percentile(latencies, 95),
        "p99":  percentile(latencies, 99),
        "rss":  rss_peak,
    }


def accuracy_delta(fp32_sess: ort.InferenceSession,
                   int8_sess: ort.InferenceSession,
                   n_samples: int = 100) -> float:
    rng = np.random.default_rng(42)
    fp32_in  = fp32_sess.get_inputs()[0].name
    int8_in  = int8_sess.get_inputs()[0].name
    fp32_out = fp32_sess.get_outputs()[0].name
    int8_out = int8_sess.get_outputs()[0].name
    shape = infer_input_shape(fp32_sess)

    diffs = []
    for _ in range(n_samples):
        x = rng.standard_normal(shape).astype(np.float32)
        a = fp32_sess.run([fp32_out], {fp32_in: x})[0]
        b = int8_sess.run([int8_out], {int8_in: x})[0]
        diffs.append(float(np.abs(a - b).mean()))
    return float(np.mean(diffs))


def fmt_speedup(a: float, b: float) -> str:
    return f"{a / b:.2f}x"


def print_table(fp32: dict, int8: dict) -> None:
    rows = [
        ("Mean (ms)",    f"{fp32['mean']:.3f}", f"{int8['mean']:.3f}", fmt_speedup(fp32['mean'], int8['mean'])),
        ("Min (ms)",     f"{fp32['min']:.3f}",  f"{int8['min']:.3f}",  fmt_speedup(fp32['min'],  int8['min'])),
        ("P95 (ms)",     f"{fp32['p95']:.3f}",  f"{int8['p95']:.3f}",  fmt_speedup(fp32['p95'],  int8['p95'])),
        ("P99 (ms)",     f"{fp32['p99']:.3f}",  f"{int8['p99']:.3f}",  fmt_speedup(fp32['p99'],  int8['p99'])),
        ("Peak RSS(MB)", f"{fp32['rss']:.2f}",  f"{int8['rss']:.2f}",  fmt_speedup(fp32['rss'],  int8['rss'])),
    ]
    w = [14, 10, 10, 10]
    sep = "+" + "+".join("-" * (c + 2) for c in w) + "+"
    hdr = "| {:<{}} | {:<{}} | {:<{}} | {:<{}} |".format(
        "Metric", w[0], "fp32", w[1], "int8", w[2], "speedup", w[3])
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        print("| {:<{}} | {:<{}} | {:<{}} | {:<{}} |".format(
            r[0], w[0], r[1], w[1], r[2], w[2], r[3], w[3]))
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="model.onnx")
    parser.add_argument("--quantized", default="model_int8.onnx")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}")

    # --- Quantize ---
    print(f"Quantizing {args.model} -> {args.quantized} ...")
    quantize_dynamic(args.model, args.quantized, weight_type=QuantType.QInt8)
    fp32_mb = os.path.getsize(args.model)      / (1024 ** 2)
    int8_mb = os.path.getsize(args.quantized)  / (1024 ** 2)
    print(f"  FP32 size : {fp32_mb:.1f} MB")
    print(f"  INT8 size : {int8_mb:.1f} MB  ({fp32_mb/int8_mb:.2f}x smaller)")

    # --- Sessions ---
    fp32_sess = make_session(args.model)
    int8_sess = make_session(args.quantized)

    # --- Benchmark ---
    print(f"\nBenchmarking FP32  ({WARMUP} warm-up + {RUNS} timed runs) ...")
    fp32_stats = benchmark(fp32_sess)
    print(f"Benchmarking INT8  ({WARMUP} warm-up + {RUNS} timed runs) ...")
    int8_stats = benchmark(int8_sess)

    # --- Accuracy delta ---
    print("\nMeasuring accuracy delta (100 random inputs) ...")
    delta = accuracy_delta(fp32_sess, int8_sess)

    # --- Report ---
    print("\n=== FP32 vs INT8 Comparison ===")
    print_table(fp32_stats, int8_stats)
    print(f"\nMean absolute logit delta (fp32 vs int8): {delta:.6f}")
    print(f"Model size reduction: {fp32_mb:.1f} MB -> {int8_mb:.1f} MB ({fp32_mb/int8_mb:.2f}x smaller)")
    print("\nNote: dynamic quantization quantizes weights only; activations are cast at runtime.")
    print("For Conv2d-heavy nets like MobileNetV2 this adds overhead vs FP32 SIMD kernels.")
    print("Static (QDQ) quantization with a calibration dataset is required for CNN speedup.")


if __name__ == "__main__":
    main()

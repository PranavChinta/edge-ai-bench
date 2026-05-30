"""
Layer 4: Roofline model for MobileNetV2 inference on Intel i7 (AVX2).

Benchmark inputs (from Layers 2 & 3):
  fp32: mean 7.363 ms, model 13.3 MB
  int8: mean 151.303 ms, model 3.5 MB

Hardware ceiling (Intel i7, AVX2 float32, single thread):
  Peak compute : 192 GFLOP/s  (8 FP32 floats/cycle * 2 FMA * 4 GHz * 3 cores ~ 192)
  Peak BW      :  45 GB/s     (LPDDR4 dual-channel measured)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Hardware parameters
# ---------------------------------------------------------------------------
PEAK_COMPUTE_GFLOPS = 192.0   # AVX2 FP32 theoretical peak, single socket
PEAK_BW_GBS         = 45.0    # measured LPDDR4 dual-channel bandwidth (GB/s)
RIDGE_POINT         = PEAK_COMPUTE_GFLOPS / PEAK_BW_GBS  # FLOP / byte

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
FLOPS = 300e6   # standard MobileNetV2 figure for one forward pass (300 MFLOPs)

FP32_MODEL_BYTES = 13.3 * 1024 ** 2
INT8_MODEL_BYTES =  3.5 * 1024 ** 2

# Arithmetic intensity = FLOP / bytes moved
AI_FP32 = FLOPS / FP32_MODEL_BYTES
AI_INT8  = FLOPS / INT8_MODEL_BYTES

# ---------------------------------------------------------------------------
# Measured benchmark results (Layers 2 & 3, mean latency)
# ---------------------------------------------------------------------------
FP32_MEAN_MS  =   7.363
INT8_MEAN_MS  = 151.303

# Attainable performance = FLOPs / time
PERF_FP32_GFLOPS = FLOPS / (FP32_MEAN_MS * 1e-3) / 1e9
PERF_INT8_GFLOPS = FLOPS / (INT8_MEAN_MS * 1e-3) / 1e9

# ---------------------------------------------------------------------------
# Roofline ceiling function
# ---------------------------------------------------------------------------
def roofline(ai: np.ndarray) -> np.ndarray:
    return np.minimum(PEAK_BW_GBS * ai, PEAK_COMPUTE_GFLOPS)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(out_path: str) -> None:
    ai_range = np.logspace(-2, 4, 1000)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Roofline ceiling
    ax.plot(ai_range, roofline(ai_range), color="black", linewidth=2.0,
            label=f"Roofline ceiling")

    # Memory-bandwidth ceiling (slope line, extended)
    ax.plot(ai_range, PEAK_BW_GBS * ai_range, color="steelblue",
            linewidth=1.2, linestyle="--", label=f"Mem BW limit ({PEAK_BW_GBS} GB/s)")

    # Compute ceiling (horizontal)
    ax.axhline(PEAK_COMPUTE_GFLOPS, color="firebrick",
               linewidth=1.2, linestyle="--",
               label=f"Compute limit ({PEAK_COMPUTE_GFLOPS:.0f} GFLOP/s, AVX2 FP32)")

    # Ridge point
    ax.axvline(RIDGE_POINT, color="gray", linewidth=1.0, linestyle=":",
               label=f"Ridge point ({RIDGE_POINT:.2f} FLOP/byte)")

    # FP32 operating point
    ax.scatter(AI_FP32, PERF_FP32_GFLOPS, s=100, color="darkorange",
               zorder=5, label=f"fp32  AI={AI_FP32:.2f}  {PERF_FP32_GFLOPS:.2f} GFLOP/s")
    ax.annotate(
        f"fp32\n{PERF_FP32_GFLOPS:.2f} GFLOP/s",
        (AI_FP32, PERF_FP32_GFLOPS),
        xytext=(12, 6), textcoords="offset points",
        fontsize=8, color="darkorange",
    )

    # INT8 operating point
    ax.scatter(AI_INT8, PERF_INT8_GFLOPS, s=100, color="seagreen",
               zorder=5, label=f"int8  AI={AI_INT8:.2f}  {PERF_INT8_GFLOPS:.3f} GFLOP/s")
    ax.annotate(
        f"int8\n{PERF_INT8_GFLOPS:.3f} GFLOP/s",
        (AI_INT8, PERF_INT8_GFLOPS),
        xytext=(12, -18), textcoords="offset points",
        fontsize=8, color="seagreen",
    )

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11)
    ax.set_ylabel("Attainable Performance (GFLOP/s)", fontsize=11)
    ax.set_title("MobileNetV2 Roofline — CPU (Intel i7, AVX2)", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Chart saved: {out_path}  ({os.path.getsize(out_path)//1024} KB)")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def print_summary() -> None:
    def bound(ai: float) -> str:
        return "memory-bound" if ai < RIDGE_POINT else "compute-bound"

    print("=== Roofline Summary ===")
    print(f"Hardware : Intel i7, AVX2 FP32 single-thread")
    print(f"  Peak compute : {PEAK_COMPUTE_GFLOPS:.0f} GFLOP/s")
    print(f"  Peak BW      : {PEAK_BW_GBS:.0f} GB/s")
    print(f"  Ridge point  : {RIDGE_POINT:.2f} FLOP/byte")
    print()
    print(f"MobileNetV2 FLOPs (1 forward pass) : {FLOPS/1e6:.0f} MFLOPs")
    print()
    print(f"{'':4} {'AI (FLOP/B)':>14} {'Perf (GFLOP/s)':>16} {'Bound':>14} {'Latency':>10}")
    print(f"{'':4} {'-'*14} {'-'*16} {'-'*14} {'-'*10}")
    for label, ai, perf, lat in [
        ("fp32", AI_FP32, PERF_FP32_GFLOPS, FP32_MEAN_MS),
        ("int8", AI_INT8, PERF_INT8_GFLOPS, INT8_MEAN_MS),
    ]:
        print(f"  {label:4} {ai:>14.4f} {perf:>16.4f} {bound(ai):>14} {lat:>8.3f} ms")


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "roofline.png")
    plot(out)
    print()
    print_summary()


if __name__ == "__main__":
    main()

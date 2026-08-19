"""
Layer 4 (phone variant): Roofline model for MobileNetV2 inference on
Samsung Galaxy A23 5G (Snapdragon 695, Kryo 660 / Cortex-A78, NEON).

Benchmark inputs (from layer5_android, single-threaded ORT session):
  fp32:            mean 36.208 ms, model 13.3 MB
  int8 (QDQ static): mean 12.810 ms, model 3.5 MB
    (dynamic quantization, used on desktop, crashes on this ORT 1.21.0
     Android build: ConvInteger has no CPU kernel registered. Static QDQ
     quantization emits QLinearConv nodes instead, which run fine.)

Hardware ceiling (Snapdragon 695, Cortex-A78, single core, NEON float32):
  Peak compute : 35.2 GFLOP/s  ESTIMATED — 2 NEON FMA pipelines * 4 FP32
                 lanes (128-bit) * 2 FLOP/FMA * 2.2 GHz, 1 core (ORT runs
                 with intra_op_num_threads=1, same as the desktop runner).
                 ARM has not published an official per-core FP32 GFLOP/s
                 figure for A78; this follows the same lanes*FMA*clock
                 method used for the desktop AVX2 figure, applied to a
                 single A78 "gold" core. Treat as an estimate, not a
                 vendor-quoted spec.
  Peak BW      : 17.0 GB/s     Snapdragon 695 LPDDR4X @ 4266 MT/s, vendor/
                 aggregator-quoted peak (not independently measured on
                 this device, unlike the desktop's measured 45 GB/s).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Hardware parameters — ESTIMATED, see module docstring for assumptions/sources
# ---------------------------------------------------------------------------
PEAK_COMPUTE_GFLOPS = 35.2   # Cortex-A78 NEON FP32, 1 core, 2.2 GHz (estimated)
PEAK_BW_GBS         = 17.0   # Snapdragon 695 LPDDR4X @ 4266 MT/s (vendor-quoted)
RIDGE_POINT         = PEAK_COMPUTE_GFLOPS / PEAK_BW_GBS  # FLOP / byte

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
FLOPS = 300e6   # standard MobileNetV2 figure for one forward pass (300 MFLOPs)

FP32_MODEL_BYTES = 13.3 * 1024 ** 2
INT8_MODEL_BYTES =  3.5 * 1024 ** 2   # static QDQ INT8 size

AI_FP32 = FLOPS / FP32_MODEL_BYTES
AI_INT8 = FLOPS / INT8_MODEL_BYTES

# ---------------------------------------------------------------------------
# Measured benchmark results (layer5_android, mean latency)
# ---------------------------------------------------------------------------
FP32_MEAN_MS =  36.208   # dynamic-quantization-crashes on this ORT build; N/A
INT8_MEAN_MS =  12.810   # static QDQ INT8, measured

PERF_FP32_GFLOPS = FLOPS / (FP32_MEAN_MS * 1e-3) / 1e9
PERF_INT8_GFLOPS = FLOPS / (INT8_MEAN_MS * 1e-3) / 1e9


def roofline(ai: np.ndarray) -> np.ndarray:
    return np.minimum(PEAK_BW_GBS * ai, PEAK_COMPUTE_GFLOPS)


def plot(out_path: str) -> None:
    ai_range = np.logspace(-2, 4, 1000)

    bg        = "#1e1e1e"
    fg        = "#e8e8e8"
    grid_col  = "#555555"

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(ai_range, roofline(ai_range), color=fg, linewidth=2.0,
            label="Roofline ceiling")

    ax.plot(ai_range, PEAK_BW_GBS * ai_range, color="#6ab0f3",
            linewidth=1.2, linestyle="--", label=f"Mem BW limit ({PEAK_BW_GBS:.1f} GB/s, est.)")

    ax.axhline(PEAK_COMPUTE_GFLOPS, color="#f07070",
               linewidth=1.2, linestyle="--",
               label=f"Compute limit ({PEAK_COMPUTE_GFLOPS:.1f} GFLOP/s, NEON FP32 1-core, est.)")

    ax.axvline(RIDGE_POINT, color="#aaaaaa", linewidth=1.0, linestyle=":",
               label=f"Ridge point ({RIDGE_POINT:.2f} FLOP/byte)")

    ax.scatter(AI_FP32, PERF_FP32_GFLOPS, s=100, color="darkorange",
               zorder=5, label=f"fp32  AI={AI_FP32:.2f}  {PERF_FP32_GFLOPS:.2f} GFLOP/s")
    ax.annotate(
        f"fp32\n{PERF_FP32_GFLOPS:.2f} GFLOP/s",
        (AI_FP32, PERF_FP32_GFLOPS),
        xytext=(12, 6), textcoords="offset points",
        fontsize=8, color="darkorange",
    )

    ax.scatter(AI_INT8, PERF_INT8_GFLOPS, s=100, color="mediumseagreen",
               zorder=5, label=f"int8 (QDQ)  AI={AI_INT8:.2f}  {PERF_INT8_GFLOPS:.2f} GFLOP/s")
    ax.annotate(
        f"int8 (QDQ)\n{PERF_INT8_GFLOPS:.2f} GFLOP/s",
        (AI_INT8, PERF_INT8_GFLOPS),
        xytext=(12, -18), textcoords="offset points",
        fontsize=8, color="mediumseagreen",
    )

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11, color=fg)
    ax.set_ylabel("Attainable Performance (GFLOP/s)", fontsize=11, color=fg)
    ax.set_title("MobileNetV2 Roofline — Phone (Snapdragon 695, NEON, estimated HW ceiling)", fontsize=12, color=fg)
    legend = ax.legend(fontsize=8, loc="upper left", facecolor=bg, edgecolor=grid_col)
    for text in legend.get_texts():
        text.set_color(fg)
    ax.grid(True, which="both", alpha=0.25, color=grid_col)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.tick_params(colors=fg, which="both")
    for spine in ax.spines.values():
        spine.set_color(grid_col)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=bg)
    plt.close()
    print(f"Chart saved: {out_path}  ({os.path.getsize(out_path)//1024} KB)")


def print_summary() -> None:
    def bound(ai: float) -> str:
        return "memory-bound" if ai < RIDGE_POINT else "compute-bound"

    print("=== Roofline Summary (Phone, Snapdragon 695) ===")
    print("Hardware : Cortex-A78 NEON FP32, 1 core (ESTIMATED, see script docstring)")
    print(f"  Peak compute : {PEAK_COMPUTE_GFLOPS:.1f} GFLOP/s")
    print(f"  Peak BW      : {PEAK_BW_GBS:.1f} GB/s")
    print(f"  Ridge point  : {RIDGE_POINT:.2f} FLOP/byte")
    print()
    print(f"MobileNetV2 FLOPs (1 forward pass) : {FLOPS/1e6:.0f} MFLOPs")
    print()
    print(f"{'':4} {'AI (FLOP/B)':>14} {'Perf (GFLOP/s)':>16} {'Bound':>14} {'Latency':>10}")
    print(f"{'':4} {'-'*14} {'-'*16} {'-'*14} {'-'*10}")
    for label, ai, perf, lat in [
        ("fp32", AI_FP32, PERF_FP32_GFLOPS, FP32_MEAN_MS),
        ("int8 (QDQ)", AI_INT8, PERF_INT8_GFLOPS, INT8_MEAN_MS),
    ]:
        print(f"  {label:10} {ai:>14.4f} {perf:>16.4f} {bound(ai):>14} {lat:>8.3f} ms")


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "roofline_phone.png")
    plot(out)
    print()
    print_summary()


if __name__ == "__main__":
    main()

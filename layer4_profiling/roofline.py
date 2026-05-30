"""
Layer 4: Roofline model for the MLP — plots compute vs. memory bandwidth limits.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


# MLP: 128→256→128→10
LAYERS = [
    {"name": "Linear(128,256)", "flops": 2 * 128 * 256, "bytes": (128 * 256 + 256 + 128 + 256) * 4},
    {"name": "Linear(256,128)", "flops": 2 * 256 * 128, "bytes": (256 * 128 + 128 + 256 + 128) * 4},
    {"name": "Linear(128,10)",  "flops": 2 * 128 * 10,  "bytes": (128 * 10 + 10 + 128 + 10) * 4},
]


def roofline(peak_flops_gflops: float, peak_bw_gbps: float):
    ridge = peak_flops_gflops / peak_bw_gbps  # FLOP/byte

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.logspace(-1, 3, 400)
    roof = np.minimum(peak_bw_gbps * x, peak_flops_gflops)
    ax.loglog(x, roof, "k-", linewidth=2, label="Roofline")
    ax.axvline(ridge, color="gray", linestyle="--", linewidth=1, label=f"Ridge = {ridge:.1f} FLOP/B")

    for layer in LAYERS:
        ai = layer["flops"] / layer["bytes"]
        perf = min(peak_bw_gbps * ai, peak_flops_gflops)
        ax.scatter(ai, perf, s=80, zorder=5)
        ax.annotate(layer["name"], (ai, perf), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Roofline — MLP edge inference")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = "roofline.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-flops", type=float, default=8.0,
                        help="Peak compute in GFLOP/s (default: 8 for a mid-range Cortex-A CPU)")
    parser.add_argument("--peak-bw", type=float, default=25.6,
                        help="Peak memory bandwidth in GB/s (default: 25.6 for LPDDR5)")
    args = parser.parse_args()
    roofline(args.peak_flops, args.peak_bw)


if __name__ == "__main__":
    main()

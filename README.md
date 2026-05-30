# edge-ai-bench

End-to-end benchmarking suite for neural network inference on constrained edge hardware. The project walks a small MLP from PyTorch training through ONNX export, C++ runtime execution, INT8 post-training quantization, roofline performance modeling, and cross-compilation for Android — demonstrating the full stack of skills required to ship ML models on resource-limited devices such as mobile SoCs, DSPs, and microcontrollers.

## Layers

| # | Directory | Description | Status |
|---|-----------|-------------|--------|
| 1 | `layer1_export/` | Define a PyTorch MLP and export it to ONNX | ✅ Done |
| 2 | `layer2_cpp_runner/` | Load and run the ONNX model with ONNX Runtime in C++ | 🔄 In progress |
| 3 | `layer3_quantization/` | Apply INT8 post-training quantization and compare accuracy | 🔄 In progress |
| 4 | `layer4_profiling/` | Build a roofline model to identify compute vs. memory bottlenecks | 🔄 In progress |
| 5 | `layer5_android/` | Cross-compile the C++ runner for Android (ARM64) via CMake | 🔄 In progress |

## Quick start

```bash
# Layer 1 — export model to ONNX
cd edge-ai-bench
python layer1_export/export_model.py
# produces model.onnx in the working directory
```

## Requirements

- Python 3.9+, PyTorch 2.x, onnx, onnxruntime
- CMake 3.22+, a C++17 compiler
- Android NDK r25+ (Layer 5 only)

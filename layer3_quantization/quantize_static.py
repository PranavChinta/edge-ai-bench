"""
Layer 3b: Static QDQ INT8 quantization of MobileNetV2 ONNX model (for Android deployment).

Dynamic quantization (quantize_and_compare.py) turns Conv nodes into ConvInteger,
which has no CPU kernel on the Android ORT 1.21.0 build used in layer5_android.
Static QDQ quantization instead calibrates activation ranges offline and emits
QLinearConv/QLinearMatMul nodes, which have broad CPU kernel support including ARM64.
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat


class RandomCalibrationDataReader(CalibrationDataReader):
    """Synthetic calibration data (no representative dataset on hand; benchmark-only use)."""

    def __init__(self, model_path: str, n_samples: int = 50):
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = sess.get_inputs()[0].name
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in sess.get_inputs()[0].shape]
        rng = np.random.default_rng(42)
        self._data = [
            {self._input_name: rng.standard_normal(shape).astype(np.float32)}
            for _ in range(n_samples)
        ]
        self._iter = iter(self._data)

    def get_next(self):
        return next(self._iter, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.onnx")
    parser.add_argument("--out", default="model_int8_qdq.onnx")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    reader = RandomCalibrationDataReader(args.model, args.samples)

    print(f"Static QDQ quantizing {args.model} -> {args.out} ({args.samples} calibration samples) ...")
    quantize_static(
        args.model,
        args.out,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    fp32_mb = os.path.getsize(args.model) / (1024 ** 2)
    qdq_mb = os.path.getsize(args.out) / (1024 ** 2)
    print(f"  FP32 size : {fp32_mb:.1f} MB")
    print(f"  QDQ size  : {qdq_mb:.1f} MB  ({fp32_mb/qdq_mb:.2f}x smaller)")


if __name__ == "__main__":
    main()

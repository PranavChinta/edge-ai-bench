"""
Layer 3: INT8 post-training quantization via ONNX Runtime and accuracy comparison.
"""

import argparse
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def run_inference(session: ort.InferenceSession, inputs: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: inputs})[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.onnx")
    parser.add_argument("--quantized", default="model_int8.onnx")
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()

    quantize_dynamic(args.model, args.quantized, weight_type=QuantType.QInt8)
    print(f"Quantized model saved to {args.quantized}")

    rng = np.random.default_rng(42)
    inputs = rng.standard_normal((args.samples, 128)).astype(np.float32)

    fp32_sess = ort.InferenceSession(args.model)
    int8_sess = ort.InferenceSession(args.quantized)

    fp32_out = run_inference(fp32_sess, inputs)
    int8_out  = run_inference(int8_sess, inputs)

    max_err = float(np.abs(fp32_out - int8_out).max())
    mean_err = float(np.abs(fp32_out - int8_out).mean())
    print(f"Max absolute error:  {max_err:.6f}")
    print(f"Mean absolute error: {mean_err:.6f}")


if __name__ == "__main__":
    main()

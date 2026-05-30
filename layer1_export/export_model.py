"""
Layer 1: Load pretrained MobileNetV2 and export to ONNX.
"""

import os
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def main():
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model         : MobileNetV2 (IMAGENET1K_V1)")
    print(f"Parameters    : {total_params:,}")

    dummy_input = torch.zeros(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape  : {tuple(output.shape)}")

    export_path = "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )

    size_mb = os.path.getsize(export_path) / (1024 ** 2)
    print(f"\nExport succeeded: {export_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

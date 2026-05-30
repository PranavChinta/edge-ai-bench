"""
Layer 1: PyTorch MLP definition and ONNX export.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    model = MLP()
    model.eval()

    print("Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    dummy_input = torch.randn(1, 128)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

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

    import os
    size_kb = os.path.getsize(export_path) / 1024
    print(f"\nExport succeeded: {export_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

"""Export DINOv3 ViT patch tokens to ONNX.

Runtime input stays normalized pixel_values with shape (1,3,384,384). The
output is (1,576,384), matching VisionFeatureCacher._encode_dino() after
dropping CLS and register tokens.
"""
import os
from pathlib import Path

import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
DINO_PATH = os.environ.get(
    "DINOV3_MODEL_PATH",
    str(ROOT / "minicpm_robot_track/backbones/dino_local_hf"),
)
OUT_PATH = os.environ.get(
    "DINO_ONNX_PATH",
    str(Path(__file__).resolve().parent / "dino_patch_jp6_op17.onnx"),
)
OPSET = int(os.environ.get("DINO_ONNX_OPSET", "17"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DinoPatchONNXWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.num_register_tokens = int(getattr(model.config, "num_register_tokens", 0) or 0)

    def forward(self, pixel_values):
        out = self.m(pixel_values=pixel_values)
        return out.last_hidden_state[:, 1 + self.num_register_tokens :, :]


print(f"[export-dino] loading DINO model from {DINO_PATH}")
try:
    model = AutoModel.from_pretrained(DINO_PATH, attn_implementation="eager")
except TypeError:
    model = AutoModel.from_pretrained(DINO_PATH)
model = model.eval().to(DEVICE)
wrap = DinoPatchONNXWrap(model).eval().to(DEVICE)

dummy = torch.randn(1, 3, 384, 384, device=DEVICE, dtype=torch.float32)
with torch.inference_mode():
    out = wrap(dummy)
print(f"[export-dino] output shape={tuple(out.shape)} dtype={out.dtype}")

print(f"[export-dino] exporting ONNX to {OUT_PATH} opset={OPSET}")
torch.onnx.export(
    wrap,
    (dummy,),
    OUT_PATH,
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    dynamic_axes=None,
    opset_version=OPSET,
    do_constant_folding=True,
)
print(f"[export-dino] done. file size: {os.path.getsize(OUT_PATH)/1024/1024:.1f} MB")

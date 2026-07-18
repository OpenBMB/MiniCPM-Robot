"""Export SiglipVisionModel + exact 27x27->24x24 token pooling to ONNX.

Runtime input stays (1,3,384,384) normalized pixel_values. The output is
(1,576,1152), equivalent to applying adapt_siglip_grid(..., out_hw=(24,24))
to the original SigLIP patch tokens.
"""
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import SiglipVisionModel

ROOT = Path(__file__).resolve().parents[2]
SIGLIP_PATH = os.environ.get(
    "SIGLIP_MODEL_PATH",
    str(ROOT / "minicpm_robot_track/backbones/siglip-so400m-patch14-384"),
)
OUT_PATH = os.environ.get(
    "SIGLIP_POOLED_ONNX_PATH",
    str(Path(__file__).resolve().parent / "siglip_pooled_jp6.onnx"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def adaptive_pool_matrix(in_h: int = 27, in_w: int = 27, out_h: int = 24, out_w: int = 24) -> torch.Tensor:
    rows = []
    for oh in range(out_h):
        h0 = math.floor(oh * in_h / out_h)
        h1 = math.ceil((oh + 1) * in_h / out_h)
        for ow in range(out_w):
            w0 = math.floor(ow * in_w / out_w)
            w1 = math.ceil((ow + 1) * in_w / out_w)
            row = torch.zeros(in_h * in_w, dtype=torch.float32)
            weight = 1.0 / float((h1 - h0) * (w1 - w0))
            for ih in range(h0, h1):
                for iw in range(w0, w1):
                    row[ih * in_w + iw] = weight
            rows.append(row)
    return torch.stack(rows, dim=0)


class SigLipPooledONNXWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.register_buffer("pool", adaptive_pool_matrix(), persistent=True)

    def forward(self, pixel_values):
        out = self.m(pixel_values=pixel_values)
        tok = out.last_hidden_state
        if tok.shape[1] == 730:
            tok = tok[:, 1:, :]
        return torch.matmul(self.pool.to(dtype=tok.dtype), tok)


print(f"[export-pooled] loading SiglipVisionModel from {SIGLIP_PATH}")
model = SiglipVisionModel.from_pretrained(SIGLIP_PATH, attn_implementation="eager").eval().to(DEVICE)
wrap = SigLipPooledONNXWrap(model).eval().to(DEVICE)

dummy = torch.randn(1, 3, 384, 384, device=DEVICE, dtype=torch.float32)
with torch.inference_mode():
    out = wrap(dummy)
    raw = model(pixel_values=dummy).last_hidden_state
    if raw.shape[1] == 730:
        raw = raw[:, 1:, :]
    ref = F.adaptive_avg_pool2d(
        raw.transpose(1, 2).contiguous().view(1, raw.shape[-1], 27, 27),
        output_size=(24, 24),
    ).flatten(2).transpose(1, 2).contiguous()
    max_abs = (out - ref).abs().max().item()
print(f"[export-pooled] output shape={tuple(out.shape)} max_abs_vs_adaptive_avg_pool={max_abs:.6g}")

print(f"[export-pooled] exporting ONNX to {OUT_PATH}")
torch.onnx.export(
    wrap,
    (dummy,),
    OUT_PATH,
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    dynamic_axes=None,
    opset_version=13,
    do_constant_folding=True,
)
print(f"[export-pooled] done. file size: {os.path.getsize(OUT_PATH)/1024/1024:.1f} MB")

"""In-process TensorRT DINOv3 backend.

Drop-in replacement for the DINO AutoModel used by ``cache_gridpool.py``. The
engine exports patch tokens directly, so ``patch_tokens_only`` tells the caller
not to drop CLS/register tokens a second time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import tensorrt as trt

IN_SHAPE = (1, 3, 384, 384)
DEFAULT_OUTPUT_TOKENS = 576
HIDDEN_SIZE = 384
DEFAULT_ENGINE = str(
    Path(__file__).resolve().parents[1] / "trt_artifacts" / "dino_patch_target_fp16.engine"
)


@dataclass
class _Config:
    hidden_size: int = HIDDEN_SIZE
    patch_size: int = 16
    num_register_tokens: int = 0


class _Output:
    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


def _resolve_io(engine) -> Tuple[Optional[str], Optional[str], Tuple[int, ...]]:
    if hasattr(engine, "get_binding_shape"):
        return None, None, tuple(int(x) for x in engine.get_binding_shape(1))

    input_name = None
    output_name = None
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            output_name = name
    if input_name is None or output_name is None:
        raise RuntimeError(f"could not resolve TRT IO tensors: input={input_name}, output={output_name}")
    return input_name, output_name, tuple(int(x) for x in engine.get_tensor_shape(output_name))


class DinoTrtDirect:
    """In-process TensorRT replacement for the DINO vision model."""

    patch_tokens_only = True

    def __init__(self, engine_path: str = DEFAULT_ENGINE, output_tokens: int = DEFAULT_OUTPUT_TOKENS):
        if not torch.cuda.is_available():
            raise RuntimeError("DinoTrtDirect requires CUDA")
        self.config = _Config()
        self._engine_path = engine_path
        self._output_tokens = int(output_tokens)
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine: {engine_path}")
        self._ctx = self._engine.create_execution_context()
        self._input_name, self._output_name, self._out_shape = _resolve_io(self._engine)
        if len(self._out_shape) != 3 or self._out_shape[0] != 1 or self._out_shape[2] != HIDDEN_SIZE:
            raise RuntimeError(f"unexpected DINO TRT output shape: {self._out_shape}")
        if self._out_shape[1] != self._output_tokens:
            raise RuntimeError(
                f"engine output tokens {self._out_shape[1]} != expected {self._output_tokens}"
            )
        self._stream = torch.cuda.Stream()
        self._output = torch.empty(self._out_shape, device="cuda", dtype=torch.float32)
        self._warmup()
        print(f"[dino-trt] loaded engine={engine_path} in={IN_SHAPE} out={self._out_shape}", flush=True)

    def _execute(self, x: torch.Tensor, out: torch.Tensor) -> None:
        current_stream = torch.cuda.current_stream(x.device)
        self._stream.wait_stream(current_stream)
        stream = self._stream.cuda_stream
        if self._input_name is not None:
            self._ctx.set_tensor_address(self._input_name, int(x.data_ptr()))
            self._ctx.set_tensor_address(self._output_name, int(out.data_ptr()))
            ok = self._ctx.execute_async_v3(stream)
        else:
            ok = self._ctx.execute_async_v2([int(x.data_ptr()), int(out.data_ptr())], stream)
        if ok is False:
            raise RuntimeError("TensorRT execute_async failed")
        current_stream.wait_stream(self._stream)

    def _warmup(self) -> None:
        dummy = torch.empty(IN_SHAPE, device="cuda", dtype=torch.float32)
        for _ in range(3):
            self._execute(dummy, self._output)
        torch.cuda.current_stream(dummy.device).synchronize()

    @torch.inference_mode()
    def __call__(self, pixel_values: torch.Tensor, **_unused) -> _Output:
        if tuple(pixel_values.shape) != IN_SHAPE:
            raise ValueError(f"DinoTrtDirect expects {IN_SHAPE}, got {tuple(pixel_values.shape)}")
        if not pixel_values.is_cuda:
            raise ValueError("DinoTrtDirect expects CUDA pixel_values")
        x = pixel_values.detach()
        if x.dtype != torch.float32 or not x.is_contiguous():
            x = x.to(device=x.device, dtype=torch.float32).contiguous()
        if self._output.device != x.device:
            self._output = torch.empty(self._out_shape, device=x.device, dtype=torch.float32)
        self._execute(x, self._output)
        return _Output(self._output)

    def eval(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

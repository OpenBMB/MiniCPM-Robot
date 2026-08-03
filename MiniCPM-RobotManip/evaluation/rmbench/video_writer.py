"""Per-rollout MP4 writer for the closed-loop RMBench eval.

The vendored ``rmbench.envs.utils.images_to_video`` shells out to a system ``ffmpeg`` binary,
which is NOT installed in the ``memvla`` conda env. This module writes MP4s through
``imageio`` + ``imageio-ffmpeg`` instead (the latter ships its own ffmpeg binary, so it works
with no system ffmpeg), falling back to ``cv2.VideoWriter`` (mp4v) if imageio is unavailable.

Each closed-loop step renders three cameras (``cam_high``, ``cam_left_wrist``,
``cam_right_wrist``), each a HWC uint8 RGB array (240x320 under demo_clean's LargeView). We
tile the three side-by-side into one frame so a single MP4 shows the whole scene + both wrists,
and write one MP4 per (task, seed) rollout.
"""

from __future__ import annotations

import os

import numpy as np

# Camera order in the tiled frame (left -> right). Keys match encode_obs' short names.
CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def tile_frame(frame: dict) -> np.ndarray | None:
    """Tile the per-camera RGB dict into ONE (H, W*ncam, 3) uint8 frame.

    ``frame`` is encode_obs' output: ``{cam_high, cam_left_wrist, cam_right_wrist}`` each HWC
    uint8. Missing/odd cameras are skipped; cameras of differing height are padded to the max
    height so the horizontal stack is well-formed. Returns None if no usable camera is present.
    """
    imgs = []
    for key in CAMERA_ORDER:
        v = frame.get(key)
        if v is None:
            continue
        a = np.asarray(v)
        if a.ndim != 3 or a.shape[2] < 3:
            continue
        imgs.append(a[:, :, :3].astype(np.uint8))
    if not imgs:
        return None
    h = max(a.shape[0] for a in imgs)
    padded = []
    for a in imgs:
        if a.shape[0] < h:
            pad = np.zeros((h - a.shape[0], a.shape[1], 3), dtype=np.uint8)
            a = np.concatenate([a, pad], axis=0)
        padded.append(a)
    return np.concatenate(padded, axis=1)


class RolloutVideoRecorder:
    """Accumulate tiled frames for ONE rollout, then write a single MP4 on close.

    Used per-env inside ``run_group``: ``add(frame)`` each observed native frame, then
    ``save(out_path, success)`` once the episode ends. Frame accumulation is cheap
    (240x960x3 uint8 ~ 0.7 MB/frame); a long blocks_ranking rollout (~3500 frames) is ~2.4 GB
    in RAM, so callers should keep at most ``group_size`` recorders alive at once (the eval
    does: one per active env in a group) and ``save`` frees them.
    """

    def __init__(self, fps: float = 16.6667):
        self.fps = float(fps)
        self.frames: list[np.ndarray] = []

    def add(self, frame: dict) -> None:
        tiled = tile_frame(frame)
        if tiled is not None:
            self.frames.append(tiled)

    def __len__(self) -> int:
        return len(self.frames)

    def save(self, out_path: str) -> bool:
        """Write the accumulated frames to ``out_path`` (mp4). Returns True on success.

        Never raises: a video-writing failure must not abort the eval rollout it is recording.
        """
        if not self.frames:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        arr = np.stack(self.frames, axis=0)  # (N, H, W, 3) uint8
        ok = _write_imageio(arr, out_path, self.fps)
        if not ok:
            ok = _write_cv2(arr, out_path, self.fps)
        # Free the buffer regardless so a failed write does not pin ~GBs of frames.
        self.frames = []
        return ok


def _write_imageio(arr: np.ndarray, out_path: str, fps: float) -> bool:
    try:
        import imageio.v2 as imageio
    except Exception:
        try:
            import imageio  # type: ignore
        except Exception:
            return False
    try:
        # libx264 + yuv420p so the mp4 plays everywhere; macro_block_size=1 tolerates odd dims.
        writer = imageio.get_writer(
            out_path, format="FFMPEG", mode="I", fps=fps,
            codec="libx264", pixelformat="yuv420p", macro_block_size=1,
        )
        try:
            for f in arr:
                writer.append_data(f)
        finally:
            writer.close()
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def _write_cv2(arr: np.ndarray, out_path: str, fps: float) -> bool:
    try:
        import cv2
    except Exception:
        return False
    try:
        n, h, w, _ = arr.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        if not vw.isOpened():
            return False
        for f in arr:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))  # cv2 expects BGR
        vw.release()
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False

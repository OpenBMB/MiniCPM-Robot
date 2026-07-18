"""End-to-end latency benchmark for the HTTP inference path.

Sends a fixed RealSense-shaped image to http://127.0.0.1:5801/eval_dual and
reports the per-stage latency breakdown that the server returns.

Usage:
    python3 realworld/bench_e2e.py --img /path/to/test.jpg
    python3 realworld/bench_e2e.py --img /path/to/test.jpg --iters 30 --warmup 5
"""
import argparse
import json
import socket
import statistics
import struct
import sys
import time
from multiprocessing import shared_memory

import cv2
import numpy as np
import requests
from PIL import Image

TCP_REQ_HEADER = struct.Struct("!4sII")
TCP_RESP_HEADER = struct.Struct("!4sI")
TCP_REQ_MAGIC = b"OVL1"
TCP_RESP_MAGIC = b"OVR1"


class TcpEvalClient:
    def __init__(self, tcp_url):
        from urllib.parse import urlparse

        if "://" not in tcp_url:
            tcp_url = "tcp://" + tcp_url
        u = urlparse(tcp_url)
        if not u.hostname or not u.port:
            raise ValueError(f"tcp_url must look like tcp://host:port, got {tcp_url!r}")
        self.host = u.hostname
        self.port = int(u.port)
        self.sock = None
        self._connect()

    def _connect(self):
        self.close()
        self.sock = socket.create_connection((self.host, self.port), timeout=10)
        self.sock.settimeout(120)
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _recv_exact(self, nbytes):
        buf = bytearray(nbytes)
        view = memoryview(buf)
        got = 0
        while got < nbytes:
            n = self.sock.recv_into(view[got:], nbytes - got)
            if n == 0:
                raise ConnectionError("TCP inference server closed the connection")
            got += n
        return bytes(buf)

    def request(self, payload, image_bytes):
        json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = TCP_REQ_HEADER.pack(TCP_REQ_MAGIC, len(json_bytes), len(image_bytes))
        self.sock.sendall(header + json_bytes + image_bytes)
        magic, resp_len = TCP_RESP_HEADER.unpack(self._recv_exact(TCP_RESP_HEADER.size))
        if magic != TCP_RESP_MAGIC:
            raise RuntimeError(f"bad TCP response magic: {magic!r}")
        out = json.loads(self._recv_exact(resp_len).decode("utf-8"))
        if "error" in out:
            raise RuntimeError(str(out["error"]))
        return out

KEYS = [
    "client_serialize_ms",
    "client_http_roundtrip_ms",
    "client_response_parse_ms",
    "server_decode_ms",
    "server_payload_parse_ms",
    "server_pre_infer_ms",
    "e2e_ms",
    "vision_encode_ms",
    "vision_preprocess_ms",
    "vision_dino_ms",
    "vision_siglip_ms",
    "vision_pool_ms",
    "history_pack_ms",
    "policy_forward_ms",
    "server_response_prepare_ms",
    "server_route_total_ms",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5801/eval_dual")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--img", required=True)
    ap.add_argument("--instruction", default="go forward")
    ap.add_argument("--transport", choices=["jpeg", "raw", "shm", "tcp_jpeg", "tcp_raw"], default="jpeg")
    ap.add_argument("--tcp-url", default="")
    ap.add_argument("--send-depth", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    pil = Image.open(args.img).convert("RGB")
    img_rgb = np.array(pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    depth = np.zeros(img_rgb.shape[:2], dtype=np.uint16)
    shm = None
    shm_view = None
    if args.transport == "shm":
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        shm = shared_memory.SharedMemory(create=True, size=img_rgb.nbytes)
        shm_view = np.ndarray(img_rgb.shape, dtype=np.uint8, buffer=shm.buf)
    session = requests.Session()
    tcp_client = None
    if args.transport in ("tcp_jpeg", "tcp_raw"):
        if args.tcp_url:
            tcp_url = args.tcp_url
        else:
            from urllib.parse import urlparse
            u = urlparse(args.url)
            tcp_url = f"tcp://{u.hostname or '127.0.0.1'}:5803"
        tcp_client = TcpEvalClient(tcp_url)

    def one(reset=False):
        ser_t0 = time.perf_counter()
        if args.transport == "shm":
            np.copyto(shm_view, img_rgb)
            rgb_payload = None
        elif args.transport in ("raw", "tcp_raw"):
            rgb_payload = np.ascontiguousarray(img_rgb, dtype=np.uint8).tobytes()
        else:
            ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("failed to encode RGB JPEG")
            rgb_payload = buf.tobytes()
        depth_bytes = None
        if args.send_depth:
            ok, buf = cv2.imencode(".png", depth)
            if not ok:
                raise RuntimeError("failed to encode depth PNG")
            depth_bytes = buf.tobytes()
        client_serialize_ms = (time.perf_counter() - ser_t0) * 1000.0
        send_ts = time.time()
        if args.transport == "shm":
            files = None
        elif args.transport == "raw":
            files = {
                "image_raw": ("rgb.raw", rgb_payload, "application/octet-stream"),
            }
        elif args.transport in ("tcp_jpeg", "tcp_raw"):
            files = None
        else:
            files = {
                "image": ("rgb.jpg", rgb_payload, "image/jpeg"),
            }
        if depth_bytes is not None:
            if files is None:
                raise ValueError(f"--send-depth is not supported with transport={args.transport}")
            files["depth"] = ("depth.png", depth_bytes, "image/png")
        payload = {
            "instruction": args.instruction,
            "reset": reset,
            "client_send_timestamp": send_ts,
            "client_serialize_ms": client_serialize_ms,
            "image_height": int(img_rgb.shape[0]),
            "image_width": int(img_rgb.shape[1]),
            "image_channels": int(img_rgb.shape[2]),
            "transport": args.transport,
        }
        if args.transport == "tcp_raw":
            payload["image_encoding"] = "raw_rgb"
        else:
            payload["image_encoding"] = "jpeg"
        if args.transport == "shm":
            payload["image_shm_name"] = shm.name
            data = None
        else:
            data = {"json": json.dumps(payload)}
        req_t0 = time.perf_counter()
        if args.transport == "shm":
            resp = session.post(args.url, json=payload, timeout=120)
            out = resp.json()
        elif args.transport in ("tcp_jpeg", "tcp_raw"):
            out = tcp_client.request(payload, rgb_payload)
        else:
            resp = session.post(args.url, files=files, data=data, timeout=120)
            out = resp.json()
        client_http_roundtrip_ms = (time.perf_counter() - req_t0) * 1000.0
        parse_t0 = time.perf_counter()
        out["client_response_parse_ms"] = (time.perf_counter() - parse_t0) * 1000.0
        out["client_serialize_ms"] = client_serialize_ms
        out["client_http_roundtrip_ms"] = client_http_roundtrip_ms
        return out

    print(f"[bench] url={args.url} transport={args.transport} send_depth={args.send_depth}")
    try:
        one(reset=True)
        for _ in range(args.warmup):
            one()

        acc = {k: [] for k in KEYS}
        for i in range(args.iters):
            j = one()
            for k in KEYS:
                v = j.get(k)
                if v is not None:
                    acc[k].append(float(v))
        print(f"\n=== {args.iters} iters, {args.warmup} warmup ===")
        print(f"{'stage':<22} {'mean':>8} {'p50':>8} {'p95':>8} {'std':>6}")
        for k in KEYS:
            a = acc[k]
            if not a:
                continue
            print(
                f"{k:<22} "
                f"{statistics.mean(a):>8.2f} "
                f"{statistics.median(a):>8.2f} "
                f"{np.percentile(a, 95):>8.2f} "
                f"{statistics.stdev(a) if len(a) > 1 else 0.0:>6.2f}"
            )
        print(f"\nfps mean = {1000.0 / statistics.mean(acc['e2e_ms']):.3f}")
        return 0
    finally:
        if shm is not None:
            shm.close()
            shm.unlink()
        if tcp_client is not None:
            tcp_client.close()


if __name__ == "__main__":
    sys.exit(main())

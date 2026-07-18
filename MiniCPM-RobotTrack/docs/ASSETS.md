# Model and Deployment Assets

**English** | [简体中文](ASSETS_zh-CN.md)

The Git repository does not include model weights, ONNX files, TensorRT
engines, or the Jetson PyTorch wheel. Users download these files from their
official sources or generate them on the target Orin NX.

Run all commands from the deployment directory:

```bash
cd MiniCPM-RobotTrack
```

## 1. Download Base Models and Jetson PyTorch

The upstream asset script downloads:

- `google/siglip-so400m-patch14-384`;
- `facebook/dinov3-vits16-pretrain-lvd1689m`;
- the NVIDIA PyTorch wheel used by the JetPack deployment.

The released MiniCPM-RobotTrack snapshot already contains its fine-tuned
MiniCPM4 backbone, so a separate `openbmb/MiniCPM4-0.5B` download is not used
by deployment.

DINOv3 is gated. Accept its license on Hugging Face and authenticate before
the first download:

```bash
hf auth login
python3 scripts/download_upstream_assets.py
```

Alternatively, provide the token through the environment:

```bash
export HF_TOKEN=your_token
python3 scripts/download_upstream_assets.py
```

The script places base models under `minicpm_robot_track/backbones/` and the
Jetson PyTorch wheel under `vendor/`. Existing files are reused.

## 2. Download the MiniCPM-RobotTrack Checkpoint Manually

Open the official model repository:

<https://huggingface.co/openbmb/MiniCPM-RobotTrack>

Download the complete Hugging Face repository snapshot yourself. Copy the
contents of the snapshot, rather than an extra outer download directory, into:

```text
minicpm_robot_track/checkpoints/MiniCPM-RobotTrack/
├── config.json
├── configuration_minicpm.py
├── configuration_robottrack.py
├── modeling_minicpm.py
├── modeling_robottrack.py
├── model.safetensors
├── tokenizer.json
├── tokenizer.model
├── tokenizer_config.json
└── ...
```

`model.safetensors` is stored through Hugging Face's large-file storage. A
source archive that contains only Python, JSON, and tokenizer files is not a
complete model download. The runtime also accepts standard sharded
`model.safetensors` or `pytorch_model.bin` layouts.

The deployment loads this directory with `AutoTokenizer.from_pretrained(...)`
and `AutoModel.from_pretrained(..., trust_remote_code=True)`. The runtime is
offline after the snapshot is installed and does not download a second
MiniCPM4 backbone.

## 3. Export ONNX

Install build dependencies, then export the DINOv3 and SigLIP models from the
downloaded official base models:

```bash
python3 -m pip install --user -r requirements-build.txt
./scripts/export_onnx.sh
```

The generated files are:

```text
realworld/trt_artifacts/dino_patch_jp6_op17.onnx
realworld/trt_artifacts/siglip_pooled_jp6.onnx
```

ONNX files are deployment intermediates and should not be committed to Git.

## 4. Build TensorRT Engines

TensorRT engines depend on the JetPack and TensorRT versions, power mode, and
target device. Build them on the Orin NX that will run the deployment:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
./scripts/build_engines.sh
```

The generated files are:

```text
realworld/trt_artifacts/dino_patch_target_fp16.engine
realworld/trt_artifacts/siglip_pooled_target_maxn_fp16.engine
```

Do not copy an engine built on one device to another device. After all assets
are in place, run:

```bash
./scripts/preflight.sh
./go2_runtime.py run
```

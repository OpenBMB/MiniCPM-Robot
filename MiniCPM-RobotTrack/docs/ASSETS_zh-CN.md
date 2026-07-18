# 模型与部署资产

[English](ASSETS.md) | **简体中文**

Git 仓库不包含模型权重、ONNX、TensorRT engine 或 Jetson PyTorch wheel。
这些文件由使用者从官方来源下载，或在目标 Orin NX 上生成。

所有命令都从仓库的部署子目录运行：

```bash
cd MiniCPM-RobotTrack
```

## 1. 下载官方基础模型和 Jetson PyTorch

下面的脚本下载：

- `google/siglip-so400m-patch14-384`；
- `facebook/dinov3-vits16-pretrain-lvd1689m`；
- JetPack 6.1 对应的 NVIDIA PyTorch wheel。

MiniCPM-RobotTrack snapshot 已经包含微调后的 MiniCPM4 backbone，因此部署时
不再单独下载 `openbmb/MiniCPM4-0.5B`。

DINOv3 是 gated 模型。首次下载前，需要在 Hugging Face 页面接受许可并登录：

```bash
hf auth login
python3 scripts/download_upstream_assets.py
```

也可以通过环境变量提供访问令牌：

```bash
export HF_TOKEN=your_token
python3 scripts/download_upstream_assets.py
```

脚本会把基础模型放入 `minicpm_robot_track/backbones/`，把 Jetson PyTorch wheel
放入 `vendor/`。已存在的 wheel 不会重复下载。

## 2. 手动下载 MiniCPM-RobotTrack checkpoint

打开官方模型仓库：

<https://huggingface.co/openbmb/MiniCPM-RobotTrack>

使用者自行下载完整的 Hugging Face 仓库 snapshot。将 snapshot 内的文件直接放入下列
目录，不要在目标目录中额外保留一层下载工具生成的外层同名文件夹：

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

`model.safetensors` 由 Hugging Face 大文件存储管理。只包含 Python、JSON 和 tokenizer
文件的源码压缩包不是完整模型。运行时也支持标准的分片 `model.safetensors` 或
`pytorch_model.bin` 布局。

部署端通过 `AutoTokenizer.from_pretrained(...)` 和
`AutoModel.from_pretrained(..., trust_remote_code=True)` 加载该目录。snapshot 安装完成后
运行时保持离线，也不会再次下载 MiniCPM4 backbone。

## 3. 导出 ONNX

安装构建依赖后，从已下载的 DINOv3 和 SigLIP 模型导出 ONNX：

```bash
python3 -m pip install --user -r requirements-build.txt
./scripts/export_onnx.sh
```

生成文件位于：

```text
realworld/trt_artifacts/dino_patch_jp6_op17.onnx
realworld/trt_artifacts/siglip_pooled_jp6.onnx
```

ONNX 是部署端中间产物，不需要提交到 Git 仓库。

## 4. 构建 TensorRT engine

TensorRT engine 与 JetPack、TensorRT 版本、功耗模式和目标设备相关，必须在最终运行的
Orin NX 上构建：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
./scripts/build_engines.sh
```

生成文件位于：

```text
realworld/trt_artifacts/dino_patch_target_fp16.engine
realworld/trt_artifacts/siglip_pooled_target_maxn_fp16.engine
```

不要把一台机器生成的 engine 复制给其他设备使用。完成以上步骤后，运行：

```bash
./scripts/preflight.sh
./go2_runtime.py run
```

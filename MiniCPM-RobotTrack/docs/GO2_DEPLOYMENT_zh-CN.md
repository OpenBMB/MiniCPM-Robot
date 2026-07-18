# MiniCPM-RobotTrack 在 Unitree Go2 上的部署

[English](GO2_DEPLOYMENT.md) | **简体中文**

如果设备已经完成 JetPack 6.2.2、Go2 载板补丁、CycloneDDS 0.10.2、ROS 2 Humble
和 Jetson 专用 PyTorch 的安装，可以直接从“快速开始”执行；全新设备或排查问题时，
按后续编号章节逐项操作。

已验证的软件环境为 Jetson Linux R36.5、CUDA 12.6、TensorRT 10.7、Python 3.10、
ROS 2 Humble 和 MAXN mode 0。运行时默认是 `dry-run`，不会向 Go2 下发运动命令。

## 快速开始

从仓库的部署子目录运行：

```bash
cd MiniCPM-RobotTrack
python3 scripts/download_upstream_assets.py
```

使用者需要自行从
[openbmb/MiniCPM-RobotTrack](https://huggingface.co/openbmb/MiniCPM-RobotTrack)
下载完整的项目 snapshot，并将 snapshot 内文件直接放到以下目录：

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

该目录必须包含完整权重：可以是单个 `model.safetensors`，也可以是
`model.safetensors.index.json` 及其引用的全部 `model-*.safetensors` 分片。只有代码和
JSON 文件的源码压缩包不能用于推理。

然后导出 ONNX、在目标设备上构建 TensorRT engine，并以 dry-run 启动：

```bash
python3 -m pip install --user -r requirements-build.txt
./scripts/export_onnx.sh

sudo nvpmodel -m 0
# 如果 nvpmodel 要求重启，重启并返回本目录后继续。
sudo jetson_clocks
./scripts/build_engines.sh

./scripts/preflight.sh
./check_go2_camera.py --frames 30
./go2_runtime.py run
```

状态与停止命令：

```bash
./go2_runtime.py status
./go2_runtime.py stop-control
./go2_runtime.py stop
```

同一局域网可访问 `http://192.168.123.18:5801/web`。5801 端口没有认证，不要暴露到
公网。

JetPack 5 原系统检查、文件级备份、完整分区备份、Recovery、JetPack 6.2.2 刷写、
验收和回退的完整教程见
[`JETPACK6_UPGRADE_zh-CN.md`](JETPACK6_UPGRADE_zh-CN.md)。模型部署使用下文指定的
KevinLADLee 载板补丁。

## 0. 硬件和复现参数检查

### 0.1 必选硬件

完整项目的目标 BOM 是：

- Unitree **Go2 EDU**，带扩展坞和 EDU SDK 能力；
- 扩展坞内的 **Jetson Orin NX 16GB** 和 NVMe；
- **Intel RealSense D435i** 及 USB 3 数据线；
- 遥控器或 App 急停、稳定供电；
- 用于刷机的 Ubuntu 22.04 x86_64 主机和 USB-C 数据线。

部署支持范围仅包括 Go2 EDU、Orin NX 16GB 和 D435i。Go2 Air/Pro、Orin Nano、
Orin NX 8GB 及其他 RealSense 型号不在支持范围内。默认推理输入是 Go2 前置相机；
D435i 是项目硬件 BOM 的必选项，但运行时只读取 D435i RGB，不使用 depth 或 IMU。

### 0.2 固定默认值与现场变量

下列值在使用指定补丁且没有二次修改网络时应保持一致：

```text
Go2 内部主控       192.168.123.161
Orin 有线地址      192.168.123.18/24
Orin USB gadget    192.168.55.1
默认 Linux 用户    unitree
默认有线接口       enP8p1s0
```

补丁首次创建 `unitree` 用户时初始密码为 `123`，但首次登录必须执行 `passwd` 修改。
之后 `sudo` 使用该设备当前账户密码，因此每台设备的密码可能不同。任何密码都不能写进
YAML、脚本或 Git。SSH key、D435i 序列号、NVMe 设备名、Hugging Face
repo/token 也必然或可能不同。

网卡名可能因 BSP 或 NetworkManager 配置变化。确认后修改
`go2_runtime.yaml` 的 `camera.video_network_interface`；运行、相机预检和停车命令都会读取
这个值。CycloneDDS 默认安装在 `/home/unitree/cyclonedds/install`，其他路径通过环境变量
指定：

```bash
export CYCLONEDDS_HOME=/opt/cyclonedds-0.10.2
```

仓库可以 clone 到任意目录，模型、日志和脚本路径都相对仓库根目录解析。只有
CycloneDDS 使用上述独立前缀。

### 0.3 上机前确认

在 Orin 上执行：

```bash
cat /etc/nv_tegra_release
nvpmodel -q
ip -br address
ip route get 192.168.123.161
ping -c 3 192.168.123.161
test -d "${CYCLONEDDS_HOME:-/home/unitree/cyclonedds/install}/lib"
rs-enumerate-devices | sed -n '1,30p'
sudo -v
```

`sudo -v` 应由复现者交互输入本机密码，仓库不提供也不推测密码。从外部刷机/管理主机
连接 `192.168.123.0/24` 时，为主机选择未占用地址（教程示例为
`192.168.123.222/24`），不要占用 `.18` 或 `.161`。

## 1. 应用 Go2 载板补丁

> [!IMPORTANT]
> RoboLegion `go2` 分支和 KevinLADLee 仓库是两套 BSP 构建/刷写工具链，都会修改
> Unitree 载板配置并写入 QSPI/NVMe。不要把以下步骤误解为
> JetPack 启动后直接安装的普通驱动命令，也不要未经验证就把两套补丁叠加到同一
> `Linux_for_Tegra`。如果选用该补丁构建最终镜像，应从干净的 JetPack 6.2.2 BSP
> 应用补丁并重新刷写；执行前保留完整教程生成的备份并确认目标盘会被覆盖。

参考仓库：
https://github.com/KevinLADLee/unitree-go2-jetpack6.x-patches

使用以下已验证提交：

```text
f7be6640865823c831feaaaaa7ba4e74c31808a7
```

在 Ubuntu x86_64 刷机主机执行：

```bash
git clone https://github.com/KevinLADLee/unitree-go2-jetpack6.x-patches.git
cd unitree-go2-jetpack6.x-patches
git checkout f7be6640865823c831feaaaaa7ba4e74c31808a7

export L4T_DIR=/path/to/JetPack_6.2.2_Linux_JETSON_ORIN_NX_TARGETS/Linux_for_Tegra
sudo ./apply_unitree_bsp_patches.sh
sudo ./apply_unitree_rootfs_config.sh
```

该补丁修改 hybrid USB DTB、PP.06 USB 供电、SSH、默认用户、静态有线网络和
`extlinux` FDT。运行脚本前应阅读上游 README、LICENSE、NOTICE 和脚本内容。

按上游流程刷 NVMe：

```bash
cd "$L4T_DIR"
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
  --external-device nvme0n1p1 \
  -p "-c ./bootloader/generic/cfg/flash_t234_qspi.xml" \
  -c ./tools/kernel_flash/flash_l4t_t234_nvme.xml \
  --showlogs \
  --network usb0 \
  jetson-orin-nano-devkit external
```

该固定提交会创建 `unitree/123` 并配置 `192.168.123.18/24`。首次登录后必须立即
修改默认密码，安装自己的 SSH key，并确认防火墙没有把 5801/5803 暴露到公网。

重启后的载板验收：

```bash
cat /proc/device-tree/bus@0/padctl@3520000/ports/usb2-0/mode; echo
cat /proc/device-tree/bus@0/padctl@3520000/pads/usb3/lanes/usb3-2/status; echo
tr '\0' ' ' </proc/device-tree/bus@0/usb@3610000/phy-names; echo
for f in /sys/class/drm/card*-*/status; do echo "$f: $(cat "$f")"; done
lsusb -t
systemctl status unitree-pp6-usb-enable.service --no-pager
```

期望 `usb2-0=otg`、`usb3-2=okay`、USB3 设备显示 `5000M/10000M`，供电服务为
active。补丁仓库还说明了 Type-C DP 和 NIC 名不是默认值时的处理方法。

## 2. 部署运行时

### 2.1 系统和 Python 依赖

```bash
cd MiniCPM-RobotTrack

sudo apt-get update
sudo apt-get install -y \
  git curl python3-pip python3-dev python3-yaml cmake build-essential \
  ros-humble-cv-bridge ros-humble-realsense2-camera

python3 -m pip install --user -r requirements-runtime.txt
```

安装官方 Unitree SDK2 Python 1.0.1，并确认 Python `cyclonedds==0.10.2`。Go2
VideoClient 还需要与 ROS Humble 动态库兼容的 CycloneDDS 0.10.2，验证部署路径为：

```text
/home/unitree/cyclonedds/install/lib
```

路径不同时，在安装依赖、preflight 和运行命令所在的 shell 中设置
`CYCLONEDDS_HOME`。仓库会自动使用 `$CYCLONEDDS_HOME/lib`。

如果从源码构建 CycloneDDS，应固定 0.10.2、以
`CMAKE_INSTALL_PREFIX=/home/unitree/cyclonedds/install` 安装，并在本机验证
`libddsc.so` 的 ROS Humble/iceoryx 动态链接。不要让 ROS 自带的另一版本覆盖它。

Jetson PyTorch、torchvision 和 TensorRT 不能用通用 x86/PyPI 包代替。验证组合为：

```text
torch       2.5.0a0+872d972e41.nv24.08
torchvision 0.20.0a0+afc54f7
TensorRT    10.7.0
CUDA        12.6
```

`vendor/` 中的 PyTorch wheel 由资产包提供。它需要 `libcusparseLt.so.0`；安装前用
`apt-cache policy libcusparselt0` 确认可用 ARM64 版本。升级 TensorRT 时，APT 可能移除
JetPack 元包，必须先确认不会移除 CUDA、cuDNN、驱动等实际组件。

### 2.2 装配资产

先从官方上游下载公开基础模型、gated DINOv3 和 NVIDIA wheel：

```bash
python3 scripts/download_upstream_assets.py
```

DINOv3 需要先在 Hugging Face 页面接受许可，再执行 `hf auth login` 或设置
`HF_TOKEN`。这些上游资产直接从各自官方来源下载。

项目模型由使用者自行从
[openbmb/MiniCPM-RobotTrack](https://huggingface.co/openbmb/MiniCPM-RobotTrack)
下载完整 snapshot，并将其中的文件放入
`minicpm_robot_track/checkpoints/MiniCPM-RobotTrack/`。snapshot 已包含微调后的 MiniCPM4
backbone、tokenizer、自定义模型代码、配置和权重。部署端通过
`AutoModel.from_pretrained(..., trust_remote_code=True)` 加载，不再使用单独下载的
MiniCPM4 backbone。

两份 ONNX 统一在部署端从官方基础模型重建，不随项目 checkpoint 上传：

```bash
python3 -m pip install --user -r requirements-build.txt
./scripts/export_onnx.sh
```

完整的下载目录和构建流程见
[`ASSETS_zh-CN.md`](ASSETS_zh-CN.md)。模型、ONNX 和 engine 都已被
`.gitignore` 排除，不要提交这些大文件。

### 2.3 功耗模式与 TensorRT

先切 MAXN mode 0；命令通常要求确认并重启：

```bash
sudo nvpmodel -m 0
# 按提示输入 YES，重启并重新登录
sudo jetson_clocks
nvpmodel -q
sudo jetson_clocks --show
```

目标应为 8 CPU、GPU 918 MHz、4 TPC、EMC 3199 MHz。然后构建本机 engine：

```bash
./scripts/build_engines.sh
```

期望输出：

```text
realworld/trt_artifacts/dino_patch_target_fp16.engine
realworld/trt_artifacts/siglip_pooled_target_maxn_fp16.engine
```

两个 `trtexec` 日志末尾都必须出现 `&&&& PASSED TensorRT.trtexec`。

### 2.4 部署前检查

```bash
./scripts/preflight.sh
./check_go2_camera.py --frames 30
```

若 Go2 网卡不是 `enP8p1s0`，修改 `go2_runtime.yaml` 中
`camera.video_network_interface`。摄像头检查前不能有另一个 VideoClient 进程。

## 3. Dry-run 验收

```bash
./go2_runtime.py run
```

确认终端和输出 JSON 中：

```text
mode=dry-run
camera_source=go2
actual_v=0
actual_w=0
```

至少运行 60 秒，检查 `./go2_runtime.py status`、5801 网页画面、服务日志、相机帧和
时延 JSON。每次依赖、权重、ONNX、engine 或控制代码变化后都要重新 dry-run。

D435i RGB-only 可用：

```bash
./go2_runtime.py run --camera-source d435i --mode dry-run
```

## 4. 实控安全

模型可能在画面无人时仍输出前进轨迹；目标存在门控、命令超时、失联停车和实体急停
未验收前，不得在开放场地实控。实控需要同时提供两个参数，并在交互终端输入 `MOVE`：

```bash
./go2_runtime.py run --mode live --confirm-live-control
```

默认实控上限是 `vx=0.15 m/s` 和 `wz=0.30 rad/s`。现场必须有操作员，遥控器/App
急停必须可用，并在另一个终端提前准备 `./go2_runtime.py stop-control`。不要使用
`kill -9`、`nohup` 或无人看管的后台服务启动实控。

## 许可证

项目源码使用仓库顶层 [`LICENSE`](../../LICENSE) 中的 Apache-2.0 License。使用或分发
模型资产前，请阅读 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。

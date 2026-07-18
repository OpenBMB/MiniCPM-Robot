# MiniCPM-RobotTrack Deployment on Unitree Go2

**English** | [简体中文](GO2_DEPLOYMENT_zh-CN.md)

Start with the Quick Start section when JetPack 6.2.2 and the Go2 carrier-board
patch are already installed. Follow the numbered sections for a clean device
setup or when troubleshooting an installation.

The validated target is a Unitree Go2 EDU with a Jetson Orin NX 16GB:

```text
Go2/D435i RGB -> TCP JPEG -> DINO + SigLIP TensorRT
              -> MiniCPM-RobotTrack -> waypoint -> rate-limited control
```

The validated software stack is Jetson Linux R36.5, CUDA 12.6, TensorRT 10.7,
Python 3.10, ROS 2 Humble, and MAXN mode 0. The default runtime mode is
`dry-run`, which never sends motion commands to the robot.

## Quick Start

This section assumes that JetPack 6.2.2, the pinned carrier-board patch,
CycloneDDS 0.10.2, ROS 2 Humble, and the Jetson-specific PyTorch packages are
already installed.

From the repository's deployment directory:

```bash
cd MiniCPM-RobotTrack

python3 scripts/download_upstream_assets.py
```

Download the complete project snapshot yourself from
[openbmb/MiniCPM-RobotTrack](https://huggingface.co/openbmb/MiniCPM-RobotTrack)
and copy the snapshot contents into the following directory:

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

The directory must contain complete model weights: either one
`model.safetensors` file or `model.safetensors.index.json` together with every
referenced `model-*.safetensors` shard. A source-only archive is not sufficient.

Then export ONNX, build device-local TensorRT engines, and start a dry run:

```bash
python3 -m pip install --user -r requirements-build.txt
./scripts/export_onnx.sh

sudo nvpmodel -m 0
# Reboot if nvpmodel requests it, then return to this directory.
sudo jetson_clocks
./scripts/build_engines.sh

./scripts/preflight.sh
./check_go2_camera.py --frames 30
./go2_runtime.py run
```

If CycloneDDS is installed at a different prefix, export it before running the
commands above:

```bash
export CYCLONEDDS_HOME=/opt/cyclonedds-0.10.2
```

Useful runtime commands:

```bash
./go2_runtime.py status
./go2_runtime.py stop-control
./go2_runtime.py stop
```

The local web UI is available at `http://192.168.123.18:5801/web`. Port 5801
has no authentication; never expose it to the public Internet.

## 1. Required Hardware

The complete reproduction target uses:

- Unitree **Go2 EDU** with the expansion dock and EDU SDK access;
- **Jetson Orin NX 16GB** and a bootable NVMe drive in the expansion dock;
- **Intel RealSense D435i** with a USB 3 data cable;
- stable Go2 power and a working remote controller or App emergency stop;
- an Ubuntu 22.04 x86_64 flashing host and a data-capable USB-C cable.

Supported hardware is limited to Go2 EDU, Orin NX 16GB, and D435i. Go2 Air/Pro,
Orin Nano, Orin NX 8GB, and other RealSense models are outside the deployment
support scope. The default inference source is the Go2 front camera. The D435i
is part of the project hardware bill of materials, but the runtime uses only
its RGB stream, not depth or IMU data.

## 2. Fixed Defaults and Local Values

These values apply when using the pinned patch without later network changes:

```text
Go2 controller            192.168.123.161
Orin wired address        192.168.123.18/24
Orin USB gadget address   192.168.55.1
Default Linux user        unitree
Default wired interface   enP8p1s0
```

The patch initially creates the `unitree` user with password `123`. Change it
with `passwd` on first login. Never store passwords, SSH keys, Hugging Face
tokens, device serial numbers, or other credentials in this repository.

The network interface can vary with the BSP or NetworkManager configuration.
Check it with `ip -br link`, then update
`camera.video_network_interface` in `go2_runtime.yaml`. Runtime, camera
preflight, and stop commands all use this setting.

The repository may be cloned anywhere. Its model, log, and script paths are
resolved relative to the repository. CycloneDDS is the only external prefix;
the validated path is `/home/unitree/cyclonedds/install`, or set
`CYCLONEDDS_HOME` to another installation.

Before deployment, run on the Orin:

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

Enter the local account password interactively for `sudo -v`. A management
computer connected to `192.168.123.0/24` must use another free address, such
as `192.168.123.222/24`; do not assign `.18` or `.161` to that computer.

## 3. Flash JetPack and Apply the Go2 Patch

Back up the existing system before flashing. The complete backup, APX Recovery,
JetPack 6.2.2 flash, acceptance, and rollback procedure is documented in the
[JetPack 6 upgrade guide](JETPACK6_UPGRADE.md).

The validated carrier-board patch is commit
`f7be6640865823c831feaaaaa7ba4e74c31808a7` from
[KevinLADLee/unitree-go2-jetpack6.x-patches](https://github.com/KevinLADLee/unitree-go2-jetpack6.x-patches).
Run the following on the Ubuntu x86_64 flashing host against a clean JetPack
6.2.2 BSP:

```bash
git clone https://github.com/KevinLADLee/unitree-go2-jetpack6.x-patches.git
cd unitree-go2-jetpack6.x-patches
git checkout f7be6640865823c831feaaaaa7ba4e74c31808a7

export L4T_DIR=/path/to/JetPack_6.2.2_Linux_JETSON_ORIN_NX_TARGETS/Linux_for_Tegra
sudo ./apply_unitree_bsp_patches.sh
sudo ./apply_unitree_rootfs_config.sh
```

This patch changes the hybrid USB device tree, PP.06 USB power, SSH, default
user, static wired network, and `extlinux` FDT. Read the upstream README,
LICENSE, NOTICE, and scripts before running them. Do not layer this patch and a
different Go2 BSP workflow onto the same `Linux_for_Tegra` tree.

Flash the NVMe using the upstream procedure:

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

After booting the flashed system, change the initial password, install your SSH
key, and keep ports 5801 and 5803 off public networks. Check the carrier-board
configuration:

```bash
cat /proc/device-tree/bus@0/padctl@3520000/ports/usb2-0/mode; echo
cat /proc/device-tree/bus@0/padctl@3520000/pads/usb3/lanes/usb3-2/status; echo
tr '\0' ' ' </proc/device-tree/bus@0/usb@3610000/phy-names; echo
for f in /sys/class/drm/card*-*/status; do echo "$f: $(cat "$f")"; done
lsusb -t
systemctl status unitree-pp6-usb-enable.service --no-pager
```

Expected results include `usb2-0=otg`, `usb3-2=okay`, USB 3 devices at
`5000M` or `10000M`, and an active PP.06 power service.

## 4. Install Runtime Dependencies

Run these commands on the Orin:

```bash
cd MiniCPM-RobotTrack

sudo apt-get update
sudo apt-get install -y \
  git curl python3-pip python3-dev python3-yaml cmake build-essential \
  ros-humble-cv-bridge ros-humble-realsense2-camera

python3 -m pip install --user -r requirements-runtime.txt
```

Install the official Unitree SDK2 Python 1.0.1. The validated Python package is
`cyclonedds==0.10.2`; the Go2 VideoClient also requires a CycloneDDS 0.10.2
native library compatible with ROS 2 Humble. If building it from source, pin
0.10.2 and install it to `/home/unitree/cyclonedds/install` or set
`CYCLONEDDS_HOME` accordingly.

Do not replace Jetson PyTorch, torchvision, or TensorRT with generic x86 or
PyPI builds. The validated versions are:

```text
torch       2.5.0a0+872d972e41.nv24.08
torchvision 0.20.0a0+afc54f7
TensorRT    10.7.0
CUDA        12.6
```

The NVIDIA PyTorch wheel downloaded to `vendor/` requires
`libcusparseLt.so.0`. Confirm an ARM64 `libcusparselt0` package is available
before installing it. When changing TensorRT packages, check the APT removal
plan before accepting it so CUDA, cuDNN, and driver components remain present.

## 5. Prepare Model Assets

Download the public base models and NVIDIA wheel:

```bash
python3 scripts/download_upstream_assets.py
```

DINOv3 is gated. Accept its license on Hugging Face first, then use
`hf auth login` or set `HF_TOKEN` for the upstream download script.

Download the complete MiniCPM-RobotTrack snapshot manually from
[the official model repository](https://huggingface.co/openbmb/MiniCPM-RobotTrack)
and place its contents under
`minicpm_robot_track/checkpoints/MiniCPM-RobotTrack/`. The snapshot includes the
fine-tuned MiniCPM4 backbone, tokenizer, custom model code, configuration, and
weights. Deployment loads it with `AutoModel.from_pretrained(...,
trust_remote_code=True)` and does not use a separately downloaded MiniCPM4
backbone.

Export both ONNX models from the official base models:

```bash
python3 -m pip install --user -r requirements-build.txt
./scripts/export_onnx.sh
```

See [Model and Deployment Assets](ASSETS.md) for the complete directory layout.
Model weights, ONNX files, and TensorRT engines are ignored by Git.

## 6. Select the Power Mode and Build TensorRT Engines

Switch to MAXN mode 0. `nvpmodel` may ask for confirmation and a reboot:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
nvpmodel -q
sudo jetson_clocks --show
./scripts/build_engines.sh
```

The expected outputs are:

```text
realworld/trt_artifacts/dino_patch_target_fp16.engine
realworld/trt_artifacts/siglip_pooled_target_maxn_fp16.engine
```

Both `trtexec` logs must end with `&&&& PASSED TensorRT.trtexec`. Build these
engines on every target Orin NX in its final JetPack, TensorRT, and power-mode
configuration; do not copy engines between devices.

## 7. Preflight and Dry Run

```bash
./scripts/preflight.sh
./check_go2_camera.py --frames 30
./go2_runtime.py run
```

Stop other VideoClient processes before checking the Go2 camera. In the
terminal and status JSON, confirm:

```text
mode=dry-run
camera_source=go2
actual_v=0
actual_w=0
```

Run for at least 60 seconds. Check `./go2_runtime.py status`, the web UI, service
logs, camera frames, and latency output. Repeat dry-run after changing
dependencies, weights, ONNX, engines, configuration, or control code.

To validate the D435i RGB path separately:

```bash
./go2_runtime.py run --camera-source d435i --mode dry-run
```

## 8. Live-Control Safety

Do not use live control until the camera, model, command timeout, disconnect
stop, and physical emergency stop have all been tested on a stand. The model
may still predict forward motion when no person is visible, so target-presence
gating must be validated for the intended environment.

Live control requires both flags below and interactive entry of `MOVE`:

```bash
./go2_runtime.py run --mode live --confirm-live-control
```

Default live-control limits are `vx=0.15 m/s` and `wz=0.30 rad/s`. Keep an
operator, the remote/App emergency stop, and a second terminal with
`./go2_runtime.py stop-control` ready. Do not launch live control with
`kill -9`, `nohup`, or an unattended background service.

## License

Project source is released under the top-level [Apache-2.0 License](../../LICENSE).
Review [Third-Party Notices](../THIRD_PARTY_NOTICES.md) before using or
redistributing model assets.

# Upgrade the Unitree Go2 EDU Orin NX to JetPack 6.2.2

**English** | [简体中文](JETPACK6_UPGRADE_zh-CN.md)

This guide covers inspection of the original JetPack 5 system, file-level and
full-partition backups, APX Recovery, flashing JetPack 6.2.2, post-flash
acceptance, and rollback. Read the entire procedure before beginning.

> [!CAUTION]
> Flashing or restoring writes QSPI and NVMe and can make the Orin unbootable.
> Keep stable power, verify the target device, retain an offline backup, and do
> not disconnect USB-C while an operation is running.

## 1. Target and Prerequisites

Validated target:

```text
Robot          Unitree Go2 EDU
Compute        Jetson Orin NX 16GB
Target OS      Ubuntu 22.04, Jetson Linux R36.5
Target release JetPack 6.2.2
Root storage   NVMe
Flashing host  Ubuntu 22.04 x86_64
```

Required equipment:

- the Go2 EDU and its Orin NX, with stable power;
- an Ubuntu 22.04 x86_64 host or VMware guest;
- a data-capable USB-C cable;
- at least 200 GB of free working space, with more space recommended for a
  full backup and a second copy;
- a separate physical disk or trusted offline location for backup copies.

Confirm the host architecture and storage before downloading anything:

```bash
uname -m
df -h /
```

The host must report `x86_64`. Do not use an ARM host for this workflow.

## 2. Prepare the Flashing Host

Choose a work directory on a large local disk:

```bash
export WORK_ROOT="/data/go2-jetpack"
export REPO_DIR="$WORK_ROOT/unitree-jetpack"
export TMPDIR="$WORK_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export LOG_DIR="$WORK_ROOT/logs"
export JETPACK_VERSION="6.2.2"

sudo mkdir -p "$WORK_ROOT" "$TMPDIR" "$LOG_DIR"
sudo chown -R "$USER:$USER" "$WORK_ROOT"
df -h "$WORK_ROOT"
```

These variables apply only to the current shell. Re-export them after opening a
new terminal or rebooting the host.

Install host dependencies:

```bash
sudo apt update
sudo apt install -y \
  git wget curl rsync usbutils qemu-user-static python3 python3-yaml \
  libxml2-utils nfs-kernel-server abootimg sshpass zstd
```

For VMware, also install:

```bash
sudo apt install -y open-vm-tools open-vm-tools-desktop
```

If the host uses NetworkManager commands from this guide:

```bash
sudo apt install -y network-manager
```

Confirm QEMU support:

```bash
command -v qemu-aarch64-static
qemu-aarch64-static --version | head -n 1
```

## 3. Host Networking

Backup and flashing use the USB-C APX/Recovery path. A wired Go2 connection is
useful for inspecting the original system but is not a substitute for USB-C.

Inspect host interfaces:

```bash
ip -br addr
nmcli device status
nmcli connection show
```

If connecting the host directly to the Go2 internal network, assign the host a
free address such as `192.168.123.222/24`. Do not use the Orin address
`192.168.123.18` or the Go2 controller address `192.168.123.161`.

Verify routes and connectivity:

```bash
ip route get 192.168.123.18
ping -c 3 192.168.123.18
```

## 4. Get the Go2 Flashing Repository

The complete backup/restore workflow uses the RoboLegion Go2 branch. Follow
the upstream repository's current access and license instructions:

```bash
cd "$WORK_ROOT"
git clone -b go2 https://github.com/legion1581/unitree-jetpack.git "$REPO_DIR"
cd "$REPO_DIR"
git branch --show-current
git rev-parse HEAD
git status
```

The branch must be `go2`. Record the commit used for the backup and flash so
that recovery can use the same tool version. If the upstream URL or repository
name changes, use the location named by that project's official documentation.

## 5. Inspect and Back Up the Original System

Connect over the Go2 wired network or USB gadget:

```bash
ssh unitree@192.168.123.18
# Or:
ssh unitree@192.168.55.1
```

On the original Orin, record the system and hardware state:

```bash
cat /etc/os-release
cat /etc/nv_tegra_release
uname -r
uname -m
cat /proc/device-tree/model 2>/dev/null
lsblk -o NAME,MODEL,SIZE,FSTYPE,LABEL,MOUNTPOINTS
df -h /
ip -br addr
ip route
sudo nvpmodel -q 2>/dev/null || true
ping -c 3 192.168.123.161
```

Create a file-level archive for source, configuration, calibration, maps, and
launch files that must be migrated selectively after flashing. Verify that the
archive can be listed before copying it to the host:

```bash
sudo tar -czf /var/tmp/go2_pre_jp6_files.tar.gz \
  /home/unitree \
  /etc/NetworkManager/system-connections \
  /etc/systemd/system

sudo chown unitree:unitree /var/tmp/go2_pre_jp6_files.tar.gz
tar -tzf /var/tmp/go2_pre_jp6_files.tar.gz >/dev/null
```

Adjust the source list to the files actually needed on the device. Do not use
this archive to overwrite all of `/etc`, `/opt/ros`, `/usr/local`, or another
JetPack 6 system directory.

Copy the archive to the host and test it again:

```bash
export FILE_BACKUP_DIR="$WORK_ROOT/file-backup"
mkdir -p "$FILE_BACKUP_DIR"
scp unitree@192.168.123.18:/var/tmp/go2_pre_jp6_files.tar.gz "$FILE_BACKUP_DIR/"
tar -tzf "$FILE_BACKUP_DIR/go2_pre_jp6_files.tar.gz" >/dev/null
```

The archive can contain network credentials, SSH configuration, or private
keys. Keep it in trusted storage and never publish it.

## 6. Initialize the JetPack 6.2.2 BSP

Initialize the BSP before the full backup so download, storage, and patch
problems are found early:

```bash
cd "$REPO_DIR"
export INIT_LOG="$LOG_DIR/jp622_init_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail

./go2_custom_jetpack.sh \
  -j "$JETPACK_VERSION" \
  init \
  2>&1 | tee "$INIT_LOG"
```

Verify the initialized BSP:

```bash
BSP_DIR="$REPO_DIR/bsp/6.2.2/Linux_for_Tegra"

test -f "$BSP_DIR/.g1-init-done"
test -x "$BSP_DIR/tools/kernel_flash/l4t_initrd_flash.sh"
test -x "$BSP_DIR/tools/backup_restore/l4t_backup_restore.sh"
grep -F "BSP ready:" "$INIT_LOG"
du -sh "$REPO_DIR/bsp/6.2.2"
df -h "$WORK_ROOT"
```

## 7. Enter APX Recovery

On the Orin:

```bash
sudo reboot --force forced-recovery
```

The SSH disconnect is expected. On the flashing host, confirm APX:

```bash
lsusb | grep -iE '0955|nvidia'
cd "$REPO_DIR"
./go2_custom_jetpack.sh status
```

Proceed only when the tool reports:

```text
APX — bootROM recovery (ready)
```

`RNDIS` means the Recovery initrd is already running and a new operation should
not be started. Return the device to normal boot, then enter APX again.

## 8. Create a Full Partition Backup

Use a unique backup name and retain the log:

```bash
cd "$REPO_DIR"
export BACKUP_NAME="factory-jp5-full-$(date +%Y%m%d-%H%M%S)"
export BACKUP_LOG="$LOG_DIR/${BACKUP_NAME}.log"
set -o pipefail

./go2_custom_jetpack.sh backup "$BACKUP_NAME" \
  2>&1 | tee "$BACKUP_LOG"
```

A successful log includes `Backup complete`, `Operation finishes`, and
`[+] backup saved to ...`. Root filesystem compression can run for a long time
without terminal output. Do not interrupt it while the archive is growing and
backup processes are active.

Set and inspect the final backup directory:

```bash
export BACKUP_DIR="$REPO_DIR/backups/$BACKUP_NAME"
du -sh "$BACKUP_DIR"
find "$BACKUP_DIR" -maxdepth 1 -type f -size +0 -print
```

The backup must contain a non-empty root filesystem archive, partition map,
NVMe partition images, GPT/MBR images, and `QSPI0.img`. File names may differ
slightly between upstream tool versions; the root archive may be `.tar.zst` or
`.tar.gz`.

Copy the accepted backup to another physical device:

```bash
sync
rsync -aH --info=progress2 \
  "$BACKUP_DIR/" \
  /path/to/second-disk/"$BACKUP_NAME"/
```

Do not flash until the backup and its second copy are present and readable.

## 9. Flash JetPack 6.2.2

After backup, return the device to normal boot or power-cycle it, then enter APX
again and confirm the `APX` status.

```bash
cd "$REPO_DIR"
./go2_custom_jetpack.sh status
git branch --show-current
```

The branch must be `go2`. Start the destructive QSPI and NVMe flash:

```bash
export FLASH_LOG="$LOG_DIR/jp622_flash_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail

./go2_custom_jetpack.sh \
  -j "$JETPACK_VERSION" \
  flash all \
  2>&1 | tee "$FLASH_LOG"
```

Before entering `yes`, confirm that the prompt names JetPack 6.2.2, operation
`all`, and `bsp/6.2.2/Linux_for_Tegra`. Do not add `--super` to this workflow.

During flashing:

- keep Go2 and Orin power stable;
- keep USB-C connected and assigned to the Ubuntu VM;
- prevent the host from sleeping;
- do not close VMware or press `Ctrl+C`.

The final success marker is:

```text
[+] flash 'all' complete
```

If it fails, inspect the retained log before retrying:

```bash
tail -n 150 "$FLASH_LOG"
grep -iE 'error|failed|\[x\]|timeout|no such file|cannot|not found' "$FLASH_LOG"
```

## 10. First Boot and Acceptance

Allow two to five minutes after flashing, then check both interfaces:

```bash
ping -c 5 192.168.123.18
ping -c 5 192.168.55.1
```

Remove stale SSH host keys and log in:

```bash
ssh-keygen -R 192.168.123.18
ssh-keygen -R 192.168.55.1
ssh unitree@192.168.123.18
```

Change the initial password immediately:

```bash
passwd
```

On the Orin, verify OS, L4T, architecture, storage, network, and services:

```bash
grep PRETTY_NAME /etc/os-release
cat /etc/nv_tegra_release
uname -r
uname -m
lsblk -o NAME,MODEL,SIZE,FSTYPE,LABEL,MOUNTPOINTS
df -h /
ip -br addr
ip route
ping -c 3 192.168.123.161
sudo nvpmodel -q
systemctl --failed
dmesg -T | grep -iE 'error|failed|timeout|nvme|pcie|usb' | tail -n 100
```

Expected baseline:

```text
Ubuntu 22.04
L4T R36.5
5.15.185-tegra
aarch64
NVMe mounted as the root filesystem
192.168.123.18 reachable
192.168.55.1 reachable
192.168.123.161 reachable
No critical failed services
```

A successful BSP flash does not guarantee that the complete CUDA development
stack, PyTorch, ROS, or application dependencies are installed. Inspect the
NVIDIA packages independently:

```bash
dpkg-query -W nvidia-jetpack 2>/dev/null || true
nvcc --version 2>/dev/null || echo "[INFO] nvcc is not installed"
dpkg -l | grep -E 'nvidia-jetpack|cuda-toolkit|libcudnn|tensorrt|libnvinfer|vpi'
```

Install only versions compatible with JetPack 6.2.2, CUDA 12.6, and aarch64.
Do not copy CUDA, cuDNN, TensorRT, ROS runtime libraries, or Python binaries
from the old JetPack 5 system.

## 11. Restore the Full Backup

Restore is also destructive and overwrites the current QSPI and NVMe. Confirm
the selected backup contains all expected, non-empty files, then place the
device in APX Recovery:

```bash
export BACKUP_DIR="/path/to/actual-backup"
find "$BACKUP_DIR" -maxdepth 1 -type f -size +0 -print

cd "$REPO_DIR"
./go2_custom_jetpack.sh status
```

Proceed only when status is `APX — bootROM recovery (ready)`, then restore:

```bash
./go2_custom_jetpack.sh restore "$BACKUP_DIR"
```

The name of a directory under `backups/` may be used instead:

```bash
./go2_custom_jetpack.sh restore <BACKUP_NAME>
```

Do not disconnect power or USB during restore.

## 12. Rebuild the Unitree Software Environment

Migrate source, configuration, calibration files, maps, launch files, topic
names, frame definitions, and startup logic selectively. Reinstall compatible
JetPack 6 versions of:

```text
ROS 2 Humble
unitree_ros2
CycloneDDS configuration
front-camera integration
LiDAR, IMU, and odometry interfaces
SLAM and navigation software
user applications
```

Do not overwrite JetPack 6 system trees with the old `/boot`, `/lib/modules`,
all of `/etc`, all of `/opt/ros`, all of `/usr/local`, or old NVIDIA libraries.
Verify camera, LiDAR, IMU, odometry, localization, and motion control
independently before deployment.

## 13. VMware USB Notes

During the workflow, the NVIDIA USB device can change identity:

```text
NVIDIA APX
-> NVIDIA Linux for Tegra / Recovery RNDIS
-> normal Jetson USB gadget
```

Whenever VMware asks where to connect the new USB device, attach it to the
Ubuntu guest. If `status` cannot see APX, check `lsusb`, the physical cable,
the VM's USB controller, and host USB ownership before changing scripts.

## Final Checklist

Before flashing:

- correct Go2 EDU / Orin NX target identified;
- stable power and data-capable USB-C available;
- working directory has sufficient free space;
- file-level archive copied off the Orin and readable;
- full partition backup accepted and copied to a second device;
- repository branch and commit recorded;
- device status is APX.

After flashing:

- flash log contains `[+] flash 'all' complete`;
- password changed and SSH access restored;
- L4T R36.5, aarch64, and NVMe root verified;
- Go2 controller and both Orin network paths checked;
- no critical failed services or kernel errors;
- JetPack 6 application dependencies rebuilt rather than copied from JetPack 5.

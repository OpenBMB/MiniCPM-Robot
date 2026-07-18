# Unitree Go2 EDU Orin NX 刷写 JetPack 6.2.2 教程

[English](JETPACK6_UPGRADE.md) | **简体中文**

> 上游刷机工具适用设备：Unitree Go2 EDU 扩展坞中的 Jetson Orin NX / Orin Nano
> MiniCPM-RobotTrack 已验证部署设备：Jetson Orin NX 16GB、内置 NVMe
> 目标环境：JetPack 6.2.2 / L4T 36.5.0 / Ubuntu 22.04 / Linux 5.15.185-tegra
> 刷写工具：RoboLegion `unitree-jetpack` 仓库 `go2` 分支
> 执行要求：操作前记录仓库 commit，并以实际脚本输出为准

---

## 1. 教程说明

以下流程使用一台 **x86_64 Ubuntu 主机**，通过 Go2 扩展坞的 **USB-C 刷写接口**完成：

1. 检查原 Orin 的系统、存储和网络；
2. 生成文件级迁移备份；
3. 生成可用于回退的完整分区备份；
4. 刷写 JetPack 6.2.2；
5. 验证系统版本、NVMe、USB gadget 和 Go2 内部网络；
6. 保留后续恢复 Unitree/ROS 软件栈所需的资料。

RoboLegion 的 Go2 分支会针对 Unitree 定制载板应用设备树、USB Recovery、MB2 和 rootfs 配置补丁，并通过 Recovery initrd 写入 QSPI 和 NVMe，无需拆卸 SSD。

> [!IMPORTANT]
> `flash all` 会覆盖 **QSPI 和 NVMe rootfs**。
> 开始刷写前，必须生成并校验完整备份。

> [!NOTE]
> 刷写后得到的是 JetPack 基础系统。原设备中的 Unitree 应用、ROS 工作区、SLAM、地图和导航程序不会自动保留，需要后续按 Ubuntu 22.04 / ROS 2 Humble 环境重新安装或迁移。

---

## 2. 目标版本

JetPack 6.2.2 对应：

| 组件 | 版本 |
|---|---|
| JetPack | 6.2.2 |
| Jetson Linux / L4T | 36.5.0 |
| Ubuntu rootfs | 22.04 |
| Linux 内核 | 5.15.185-tegra |
| 架构 | aarch64 |

以下步骤使用标准配置，不添加：

```text
--super
```

不使用 `--super` 可以避免额外启用高功耗模式。刷写完成后仍应执行：

```bash
sudo nvpmodel -q
```

确认当前实际功耗模式。

---

# 第一部分：硬件与环境准备

## 3. 所需硬件

### 必需

- Unitree Go2 EDU 扩展坞；
- Jetson Orin NX 16GB；上游刷机工具也支持 Orin Nano，但后者不在 MiniCPM-RobotTrack 部署支持范围内；
- 稳定供电的 Go2；
- x86_64 Ubuntu 主机；
- 支持数据传输的 USB-C 数据线；
- 空间充足的数据盘。

### 可选

- 一根连接 Go2 内部网络的网线；
- 一块额外硬盘，用于保存第二份完整备份；
- 显示器和 USB-C 转 DP/HDMI 适配器，用于本地显示调试。

> [!WARNING]
> Go2 扩展坞可能有多个 USB-C 接口。刷写时必须使用能够让主机发现 NVIDIA `0955:*` USB 设备的刷写接口。
> DisplayPort 输出接口不能代替 Recovery 刷写接口。

---

## 4. 推荐的刷机主机

仓库要求：

```text
具有 sudo 权限的 x86_64 Ubuntu 主机
```

推荐刷机主机系统：

```text
Ubuntu 22.04 x86_64
```

可以使用：

- Ubuntu 22.04 物理机；
- VMware 中的 Ubuntu 22.04；
- 其他能够稳定接管 NVIDIA APX / RNDIS USB 设备的 x86_64 Ubuntu 环境。

### 检查主机架构

```bash
lsb_release -ds
uname -m
```

预期：

```text
Ubuntu 22.04.x LTS
x86_64
```

如果 `uname -m` 输出：

```text
aarch64
```

说明当前终端位于 Jetson 上，而不是刷机主机。

---

## 5. VMware 建议配置

使用 VMware 时，推荐：

| 项目 | 建议 |
|---|---|
| 系统 | Ubuntu 22.04 x86_64 |
| CPU | 4 核以上，推荐 8 核 |
| 内存 | 8GB 以上，推荐 16GB |
| 网络 1 | NAT，用于访问 GitHub、NVIDIA 和 Ubuntu 软件源 |
| 网络 2 | 可选，桥接到连接 Go2 的物理网口 |
| USB | NVIDIA APX、Recovery RNDIS 和正常 USB gadget 均连接到 Ubuntu 虚拟机 |

刷写期间：

- 不要关闭 VMware；
- 不要让 Windows 主机睡眠；
- USB 设备切换模式时，继续将其连接到 Ubuntu 虚拟机；
- 不要在脚本运行时拔线或切换 USB 所属系统。

---

## 6. 工作盘空间

刷机主机需要保存：

```text
JetPack BSP
sample rootfs
下载缓存
Recovery 临时镜像
文件级迁移备份
完整分区备份
刷写日志
```

备份大小主要由原系统的实际使用量决定，不由 NVMe 标称容量直接决定。

先在原 Orin 上检查：

```bash
df -h /

sudo blockdev --getsize64 /dev/nvme0n1 |
numfmt --to=iec
```

### 磁盘占用参考

```text
NVMe 标称容量：约 500GB
根分区容量：约 467GB
原系统实际使用：约 21GB

文件级迁移备份：约 3.2GB
完整分区备份目录：约 11GB
根分区压缩文件：约 11.35GB
```

完整分区备份还包含：

```text
NVMe 其他分区
GPT 与 MBR
QSPI
分区映射文件
```

### 建议预留

以下建议基于参考设备：原系统实际使用约 21GB，文件级备份约 3.2GB，完整分区备份目录约 11GB。

```text
仅保存一份完整备份：
建议至少预留约 30GB 可用空间

完成上述规模的 BSP 初始化、完整备份和刷写：
最低预留约 100GB 可用空间

为下载缓存、重复操作和第二份备份保留余量：
建议预留至少 200GB 可用空间

原系统实际使用量明显更大，或需要保存多个 BSP、
多个完整备份和后续编译环境：
建议预留 150GB～200GB 或更多
```

这些数值不是所有设备的固定下限。应根据原系统实际使用量和计划保存的 BSP、备份数量增加空间。

在高占用步骤前后执行：

```bash
df -h "$WORK_ROOT"
```

---

## 7. 设置教程变量

以下命令使用这些通用目录：

```bash
export WORK_ROOT="/data/go2-jetpack"
export REPO_DIR="$WORK_ROOT/unitree-jetpack"
export TMPDIR="$WORK_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export LOG_DIR="$WORK_ROOT/logs"
export JETPACK_VERSION="6.2.2"
```

将 `/data/go2-jetpack` 替换为空间充足的数据盘目录。

创建目录：

```bash
sudo mkdir -p \
  "$WORK_ROOT" \
  "$TMPDIR" \
  "$LOG_DIR"

sudo chown -R "$USER:$USER" "$WORK_ROOT"

touch "$WORK_ROOT/write_test"
rm "$WORK_ROOT/write_test"

echo '[OK] 工作目录可写'
df -h "$WORK_ROOT"
```

这些 `export` 变量只对当前终端会话有效。重新打开终端、重启虚拟机或隔天继续操作时，应重新执行变量配置，并确认 `REPO_DIR`、`LOG_DIR`、`BACKUP_NAME` 和 `BACKUP_DIR` 指向实际目录。

预期：

```text
[OK] 工作目录可写
```

---

## 8. 安装主机依赖

```bash
sudo apt update

sudo apt install -y \
  git \
  wget \
  curl \
  rsync \
  usbutils \
  qemu-user-static \
  python3 \
  python3-yaml \
  libxml2-utils \
  nfs-kernel-server \
  abootimg \
  sshpass \
  zstd
```

VMware 环境可额外安装：

```bash
sudo apt install -y \
  open-vm-tools \
  open-vm-tools-desktop
```

若系统中没有 `nmcli`，而且需要使用下文的 NetworkManager 配置命令，再安装：

```bash
sudo apt install -y network-manager
```

验证：

```bash
command -v qemu-aarch64-static
qemu-aarch64-static --version | head -n 1
```

典型输出：

```text
/usr/bin/qemu-aarch64-static
qemu-aarch64 version ...
```

---

# 第二部分：网络与仓库准备

## 9. 刷写是否需要网线

备份、恢复和刷写的核心链路是：

```text
Ubuntu 主机
→ USB-C
→ APX / Recovery initrd
→ QSPI 和 NVMe
```

因此：

- **USB-C 数据线是必需的**；
- **网线不是刷写必需条件**；
- 网线主要用于刷机前后 SSH 和 Go2 内部网络检查。

常见地址：

| 地址 | 用途 |
|---|---|
| `192.168.123.18` | Orin 的 Go2 有线地址 |
| `192.168.55.1` | Orin 的 USB gadget 地址 |
| `192.168.123.161` | Go2 内部主控 |

若设备地址已修改，应以实际网络配置为准。

---

## 10. 可选：配置连接 Go2 的主机网卡

先查看网卡：

```bash
ip -br addr
nmcli device status
nmcli connection show
```

定义连接 Go2 的网卡和 NetworkManager 连接名：

```bash
export GO2_HOST_IF="<连接 Go2 的主机网卡>"
export GO2_NM_CONN="<该网卡对应的连接名>"
```

下面的 `192.168.123.222` 是主机侧示例地址。使用前应确认它未被其他设备占用；也可以选择 `192.168.123.0/24` 网段中的其他空闲地址，但不要使用 Orin 的 `192.168.123.18` 或 Go2 主控的 `192.168.123.161`。

配置静态地址：

```bash
sudo nmcli connection modify "$GO2_NM_CONN" \
  connection.interface-name "$GO2_HOST_IF" \
  ipv4.method manual \
  ipv4.addresses 192.168.123.222/24 \
  ipv4.gateway "" \
  ipv4.dns "" \
  ipv4.never-default yes \
  ipv6.method disabled

sudo nmcli connection down "$GO2_NM_CONN" 2>/dev/null || true
sudo nmcli connection up "$GO2_NM_CONN"
```

检查：

```bash
ip -br addr show "$GO2_HOST_IF"
ip route get 192.168.123.18
ip route get 1.1.1.1
```

到 Go2 的路由应类似：

```text
192.168.123.18 dev <GO2_HOST_IF> src 192.168.123.222
```

互联网流量应继续走 NAT、Wi-Fi 或其他上联网卡。

测试：

```bash
ping -c 3 192.168.123.18
ping -c 3 1.1.1.1
getent hosts github.com
```

---

## 11. 获取 RoboLegion Go2 分支

```bash
cd "$WORK_ROOT"

git clone -b go2 \
  https://github.com/legion1581/unitree-jetpack.git \
  "$REPO_DIR"

cd "$REPO_DIR"
chmod +x go2_custom_jetpack.sh
```

检查分支并记录实际提交：

```bash
git branch --show-current

git rev-parse HEAD |
tee "$LOG_DIR/unitree_jetpack_commit.txt"

git status
```

预期：

```text
go2
<当前提交 SHA>
```

记录提交 SHA 很重要，因为仓库后续更新可能改变脚本内部行为。

### GitHub 下载故障

先检查：

```bash
resolvectl query github.com
curl -I --connect-timeout 15 https://github.com
```

普通克隆仍失败时，可临时使用：

```bash
git -c http.version=HTTP/1.1 clone \
  --depth 1 \
  --branch go2 \
  --single-branch \
  https://github.com/legion1581/unitree-jetpack.git \
  "$REPO_DIR"
```

---

# 第三部分：检查与备份原系统

## 12. 检查原 Orin

登录原 Orin：

```bash
ssh unitree@192.168.123.18
```

也可以通过正常启动后的 USB gadget：

```bash
ssh unitree@192.168.55.1
```

执行：

```bash
echo '===== SYSTEM ====='
cat /etc/os-release
cat /etc/nv_tegra_release
uname -r
uname -m

echo
echo '===== MODEL ====='
cat /proc/device-tree/model 2>/dev/null
echo

echo
echo '===== MEMORY ====='
free -h

echo
echo '===== STORAGE ====='
lsblk -o NAME,MODEL,SIZE,FSTYPE,LABEL,MOUNTPOINTS
df -h /

echo
echo '===== NETWORK ====='
ip -br addr
ip route

echo
echo '===== POWER MODE ====='
sudo nvpmodel -q 2>/dev/null || true

echo
echo '===== GO2 CONTROLLER ====='
ping -c 3 192.168.123.161
```

建议将结果保存到刷机主机：

```bash
ssh unitree@192.168.123.18 \
  'cat /etc/os-release;
   cat /etc/nv_tegra_release;
   uname -a;
   lsblk -o NAME,MODEL,SIZE,FSTYPE,LABEL,MOUNTPOINTS;
   df -h /;
   ip -br addr;
   ip route;
   sudo nvpmodel -q 2>/dev/null || true' \
  | tee "$LOG_DIR/pre_flash_device_info.txt"
```

---

## 13. 文件级迁移备份

文件级备份用于保存：

```text
源码
用户工作区
配置
标定
地图
启动脚本
旧服务信息
```

它不能恢复：

```text
QSPI
GPT/MBR
完整分区结构
原系统全部运行环境
```

为避免远程 `sudo tar` 无法交互或 SSH 中断，建议先在 Orin 本地打包，再使用 SCP 下载。

### 13.1 在原 Orin 上生成目录清单

```bash
FILE_BACKUP="/var/tmp/go2_pre_jp6_files.tar.gz"
FILE_LIST="/var/tmp/go2_pre_jp6_paths.txt"

sudo rm -f "$FILE_BACKUP" "$FILE_LIST"

for PATH_ITEM in \
  /unitree \
  /home/unitree \
  /home/pi \
  /etc \
  /opt/ros \
  /usr/local \
  /upgradePythonServer
do
  if sudo test -e "$PATH_ITEM"; then
    echo "$PATH_ITEM" |
    sudo tee -a "$FILE_LIST" >/dev/null
  else
    echo "[SKIP] 不存在：$PATH_ITEM"
  fi
done

echo '===== 将要备份的目录 ====='
sudo cat "$FILE_LIST"
```

### 13.2 打包

```bash
sudo tar \
  --acls \
  --xattrs \
  --numeric-owner \
  -czf "$FILE_BACKUP" \
  -T "$FILE_LIST"

sudo chown unitree:unitree "$FILE_BACKUP"
```

检查：

```bash
ls -lh /var/tmp/go2_pre_jp6_files.tar.gz
tar -tzf /var/tmp/go2_pre_jp6_files.tar.gz >/dev/null \
  && echo "文件级备份压缩包：OK" \
  || echo "文件级备份压缩包：FAILED"
```

若 `tar` 报告文件不可读、文件在打包过程中发生变化或命令非零退出，应先检查原因并重新打包。文件级备份只是迁移辅助材料，不能代替后续完整分区备份。

### 13.3 下载到刷机主机

在刷机主机执行：

```bash
export FILE_BACKUP_DIR="$WORK_ROOT/file-backup"

mkdir -p "$FILE_BACKUP_DIR"

scp \
  unitree@192.168.123.18:/var/tmp/go2_pre_jp6_files.tar.gz \
  "$FILE_BACKUP_DIR/"
```

检查归档是否可以读取：

```bash
cd "$FILE_BACKUP_DIR"

tar -tzf go2_pre_jp6_files.tar.gz >/dev/null \
  && echo "文件级备份压缩包：OK" \
  || echo "文件级备份压缩包：FAILED"
```

预期：

```text
go2_pre_jp6_files.tar.gz: OK
文件级备份压缩包：OK
```

> [!WARNING]
> 文件级备份用于审计和选择性迁移。
> 刷写完成后，禁止将旧 `/etc`、`/opt/ros`、`/usr/local` 或其他系统目录整体覆盖到 JP6。

> [!CAUTION]
> 该压缩包可能包含网络配置、SSH 配置、Wi-Fi 凭据或私钥，只能保存在可信存储中，不要公开上传。

---

# 第四部分：准备 BSP 与完整分区备份

## 14. 初始化 JetPack 6.2.2 BSP

`flash` 会在需要时自动初始化 BSP。建议在完整备份前先显式构建 6.2.2，以便提前发现下载、磁盘和补丁问题。

```bash
cd "$REPO_DIR"

export INIT_LOG="$LOG_DIR/jp622_init_$(date +%Y%m%d_%H%M%S).log"

set -o pipefail

./go2_custom_jetpack.sh \
  -j "$JETPACK_VERSION" \
  init \
  2>&1 |
tee "$INIT_LOG"
```

检查：

```bash
BSP_DIR="$REPO_DIR/bsp/6.2.2/Linux_for_Tegra"

if \
  test -f "$BSP_DIR/.g1-init-done" &&
  test -x "$BSP_DIR/tools/kernel_flash/l4t_initrd_flash.sh" &&
  test -x "$BSP_DIR/tools/backup_restore/l4t_backup_restore.sh"
then
  echo "JetPack 6.2.2 BSP：OK"
else
  echo "JetPack 6.2.2 BSP：FAILED"
fi

grep -F "BSP ready:" "$INIT_LOG" \
  || echo "[WARNING] 初始化日志中没有找到 BSP ready"

du -sh "$REPO_DIR/bsp/6.2.2"
df -h "$WORK_ROOT"
```

预期：

```text
JetPack 6.2.2 BSP：OK
[+] BSP ready: .../bsp/6.2.2/Linux_for_Tegra
```

### 检查已构建 BSP

`backup` 会从已经构建完成的 BSP 中选择版本较高的 Recovery initrd。备份前检查：

```bash
find "$REPO_DIR/bsp" \
  -path '*/Linux_for_Tegra/.g1-init-done' \
  -printf '%h\n' 2>/dev/null |
sort -V
```

若输出中只有：

```text
.../bsp/6.2.2/Linux_for_Tegra
```

说明当前只构建了 JetPack 6.2.2 BSP，备份会复用 6.2.2 Recovery initrd。

---

## 15. 进入 APX Recovery

### 在 Orin 上执行

```bash
sudo reboot --force forced-recovery
```

SSH 可能出现：

```text
client_loop: send disconnect: Broken pipe
```

这只表示 SSH 连接断开。最终是否进入 Recovery，必须在刷机主机确认。

### 在刷机主机检查

```bash
lsusb | grep -iE '0955|nvidia'

cd "$REPO_DIR"
./go2_custom_jetpack.sh status
```

`lsusb` 典型输出：

```text
Bus 00x Device 00x: ID 0955:xxxx NVIDIA Corp.
```

脚本预期：

```text
APX — bootROM recovery (ready)
```

状态说明：

| 状态 | 含义 | 能否开始新操作 |
|---|---|---|
| APX | BootROM Recovery | 可以 |
| RNDIS | Recovery initrd 已运行 | 不应直接开始另一轮操作 |
| 正常系统 | Orin 已启动 | 需要重新进入 Recovery |

---

## 16. 执行完整分区备份

设置唯一备份名：

```bash
cd "$REPO_DIR"

export BACKUP_NAME="factory-jp5-full-$(date +%Y%m%d-%H%M%S)"
export BACKUP_LOG="$LOG_DIR/${BACKUP_NAME}.log"

set -o pipefail

./go2_custom_jetpack.sh backup "$BACKUP_NAME" \
  2>&1 |
tee "$BACKUP_LOG"
```

日志开头应确认：

```text
backup/restore via the JetPack 6.2.2 recovery initrd
```

备份过程通常会出现：

```text
nvbackup_partitions.sh: Start backing up ...
nvbackup_partitions.sh: Success backing up ...
```

最终成功标志：

```text
nvbackup_partitions.sh: Backup complete
Operation finishes. You can manually reset the device
[*] collecting images -> ...
[+] backup saved to ...
```

---

## 17. 监控长时间备份

根分区压缩可能长时间没有新输出。不同工具版本可能生成 `.tar.zst` 或 `.tar.gz`，因此动态匹配：

```bash
IMAGE_DIR="$REPO_DIR/bsp/6.2.2/Linux_for_Tegra/tools/backup_restore/images"

watch -n 10 "
date

echo '===== ROOTFS IMAGES ====='
find '$IMAGE_DIR' \
  -maxdepth 1 \
  -type f \
  -name 'nvme0n1p1.tar.*' \
  -printf '%f  %s bytes  %TY-%Tm-%Td %TH:%TM:%TS\n' \
  2>/dev/null

echo
echo '===== BACKUP PROCESSES ====='
ps -eo pid,stat,etime,%cpu,%mem,cmd |
grep -E '[t]ar|[z]std|[g]zip|[p]igz|[l]4t_backup_restore|[n]vbackup'

echo
echo '===== DISK SPACE ====='
df -h '$WORK_ROOT'
"
```

正常现象：

- 根分区文件持续增大；
- 修改时间持续变化；
- `tar`、`zstd`、`gzip` 或 backup 进程仍存在；
- 工作盘没有接近满容量。

不要因为数分钟没有终端输出就中断。

---

## 18. 完整备份验收

设置实际备份目录：

```bash
export BACKUP_DIR="$REPO_DIR/backups/$BACKUP_NAME"
```

检查总大小和文件：

```bash
du -sh "$BACKUP_DIR"

find "$BACKUP_DIR" \
  -maxdepth 1 \
  -type f \
  -printf '%-14s %f\n' |
sort
```

通常应包含：

```text
nvme0n1p1.tar.zst 或 nvme0n1p1.tar.gz
nvme0n1p2_bak.img
...
其他 NVMe 分区镜像
nvme0n1_gptbackup.img
nvme0n1_gptmbr.img
nvpartitionmap.txt
QSPI0.img
```

分区数量以设备实际分区表为准。

自动检查：

```bash
test -s "$BACKUP_DIR/nvpartitionmap.txt" \
  && echo "分区映射：OK" \
  || echo "分区映射：FAILED"

ROOTFS=$(
  find "$BACKUP_DIR" \
    -maxdepth 1 \
    -type f \
    -name 'nvme0n1p1.tar.*' \
    -print |
  head -n 1
)

test -n "$ROOTFS" && test -s "$ROOTFS" \
  && echo "系统分区：OK — $ROOTFS" \
  || echo "系统分区：FAILED"

test -s "$BACKUP_DIR/QSPI0.img" \
  && echo "QSPI：OK" \
  || echo "QSPI：FAILED"

test -s "$BACKUP_DIR/nvme0n1_gptbackup.img" \
  && echo "GPT：OK" \
  || echo "GPT：FAILED"

test -s "$BACKUP_DIR/nvme0n1_gptmbr.img" \
  && echo "GPT/MBR：OK" \
  || echo "GPT/MBR：FAILED"
```

预期：

```text
分区映射：OK
系统分区：OK
QSPI：OK
GPT：OK
GPT/MBR：OK
```

### 确认备份文件

```bash
sync
find "$BACKUP_DIR" -maxdepth 1 -type f -size +0 -print
```

通过标准：

```text
分区镜像、分区映射、GPT/MBR 和 QSPI 文件均存在且大小非零
日志出现 [+] backup saved to ...
```

建议再复制一份到其他物理存储：

```bash
rsync -aH --info=progress2 \
  "$BACKUP_DIR/" \
  /path/to/second-disk/"$BACKUP_NAME"/
```

---

# 第五部分：刷写 JetPack 6.2.2

## 19. 重新进入 APX

完整备份结束后，设备可能仍处于 Recovery RNDIS。

先让设备退出 Recovery initrd并正常启动或重新上电，再执行：

```bash
sudo reboot --force forced-recovery
```

回到刷机主机确认：

```bash
cd "$REPO_DIR"
./go2_custom_jetpack.sh status
```

必须重新看到：

```text
APX — bootROM recovery (ready)
```

---

## 20. 执行 `flash all`

```bash
cd "$REPO_DIR"

export FLASH_LOG="$LOG_DIR/jp622_flash_$(date +%Y%m%d_%H%M%S).log"

set -o pipefail

./go2_custom_jetpack.sh \
  -j "$JETPACK_VERSION" \
  flash all \
  2>&1 |
tee "$FLASH_LOG"
```

以下刷写命令不添加：

```text
--super
```

`flash all` 会写入：

```text
QSPI + NVMe rootfs
```

输入 `yes` 前，先确认当前仓库分支：

```bash
git branch --show-current
```

预期：

```text
go2
```

然后确认脚本提示中的：

```text
JetPack 版本：6.2.2
操作类型：all
BSP 路径：bsp/6.2.2/Linux_for_Tegra
```

当前脚本的确认提示通常类似：

```text
DESTRUCTIVE: flash JetPack 6.2.2 'all' to the board from ...
Type 'yes' to continue:
```

刷写期间必须保持：

- Go2 和 Orin 稳定供电；
- USB-C 不断开；
- VMware 不关闭；
- USB 设备始终属于 Ubuntu 虚拟机；
- Windows 主机不睡眠；
- 不按 `Ctrl+C`。

最终成功标志：

```text
Successfully flashed the QSPI.
Successfully flashed the external device.
Flash is successful
[+] flash 'all' complete
```

最关键的是：

```text
[+] flash 'all' complete
```

若失败，先查看日志：

```bash
tail -n 150 "$FLASH_LOG"

grep -iE \
'error|failed|\[x\]|timeout|no such file|cannot|not found' \
"$FLASH_LOG"
```

不要在未分析失败原因前连续重复刷写。

---

# 第六部分：首次启动与基础验收

## 21. 首次启动

刷写结束后等待约 2～5 分钟：

```bash
ping -c 5 192.168.123.18
ping -c 5 192.168.55.1
```

当前 Go2 分支的常见默认配置：

| 项目 | 默认值 |
|---|---|
| 用户 | `unitree` |
| 初始密码 | 以当前仓库 README / 脚本输出为准 |
| 主机名 | `ubuntu` |
| 有线接口 | `enP8p1s0` |
| Go2 有线地址 | `192.168.123.18` |
| USB gadget 地址 | `192.168.55.1` |

首次登录后应修改密码：

```bash
passwd
```

---

## 22. 清理旧 SSH 指纹

```bash
ssh-keygen -R 192.168.123.18
ssh-keygen -R 192.168.55.1
```

登录：

```bash
ssh unitree@192.168.123.18
```

或：

```bash
ssh unitree@192.168.55.1
```

---

## 23. 验证系统版本

在 Orin 上执行：

```bash
echo '===== OS ====='
grep PRETTY_NAME /etc/os-release

echo
echo '===== L4T ====='
cat /etc/nv_tegra_release

echo
echo '===== KERNEL ====='
uname -r

echo
echo '===== ARCH ====='
uname -m
```

预期：

```text
PRETTY_NAME="Ubuntu 22.04.x LTS"
# R36 (release), REVISION: 5.0, ...
5.15.185-tegra
aarch64
```

---

## 24. 验证 NVMe

```bash
lsblk -o NAME,MODEL,SIZE,FSTYPE,LABEL,MOUNTPOINTS
df -h /
```

确认：

- `nvme0n1` 存在；
- 根分区挂载到 `/`；
- 根分区容量合理；
- 文件系统不是只读状态；
- 剩余空间正常。

---

## 25. 验证网络

```bash
ip -br addr
ip route
```

应能看到：

```text
192.168.123.18/24
192.168.55.1/24
```

JetPack 6.2.2 的有线接口通常为：

```text
enP8p1s0
```

USB gadget 可能显示为：

```text
l4tbr0
usb0
usb1
```

因此应根据 IP 地址判断，而不是只依赖接口名称。

检查 Go2 内部主控：

```bash
ping -c 3 192.168.123.161
```

预期：

```text
3 packets transmitted, 3 received, 0% packet loss
```

---

## 26. 验证功耗与系统服务

```bash
sudo nvpmodel -q

systemctl --failed
```

检查关键内核错误：

```bash
dmesg -T |
grep -iE 'error|failed|timeout|nvme|pcie|usb' |
tail -n 100
```

应重点关注：

- NVMe 挂载失败；
- PCIe 链路失败；
- USB gadget 无法启动；
- 有线网卡无法建立链路；
- 文件系统错误。

当以下项目均通过后，可以认定基础刷写成功：

```text
Ubuntu 22.04
L4T R36.5
5.15.185-tegra
aarch64
NVMe 根分区正常
192.168.123.18 可达
192.168.55.1 可达
192.168.123.161 可达
系统无关键失败服务
```

## 27. NVIDIA 计算环境检查

JetPack 6.2.2 配套的 NVIDIA AI 计算栈版本为：

| 组件 | 版本 |
|---|---|
| CUDA Toolkit | 12.6.10 |
| TensorRT | 10.3.0 |
| cuDNN | 9.3.0 |
| VPI | 3.2 |
| DLA | 3.14 |

需要区分：

```text
Jetson Linux / BSP 刷写成功
≠
完整 CUDA 开发工具链已经安装
```

检查当前安装状态：

```bash
dpkg-query -W nvidia-jetpack 2>/dev/null || true

nvcc --version 2>/dev/null \
  || echo "[INFO] 当前未找到 nvcc"

dpkg -l |
grep -E \
'nvidia-jetpack|cuda-toolkit|libcudnn|tensorrt|libnvinfer|vpi' |
head -n 100
```

若 `nvcc` 不存在，不代表底层刷写失败。只有后续需要编译 CUDA 程序或部署 TensorRT 模型时，才需要根据 JetPack 6.2.2 的 NVIDIA 软件源安装相应组件。

PyTorch、ONNX Runtime 等框架也不保证随基础刷写自动安装，应另行选择与 JetPack 6.2.2、CUDA 12.6 和 aarch64 兼容的版本。不要从旧 JP5 系统直接复制 CUDA、cuDNN、TensorRT 或 Python 二进制包到 JP6。

---

# 第七部分：恢复与回退

## 28. 使用完整备份恢复

恢复会覆盖当前设备的 QSPI 和 NVMe，同样属于破坏性操作。

先确认备份目录和文件存在：

```bash
export BACKUP_DIR="/path/to/actual-backup"
find "$BACKUP_DIR" -maxdepth 1 -type f -size +0 -print
```

确认输出包含分区镜像、分区映射、GPT/MBR 和 QSPI 文件后再继续。

将设备置于 APX：

```bash
cd "$REPO_DIR"
./go2_custom_jetpack.sh status
```

预期：

```text
APX — bootROM recovery (ready)
```

按完整路径恢复：

```bash
./go2_custom_jetpack.sh restore "$BACKUP_DIR"
```

也可以传入 `backups/` 下的备份目录名：

```bash
./go2_custom_jetpack.sh restore <BACKUP_NAME>
```

恢复期间同样不得断电、拔线或切换 USB 设备。

---

# 第八部分：刷写后的 Unitree 软件环境

RoboLegion 刷写完成后提供的是 JetPack 基础系统。完整 CUDA 开发工具链、PyTorch、ROS、Unitree 应用、SLAM 和导航程序都不应仅凭刷写成功就假定已经安装或恢复。

后续应根据 JetPack 6.2.2 和 Ubuntu 22.04 重新安装或迁移：

```text
ROS 2 Humble
unitree_ros2
CycloneDDS 配置
前置相机
雷达、IMU 和里程计接口
SLAM 与导航程序
用户算法
```

旧 JP5 备份中的以下内容可以用于参考或选择性迁移：

```text
源码
配置文件
标定文件
地图
launch 文件
话题名称
frame 定义
启动逻辑
```

不要直接覆盖 JP6 的：

```text
/boot
/lib/modules
/etc 整体目录
/opt/ros 整体目录
/usr/local 整体目录
旧 CUDA
旧 TensorRT
旧 NVIDIA L4T 库
旧 ROS 运行库
```

基础检查：

```bash
ping -c 3 192.168.123.161
systemctl --failed
```

安装 `unitree_ros2` 后：

```bash
source ~/unitree_ros2/setup_go2.sh
ros2 topic list | sort
```

能够访问 Go2 主控不代表 SLAM 和导航已经恢复。相机、雷达、IMU、里程计、定位和运动控制仍应分别验收。

---

# 附录 A：VMware USB 直通

刷写期间 NVIDIA 设备可能依次表现为：

```text
NVIDIA APX
→ NVIDIA Linux for Tegra / Recovery RNDIS
→ 正常 Jetson USB gadget
```

每次 VMware 弹出 USB 设备选择时，应将设备连接到 Ubuntu 虚拟机。

检查：

```bash
lsusb | grep -iE '0955|nvidia'
```

若看不到 NVIDIA 设备：

1. 确认 Go2 供电；
2. 确认使用刷写 USB-C 接口；
3. 更换支持数据传输的 USB-C 线；
4. 检查 VMware USB 控制器；
5. 将设备从 Windows 重新连接到 Ubuntu 虚拟机；
6. 重新进入 Recovery。

---

# 附录 B：通过 USB-C 为 Orin 共享互联网

正常启动后常见地址：

```text
主机 USB 地址：192.168.55.100
Orin USB 地址：192.168.55.1
```

## 主机侧

自动识别接口：

```bash
UP_IF=$(
  ip route get 1.1.1.1 |
  awk '{
    for (i=1; i<=NF; i++) {
      if ($i=="dev") {
        print $(i+1)
        exit
      }
    }
  }'
)

USB_IF=$(
  ip -o -4 addr show |
  awk '$4=="192.168.55.100/24" {
    print $2
    exit
  }'
)

echo "Internet interface: ${UP_IF:-NOT_FOUND}"
echo "Go2 USB interface:  ${USB_IF:-NOT_FOUND}"
```

确认无误后：

```bash
sudo sysctl -w net.ipv4.ip_forward=1

sudo iptables -t nat -C POSTROUTING \
  -s 192.168.55.0/24 \
  -o "$UP_IF" \
  -j MASQUERADE 2>/dev/null ||
sudo iptables -t nat -A POSTROUTING \
  -s 192.168.55.0/24 \
  -o "$UP_IF" \
  -j MASQUERADE

sudo iptables -C FORWARD \
  -i "$USB_IF" \
  -o "$UP_IF" \
  -j ACCEPT 2>/dev/null ||
sudo iptables -A FORWARD \
  -i "$USB_IF" \
  -o "$UP_IF" \
  -j ACCEPT

sudo iptables -C FORWARD \
  -i "$UP_IF" \
  -o "$USB_IF" \
  -m conntrack \
  --ctstate RELATED,ESTABLISHED \
  -j ACCEPT 2>/dev/null ||
sudo iptables -A FORWARD \
  -i "$UP_IF" \
  -o "$USB_IF" \
  -m conntrack \
  --ctstate RELATED,ESTABLISHED \
  -j ACCEPT
```

## Orin 侧

先检查：

```bash
ip -br addr
ip route
ip route get 1.1.1.1
resolvectl status
```

若已经存在：

```text
default via 192.168.55.100 dev <USB接口>
```

则不要重复修改。

只有缺少 USB 默认路由时，临时执行：

```bash
USB_DEV=$(
  ip -o -4 addr show |
  awk '$4=="192.168.55.1/24" {
    print $2
    exit
  }'
)

sudo ip route replace \
  default via 192.168.55.100 \
  dev "$USB_DEV" \
  metric 600
```

需要时配置 DNS：

```bash
sudo resolvectl dns "$USB_DEV" 1.1.1.1 8.8.8.8
```

验证：

```bash
ping -c 3 1.1.1.1
getent hosts github.com
```

---

# 附录 C：常见问题

## C.1 `status` 看不到 APX

```bash
lsusb
```

若没有 NVIDIA `0955:*`：

- 检查刷写接口；
- 检查 USB-C 数据线；
- 检查 Go2 电源；
- 检查 VMware USB 直通；
- 重新执行 `forced-recovery`；
- 必要时使用硬件 Recovery 方法。

---

## C.2 显示 RNDIS 而不是 APX

说明 Recovery initrd 已经运行。

处理步骤：

1. 确认之前的 backup/flash 已结束；
2. 让设备退出 Recovery initrd；
3. 正常启动或重新上电；
4. 再次进入 `forced-recovery`；
5. 确认 `status` 为 APX。

---

## C.3 备份长时间没有输出

检查：

```bash
ps -eo pid,stat,etime,%cpu,%mem,cmd |
grep -E '[t]ar|[z]std|[g]zip|[l]4t_backup_restore|[n]vbackup'

df -h "$WORK_ROOT"
```

若进程仍存在、镜像仍增长，继续等待。

---

## C.4 SSH 主机密钥变化

```bash
ssh-keygen -R 192.168.123.18
ssh-keygen -R 192.168.55.1
```

---

## C.5 能连接 Go2，但不能访问互联网

检查：

```bash
ip route get 1.1.1.1
getent hosts github.com
```

互联网流量应走 NAT、Wi-Fi 或其他上联网卡，不应走仅连接 Go2 的静态网卡。

---

## C.6 数据盘空间不足

```bash
df -h "$WORK_ROOT"

du -sh "$REPO_DIR"/downloads/* 2>/dev/null
du -sh "$REPO_DIR"/bsp/* 2>/dev/null
du -sh "$REPO_DIR"/backups/* 2>/dev/null
```

不要在 backup 或 flash 运行期间删除当前 BSP、临时镜像或唯一备份。

---

# 最终检查表

## 刷写前

- [ ] x86_64 Ubuntu 主机；
- [ ] 推荐 Ubuntu 22.04；
- [ ] USB-C 数据线连接正确刷写接口；
- [ ] 工作盘空间充足；
- [ ] 主机可访问 GitHub 和 NVIDIA；
- [ ] 已记录仓库 commit；
- [ ] 已记录原系统版本、存储、网络和功耗模式；
- [ ] 文件级迁移备份已校验；
- [ ] 完整分区备份已生成；
- [ ] 根分区、其他分区、GPT/MBR、QSPI 和分区映射均存在；
- [ ] 备份文件均存在且大小非零；
- [ ] 建议已复制第二份备份。

## 刷写时

- [ ] `status` 显示 APX；
- [ ] 目标版本为 JetPack 6.2.2；
- [ ] 使用 `flash all`；
- [ ] 未误加 `--super`；
- [ ] 电源、USB 和 VMware 保持稳定；
- [ ] 最终出现 `[+] flash 'all' complete`。

## 刷写后

- [ ] Ubuntu 22.04；
- [ ] L4T R36.5；
- [ ] 内核 `5.15.185-tegra`；
- [ ] 架构 `aarch64`；
- [ ] NVMe 根分区正常；
- [ ] `192.168.123.18` 可达；
- [ ] `192.168.55.1` 可达；
- [ ] `192.168.123.161` 可达；
- [ ] 已记录 `nvpmodel -q`；
- [ ] 已检查 NVIDIA 计算组件安装状态，未将 `nvcc` 缺失误判为刷写失败；
- [ ] 已修改默认密码；
- [ ] 没有用 JP5 系统目录整体覆盖 JP6；
- [ ] Unitree/ROS 软件栈作为后续独立阶段恢复。

---

# 参考资料

1. [RoboLegion：Unitree Go2 & G1 EDU Jetson — Custom Jetpack](https://robolegion.com/unitree-go2-g1-jetpack/)
2. [RoboLegion `unitree-jetpack` Go2 分支](https://github.com/legion1581/unitree-jetpack/tree/go2)
3. [NVIDIA JetPack 6.2.2（含 CUDA、TensorRT、cuDNN 与 VPI 版本）](https://developer.nvidia.com/embedded/jetpack-sdk-622)
4. [NVIDIA Jetson Linux 36.5](https://developer.nvidia.com/embedded/jetson-linux-r365)

---

## 流程概览

```text
准备 x86_64 Ubuntu 22.04 主机
→ 准备工作盘与网络
→ 下载 go2 分支并记录 commit
→ 检查原 Orin
→ 生成文件级迁移备份
→ 初始化 JetPack 6.2.2 BSP
→ USB-C 进入 APX
→ 完整备份 NVMe、GPT/MBR 与 QSPI
→ 确认备份文件完整存在
→ 退出 Recovery initrd
→ 再次进入 APX
→ flash JetPack 6.2.2 all
→ 验证系统、NVMe 与 Go2 网络
→ 检查 CUDA、TensorRT 等计算组件安装状态
→ 根据 JP6 版本恢复所需 Unitree/ROS 软件
```

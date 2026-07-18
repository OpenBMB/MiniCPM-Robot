<p align="center">
  <img src="assets/minicpm_robot.png" width="400" alt="MiniCPM-Robot" />
</p>

<p align="center">
  <strong>A Smarter and Faster On-Device AI Brain for Robots</strong>
</p>

<p align="center">
  <b>English</b> · <a href="./README_zh.md">中文</a>
</p>

<p align="center">
  <span style="display: inline-flex; align-items: center; margin-right: 2px;">
    <img src="./assets/X_logo.jpg" alt="X" width="15" height="15" style="margin-right: 4px;">
    <a href="assets/x.png" target="_blank"> X</a> &nbsp;|
  </span>
  <span style="display: inline-flex; align-items: center; margin-right: 2px;">
    <img src="./assets/discord_logo.png" alt="Discord" width="15" height="15" style="margin-right: 4px;">
    <a href="assets/discord.jpeg" target="_blank"> Discord</a> &nbsp;|
  </span>
</p>

<p align="center">
  MiniCPM-RobotManip<a href="https://huggingface.co/openbmb/MiniCPM-RobotManip">🤗</a> | MiniCPM-RobotTrack<a href="https://huggingface.co/openbmb/MiniCPM-RobotTrack">🤗</a> | <a href="#quick-start">🍳 Cookbook</a>
</p>

**MiniCPM-Robot** is MiniCPM's embodied intelligence model family for perception, decision-making, and action in the physical world, advancing MiniCPM from multimodal understanding toward real-world interaction. The first models in the family include:

- **MiniCPM-RobotManip**: 🦾 Designed for generalist robot manipulation in simulation and the real world. Built on MiniCPM-V 4.6 as a **1.5B-parameter general-purpose embodied model**, it **uses one set of weights across all downstream tasks** and **outperforms larger models such as π₀.₅ and Qwen-VLA across representative evaluations**. It inherits efficient visual encoding and visual token compression, while streaming inference continuously incorporates historical observations into the model context, **reducing per-step compute from 125 TFLOPs to 3.3 TFLOPs while retaining 60 frames of history** and supporting **up to one minute of visual memory**. This moves VLA beyond reactive action generation from single-frame observations toward continuous decision-making grounded in long-horizon visual context.

- **MiniCPM-RobotTrack**: 🎯 The **first fully on-device embodied target-tracking solution**, covering static-target, dynamic-target, and adversarial-target settings. Built on MiniCPM4-0.5B with **0.9B total parameters**, it improves robustness in long-tail scenarios through a self-evolving data pipeline and DAgger-style closed-loop training, achieving **state-of-the-art performance among open-source models on EVT-Bench**. End-to-end system optimization enables **5+ FPS with approximately 180 ms latency** on the Unitree Go2 EDU's onboard compute, delivering **fully local, vision-only natural-language tracking**.

<p align="center">
  <img src="MiniCPM-RobotManip/assets/manip_case_en.gif" width="800" alt="MiniCPM-RobotManip task demonstrations" />
</p>

## 📰 News

* [2026.07.19] 🔥🔥🔥 We release and open-source MiniCPM-Robot, MiniCPM's first embodied intelligence model family for interaction with the physical world. Its first releases are [MiniCPM-RobotManip](https://huggingface.co/openbmb/MiniCPM-RobotManip) for generalist robot manipulation and [MiniCPM-RobotTrack](https://huggingface.co/openbmb/MiniCPM-RobotTrack) for embodied target tracking. Try it now!

* [2026.07.19] 🚀🚀🚀 [PhyAI](https://github.com/MEmbodied/phyai) adds Day-0 support for MiniCPM-Robot, increasing inference throughput on NVIDIA H20 from 10.12 Hz to 36.77 Hz through CUDA Graph and custom Triton fused kernels.

## Contents

- [MiniCPM-RobotManip](#minicpm-robotmanip)
  - [Benchmark Results](#benchmark-results)
  - [Quick Start](#quick-start)
  - [Inference](#inference)
- [MiniCPM-RobotTrack](#minicpm-robottrack)
  - [Examples](#examples)
  - [EVT-Bench Results](#evt-bench-results)
  - [Quick Start](#quick-start-1)
  - [Data Preparation](#data-preparation)
  - [Finetuning](#finetuning)
  - [Evaluation](#evaluation)
  - [Go2 Deployment](#go2-deployment)
- [Model Zoo](#model-zoo)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## MiniCPM-RobotManip
<strong>MiniCPM-RobotManip</strong> is a 1.5B vision-language-action model for embodied manipulation with the following highlights:
<ul>
  <li><b>Generalist Manipulation:</b> A unified 1.5B generalist policy that covers all downstream tasks with one set of weights.</li>
  <li><b>Streaming Context:</b> Historical observations are incorporated into the model context through streaming inference. With 60 frames of history, traditional recomputation requires 125 TFLOPs per decision step, while streaming inference needs only 3.3 TFLOPs. The model supports up to 1 minute of visual context memory while keeping online cost comparable to traditional single-frame reactive inference.</li>
  <li><b>Efficient Inference:</b> Inherits MiniCPM-V 4.6's visual token compression, reducing each frame from 256 to 64 visual tokens for 4× compression. With H100, BF16, and single-frame input, model-forward latency per decision step is 120 ms, compared with 234 ms for π0.5. The measurement excludes task autoregressive decoding.</li>
</ul>

### Benchmark Results

<table align="center">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Eval Setting</th>
      <th rowspan="2">Open Weights</th>
      <th rowspan="2">Model Size</th>
      <th rowspan="2">LIBERO</th>
      <th rowspan="2">Calvin (ABC→D)</th>
      <th colspan="2">RoboTwin2 (clean+random)</th>
      <th rowspan="2">RMBench</th>
    </tr>
    <tr>
      <th>easy</th><th>hard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>π₀</td><td>Specialist</td><td>✅</td><td>3B</td><td>94.4</td><td>3.9</td><td>65.9</td><td>58.4</td><td>&mdash;</td>
    </tr>
    <tr>
      <td>π₀.₅</td><td>Specialist</td><td>✅</td><td>3B</td><td>96.9</td><td>4.1</td><td>82.7</td><td>76.8</td><td>10.4</td>
    </tr>
    <tr>
      <td>Abot-M0</td><td>Specialist</td><td>✅</td><td>4B+</td><td>98.6</td><td>&mdash;</td><td>86.1</td><td>85.1</td><td>&mdash;</td>
    </tr>
    <tr>
      <td>StarVLA-α</td><td>Generalist</td><td>✅</td><td>4B+</td><td>97.8</td><td>&mdash;</td><td>88.7</td><td>87.8</td><td>&mdash;</td>
    </tr>
    <tr>
      <td>Qwen-VLA</td><td>Generalist</td><td>❌</td><td>5B+</td><td>97.9</td><td>&mdash;</td><td>86.1</td><td>87.2</td><td>&mdash;</td>
    </tr>
    <tr>
      <td>LingBot-VA</td><td>Specialist</td><td>✅</td><td>5B+</td><td>98.5</td><td>&mdash;</td><td>92.9</td><td>91.6</td><td>&mdash;</td>
    </tr>
    <tr>
      <td><b>MiniCPM-RobotManip</b></td><td>Generalist</td><td>✅</td><td><b>1.5B</b></td><td><b>97.5</b></td><td><b>4.1</b></td><td><b>91.3</b></td><td><b>91.6</b></td><td><b>53.3</b></td>
    </tr>
  </tbody>
</table>

### Quick Start

<p>Install and initialize Conda first. The specification follows the tested Python 3.10 and PyTorch 2.6.0 (CUDA 12.4) setup.</p>

<pre><code class="language-bash">cd MiniCPM-RobotManip
conda env create -f environment.yml
conda activate MiniCPM-RobotManip</code></pre>

### Inference

<p>Run single-sample inference with <code>vla_infer.py</code>. Provide at least one image and a language instruction; robot state defaults to zeros if omitted. The model returns an action chunk of shape <code>(30, 80)</code>.</p>

<pre><code class="language-bash">cd MiniCPM-RobotManip
python vla_infer.py \
    --image frame.jpg \
    --text "Pick up the red block." \
    --checkpoint ./checkpoint \
    --state-file state.npy \
    --embodiment-id 0 \
    --output action.npy</code></pre>

<p>Use multiple <code>--image</code> flags for multi-view inputs. Without <code>--output</code>, the predicted action is printed as JSON.</p>

<pre><code class="language-bash">cd MiniCPM-RobotManip
python vla_infer.py \
    --image cam_front.jpg \
    --image cam_wrist.jpg \
    --text "Pick up the red block." \
    --checkpoint ./checkpoint</code></pre>

## MiniCPM-RobotTrack
<strong>MiniCPM-RobotTrack</strong> is a compact vision-language-action policy for embodied target tracking built on MiniCPM4-0.5B with following highlights:

<ul>
  <li><b>Quality-driven self-evolving data pipeline:</b> automated checks and manual review remove abnormal trajectories, incorrect actions, and invalid interactions, while continual model-environment interaction adds high-value training samples.</li>
  <li><b>DAgger for embodied tracking:</b> after learning from large-scale general scenarios, the model interacts with simulators and real robots to expose failures in long-tail cases such as target crossings, rapid turns, short occlusions, and multi-person intersections. Samples corrected by expert policies or rules are aggregated into the next training round to continuously improve tracking and generalization.</li>
  <li><b>End-to-end Go2 optimization:</b> joint optimization across visual capture, input encoding, inference, action generation, command transmission, and execution delivers a stable <b>5+ FPS</b> with approximately <b>180 ms</b> end-to-end latency on the Unitree Go2's native onboard compute.</li>
  <li><b>One-command Go2 Edu deployment and launch:</b> the <a href="MiniCPM-RobotTrack/docs/GO2_DEPLOYMENT.md">local deployment workflow</a> covers environment setup, dependencies, model loading, camera input, text commands, robot control, and a one-command deployment-and-launch workflow, enabling fully local, vision-only natural-language tracking without rebuilding a separate perception, planning, and control stack.</li>
</ul>

### Examples

<table align="center">
  <tr>
    <td align="center" width="33%">
      <img src="MiniCPM-RobotTrack/assets/track1_en.gif" width="100%" alt="Outdoor obstacle-aware target-tracking demo" />
    </td>
    <td align="center" width="33%">
      <img src="MiniCPM-RobotTrack/assets/track2_en.gif" width="100%" alt="Elevator target-tracking demo" />
    </td>
    <td align="center" width="33%">
      <img src="MiniCPM-RobotTrack/assets/track3_en.gif" width="100%" alt="Underground parking target-tracking demo" />
    </td>
  </tr>
  <tr>
    <td align="center"><b>Outdoor Obstacle-aware Tracking</b></td>
    <td align="center"><b>Elevator Tracking</b></td>
    <td align="center"><b>Underground Parking Tracking</b></td>
  </tr>
</table>

### EVT-Bench Results

<p>
  Results are reported as <b>SR / TR / CR</b>: success rate and tracking rate are higher-is-better, while collision rate is lower-is-better. All values are percentages.
</p>

<table align="center">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Open Weights</th>
      <th rowspan="2">Model Size</th>
      <th colspan="3">STT</th>
      <th colspan="3">DT</th>
      <th colspan="3">AT</th>
    </tr>
    <tr>
      <th>SR ↑</th><th>TR ↑</th><th>CR ↓</th>
      <th>SR ↑</th><th>TR ↑</th><th>CR ↓</th>
      <th>SR ↑</th><th>TR ↑</th><th>CR ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TrackVLA</td>
      <td>❌</td>
      <td>7.8B+</td>
      <td>85.1</td><td>78.6</td><td>1.7</td>
      <td>57.6</td><td>63.2</td><td>5.8</td>
      <td>50.2</td><td>63.7</td><td>17.1</td>
    </tr>
    <tr>
      <td>TrackVLA++</td>
      <td>❌</td>
      <td>7.8B+</td>
      <td>86.0</td><td>81.0</td><td>2.1</td>
      <td>66.5</td><td>68.8</td><td>4.7</td>
      <td>51.2</td><td>63.4</td><td>15.9</td>
    </tr>
    <tr>
      <td>Qwen-RobotNav</td>
      <td>❌</td>
      <td>4.4B+</td>
      <td>77.4</td><td>90.0</td><td>6.4</td>
      <td>&mdash;</td><td>&mdash;</td><td>&mdash;</td>
      <td>&mdash;</td><td>&mdash;</td><td>&mdash;</td>
    </tr>
    <tr>
      <td>OmTrackVLA</td>
      <td>✅</td>
      <td>1.0B+</td>
      <td>81.4</td><td>82.8</td><td>5.1</td>
      <td>41.5</td><td>58.8</td><td>11.3</td>
      <td>60.0</td><td>73.9</td><td>7.6</td>
    </tr>
    <tr>
      <td><b>MiniCPM-RobotTrack</b></td>
      <td>✅</td>
      <td><b>0.9B</b></td>
      <td><b>84.1</b></td><td><b>89.8</b></td><td><b>3.0</b></td>
      <td><b>53.2</b></td><td><b>73.4</b></td><td><b>13.6</b></td>
      <td><b>58.0</b></td><td><b>80.4</b></td><td><b>9.0</b></td>
    </tr>
  </tbody>
</table>

<p>
  Among the open-source checkpoints shown above, MiniCPM-RobotTrack obtains the best STT SR/TR/CR and the best DT SR/TR. It also reaches <b>80.35 TR</b> on AT with a 0.5B backbone.
</p>

### Quick Start

#### 1. Create the environment

<p>Install and initialize Conda first. The specification follows the tested Python 3.9, Habitat-Sim 0.3.1, Bullet, PyTorch 2.4.1, and CUDA 12.1 setup.</p>

<pre><code class="language-bash">cd MiniCPM-RobotTrack
conda env create -f environment.yml
conda activate MiniCPM-RobotTrack</code></pre>

#### 2. Prepare simulator data and assets

<p>
  Download HM3D, MP3D, humanoid, and robot assets following the
  <a href="https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md">Habitat-Sim data instructions</a>
  and <a href="https://github.com/wsakobe/TrackVLA">TrackVLA asset instructions</a>.
  Preserve their original directory structure under the project's <code>data/</code> directory.
</p>

<pre><code>data/
├── datasets/
├── scene_datasets/
├── humanoids/
└── robots/</code></pre>

### Data Preparation

#### 1. Unprocessed rollouts

<p>
  Each Habitat rollout retains its video, per-step simulator state, and episode result.
  The runnable <a href="MiniCPM-RobotTrack/sim_data/raw_sample"><code>sim_data/raw_sample</code></a>
  example includes per-step records in the following format:
</p>

<pre><code class="language-json">{
  "base_velocity": [0.62, 0.00, 0.02],
  "collision": false,
  "target_distance": 1.72,
  "human_center_norm": [0.50, 0.48]
}</code></pre>

#### 2. Generate trajectory data

<p>The command below keeps successful episodes, extracts RGB frames, and converts future actions into trajectory JSONL for finetuning:</p>

<pre><code class="language-bash">cd MiniCPM-RobotTrack
python tools/make_tracking_data.py \
  --input_root sim_data/raw_sample \
  --output_root sim_data/train/stt \
  --only_success \
  --history 31 \
  --horizon 8 \
  --dt 0.1 \
  --incremental</code></pre>

<p>For the full dataset, place the raw STT, DT, and AT rollouts under <code>sim_data/raw/&lt;task&gt;</code> and run the same command for each task.</p>

#### 3. Pre-cache visual features

<pre><code class="language-bash">cd MiniCPM-RobotTrack
for task in stt dt at; do
  python tools/precompute_features.py \
    --json "sim_data/train/${task}/jsonl" \
    --data-root "sim_data/train/${task}" \
    --cache-root "sim_data/train/${task}/vision_cache"
done</code></pre>

<p>
  The processed outputs contain <code>frames/</code>, <code>jsonl/</code>, and
  <code>vision_cache/</code>.
  <a href="MiniCPM-RobotTrack/sim_data/sample"><code>sim_data/sample</code></a>
  provides two processed records for each of STT, DT, and AT, showing the final training-data format.
</p>

### Finetuning

<p>After preparing data and visual caches for all three tasks, run the public finetuning entry point:</p>

<pre><code class="language-bash">cd MiniCPM-RobotTrack
bash scripts/train.sh</code></pre>

<p>Data paths, batch size, learning rates, and epoch count can be adjusted in <code>scripts/train.sh</code>.</p>

### Evaluation

<p>
  Download the complete released Hugging Face snapshot to
  <code>minicpm_robot_track/checkpoints/MiniCPM-RobotTrack/</code>. It contains the
  fine-tuned MiniCPM4 backbone and is loaded with Transformers
  <code>from_pretrained</code>. Evaluation still requires DINOv3 ViT-S/16 and
  SigLIP So400m; existing local vision-model directories can be selected with
  environment variables:
</p>

<pre><code class="language-bash">export DINOV3_MODEL_PATH=/path/to/dinov3-vits16
export SIGLIP_MODEL_PATH=/path/to/siglip-so400m-patch14-384</code></pre>

<p>After the environment, models, and simulator assets are ready, run the evaluation script for each task:</p>

<pre><code class="language-bash">cd MiniCPM-RobotTrack
CKPT=minicpm_robot_track/checkpoints/MiniCPM-RobotTrack
bash scripts/eval_stt.sh "$CKPT" results/stt
bash scripts/eval_dt.sh  "$CKPT" results/dt
bash scripts/eval_at.sh  "$CKPT" results/at</code></pre>

<p>Each command runs the evaluation for the corresponding task.</p>

### Go2 Deployment

<p>
  The real-robot workflow targets a Unitree Go2 EDU with a Jetson Orin NX 16GB.
  The validated stack is Jetson Linux R36.5, CUDA 12.6, TensorRT 10.7,
  Python 3.10, ROS 2 Humble, and MAXN mode 0. Deployment uses separate Jetson
  dependencies and defaults to <code>dry-run</code>, which never sends motion
  commands.
</p>

<pre><code>Go2/D435i RGB -> TCP JPEG -> DINO + SigLIP TensorRT
              -> MiniCPM-RobotTrack -> waypoint -> rate-limited control</code></pre>

<p>
  The complete setup uses a Go2 EDU, Orin NX 16GB, and D435i. The built-in Go2
  front camera is the default validated source; each installation must validate
  the D435i RGB path separately. Detailed hardware parameters, network settings,
  asset download instructions, flashing instructions, and live-control procedures are kept
  in the deployment documentation:
</p>

<ul>
  <li><a href="MiniCPM-RobotTrack/docs/GO2_DEPLOYMENT.md">Go2 deployment and reproduction guide</a></li>
  <li><a href="MiniCPM-RobotTrack/docs/ASSETS.md">Model and deployment assets</a></li>
  <li><a href="MiniCPM-RobotTrack/docs/JETPACK6_UPGRADE.md">JetPack 6 upgrade guide</a></li>
</ul>

#### Quick Start

<p>The following assumes JetPack 6 and the carrier-board patch are already installed:</p>

<p>
  Download the checkpoint manually from
  <a href="https://huggingface.co/openbmb/MiniCPM-RobotTrack">openbmb/MiniCPM-RobotTrack</a>.
  Place the complete Hugging Face snapshot, including custom model code,
  tokenizer files, and <code>model.safetensors</code>, in
  <code>MiniCPM-RobotTrack/minicpm_robot_track/checkpoints/MiniCPM-RobotTrack/</code>
  before running preflight.
</p>

<pre><code class="language-bash">cd MiniCPM-RobotTrack

python3 scripts/download_upstream_assets.py

python3 -m pip install --user -r requirements-build.txt
./scripts/export_onnx.sh

sudo nvpmodel -m 0
sudo jetson_clocks
./scripts/build_engines.sh

./scripts/preflight.sh
./go2_runtime.py run</code></pre>

<p>Status and shutdown:</p>

<pre><code class="language-bash">cd MiniCPM-RobotTrack
./go2_runtime.py status
./go2_runtime.py stop-control
./go2_runtime.py stop</code></pre>

> **Safety:** Keep `runtime.mode: dry-run` for the first run. Before enabling
> live control, validate the camera, model, latency, and stop path on a stand
> with an on-site operator and a working remote/App emergency stop. Follow the
> [Go2 deployment guide](MiniCPM-RobotTrack/docs/GO2_DEPLOYMENT.md) for the
> complete live-control procedure and release limits.

## Model Zoo

| Model | Description | Download |
| --- | --- | --- |
| MiniCPM-RobotManip | A 1.5B vision-language-action model for Robot Manipulation | [🤗](https://huggingface.co/openbmb/MiniCPM-RobotManip) |
| MiniCPM-RobotTrack | A 0.9B vision-language-action model for Target Tracking | [🤗](https://huggingface.co/openbmb/MiniCPM-RobotTrack) |

## License

Model weights and code are open-sourced under the [Apache-2.0](./LICENSE) license.
## Acknowledgments

<p>
  MiniCPM-Robot builds on and references MiniCPM,
  <a href="https://github.com/starVLA/starVLA">starVLA</a>,
  <a href="https://github.com/huggingface/lerobot">LeRobot</a>,
  DINOv3, SigLIP, Habitat-Lab, Habitat-Sim, EVT-Bench, and TrackVLA.
  We thank the authors and communities for their open-source contributions.
  Third-party models, simulator code, datasets, and assets retain their own licenses; see
  <a href="MiniCPM-RobotTrack/THIRD_PARTY_NOTICES.md">THIRD_PARTY_NOTICES.md</a>.
</p>

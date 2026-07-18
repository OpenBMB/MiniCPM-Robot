# Third-party components

MiniCPM-RobotTrack integrates with the projects listed below. Their code,
models, datasets, and assets retain their own terms and are not relicensed by
the top-level Apache-2.0 `LICENSE`.

- MiniCPM4-0.5B and SigLIP are provided under Apache-2.0. A copy of the
  license is included at `third_party/APACHE-2.0.txt`.
- DINOv3 is provided by Meta under the DINOv3 License. A copy used by this
  repository is included at `third_party/DINOV3_LICENSE.md`.
- Habitat-Lab and Habitat-Sim are provided by Meta. The vendored Habitat-Lab
  0.3.1 fork is under `third_party/habitat-lab` and retains its upstream
  copyright headers and MIT terms.
- EVT-Bench tracking extensions are provided under `evt_bench`.
- Unitree SDK2 Python is provided by Unitree Robotics under BSD-3-Clause.
- The JetPack 6 carrier-board patch repository has its own LICENSE and NOTICE.

This repository does not vendor third-party model weights, JetPack, TensorRT,
ROS 2, Unitree SDK2, CycloneDDS, HM3D, MP3D, humanoid assets, or robot assets.
Users must obtain those components from their providers and accept the
applicable licenses.

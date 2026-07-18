# Evaluation manifests

This directory contains the STT, AT, and DT validation episode manifests used
by the evaluation scripts. It does not contain scene meshes, humanoid assets,
robot assets, or model checkpoints.

Expected layout after separately licensed assets are installed:

```text
data/
  datasets/track/{STT,AT,DT}/val/val.json.gz
  humanoids/
  robots/
  scene_datasets/hm3d/
  scene_datasets/mp3d/
```

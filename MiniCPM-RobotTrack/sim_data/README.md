# Sample tracking data

`raw_sample/` is one unprocessed simulator rollout. It contains the recorded
video, per-step simulator metadata, and episode status in the layout consumed
by `tools/make_tracking_data.py`.

`sample/` shows the processed fine-tuning format. It contains RGB frames and
two JSONL records for each of STT, AT, and DT. Image paths are relative to the
dataset root. Visual token caches are generated separately and are not
committed.

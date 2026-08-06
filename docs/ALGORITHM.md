# Algorithm contract

This document defines which operations are upstream algorithm and which are
dataset/model boundary adapters.

## Pinned upstream files

| Component | Official file | Blob SHA |
|---|---|---|
| Endo3R inference | `wrld/Endo3R/demo.py` | `b444081d680d198253aefa85ce6f7c88ea49b7f2` |
| VDA sequence evaluation | `DepthAnything/Video-Depth-Anything/benchmark/eval/eval.py` | `411d416aceb3a827331efc5fcdcdba15b3023936` |
| VDA depth metrics | `DepthAnything/Video-Depth-Anything/benchmark/eval/metric.py` | `6cfd4106d1e929fc2f688b80b01ae8769bea7f77` |

These identifiers are also written into `preflight.json` and the final result.

## Unmodified Endo3R execution

`src/endo3reval/endo3r.py` launches the official entry point with its public
arguments:

```text
python -s <Endo3R>/demo.py
  --demo_path <SCARED RGB directory>
  --kf_every 1
  --save_path <sequence output>
  --ckpt_path <endo3r.pth>
  --device cuda:0
  --resolution 320
  --save_result
```

The subprocess working directory is the official Endo3R repository. No source
file is copied, patched, imported under monkeypatching, or overwritten.

## VDA evaluation invariants

For each complete SCARED keyframe sequence:

1. Load GT depth and convert the preprocessed millimetre values to metres.
2. Load the Endo3R Z-depth and resize it to GT shape using OpenCV's default
   linear resize, matching official `get_infer`.
3. Convert Endo3R depth to predicted disparity with `1 / depth` for positive
   pixels. This is the required representation adapter because VDA aligns its
   relative model output against GT disparity.
4. Stack the complete sequence and create the official valid mask:
   `GT > 1e-3` and `GT < dataset_max_depth`.
5. Solve one least-squares scale and shift over every valid pixel in the full
   sequence: `GT_disparity = scale * predicted_disparity + shift`.
6. Clip aligned disparity at `1e-3`, invert it to depth, and clip depth into
   `[1e-3, dataset_max_depth]`.
7. Compute the official `abs_relative_difference`, `rmse_linear`, and
   `delta1_acc` functions.
8. Average per-sequence metric values arithmetically, as official `eval.py`.

The minimum valid depth is frozen at `1e-3`; configuration values that attempt
to change it are rejected. Maximum depth remains dataset configuration, just as
the official evaluator selects it per dataset.

## Out of scope

- Endo3R training, finetuning, checkpoint conversion, or architecture changes.
- Video Depth Anything model inference; only its official evaluation protocol
  is used.
- VDA TAE, which is a separate pose/intrinsics-dependent evaluator and is not
  part of the main three-metric `benchmark/eval/eval.py` path. It cannot be
  computed from RGB/depth-only preprocessed SCARED input without adding another
  data contract.

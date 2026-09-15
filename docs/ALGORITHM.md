# Algorithm contract

This document defines which operations are upstream algorithm and which are
dataset/model boundary adapters.

## Pinned upstream files

| Component | Official file | Blob SHA |
|---|---|---|
| Endo3R inference | `wrld/Endo3R/demo.py` | `b444081d680d198253aefa85ce6f7c88ea49b7f2` |
| VDA sequence evaluation | `DepthAnything/Video-Depth-Anything/benchmark/eval/eval.py` | `411d416aceb3a827331efc5fcdcdba15b3023936` |
| VDA depth metrics | `DepthAnything/Video-Depth-Anything/benchmark/eval/metric.py` | `6cfd4106d1e929fc2f688b80b01ae8769bea7f77` |
| VDA temporal metric | `DepthAnything/Video-Depth-Anything/benchmark/eval/eval_tae.py` | operational reference used by `vggtoda3` |

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

1. Load Endo3R Z-depth at the exact resolution written by native `demo.py`.
   Predictions are never resized, and every frame in a sequence must have the
   same native output shape.
2. Load GT depth, convert millimetres to metres, and resize only GT to the
   native prediction shape with nearest-neighbour interpolation.
3. Clip Endo3R depth at `1e-3` and convert it to predicted disparity with `1 / depth`,
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
8. Read SCARED `KL` and world-to-camera `camera-pose`, convert translation from
   millimetres to metres, invert poses to camera-to-world, and compute VDA TAE
   over adjacent frames in both directions.
9. Preserve VDA's direct projected-depth assignment, zero-valued empty pairs,
   first-frame K in both directions, and fixed `2 * (num_frames - 1)` denominator.
10. Average per-sequence metric values arithmetically, as in `vggtoda3`.

The minimum valid depth is frozen at `1e-3`, SCARED maximum depth at `100.0 m`,
and GT scale at `0.001`; configuration values that attempt to change these
`vggtoda3` contracts are rejected. Evaluation resolution is deliberately the
native Endo3R output resolution and is recorded per sequence.

## Out of scope

- Endo3R training, finetuning, checkpoint conversion, or architecture changes.
- Video Depth Anything model inference; only its official evaluation protocol
  is used.
- Any Endo3R model-code patch, alternate inference implementation, or
  Student-predicted camera use. TAE uses only SCARED dataset cameras.

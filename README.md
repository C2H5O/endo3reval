# endo3reval

在预处理后的 SCARED dataset 8/9 上调用官方 Endo3R 推理，并用官方
[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
深度评估协议计算 AbsRel、线性 RMSE 和 δ1，保存可复现的逐序列与总体 JSON。

## 算法边界

- Endo3R 作为独立的官方 checkout，通过它自己的 `demo.py` 运行；本项目不复制、
  patch 或修改模型代码。
- VDA 评估保持官方 `benchmark/eval/eval.py` 的主流程：对完整序列在视差域做一次
  scale/shift 最小二乘对齐，转回深度后计算三项官方指标。
- 唯一的模型输出适配是把 Endo3R 保存的 Z-depth 转为 reciprocal disparity；
  SCARED 适配只负责目录发现、毫米到米转换、数值帧 ID 配对和尺寸匹配。
- 官方来源及固定 blob SHA 记录在每次输出 JSON 和 [算法说明](docs/ALGORITHM.md) 中。

## 项目结构

```text
endo3reval/
├── configs/                    # 可提交的示例配置；*.local.json 不上传
├── docs/                       # 算法边界和无 SSH 的服务器同步说明
├── scripts/                    # 服务器启动脚本
├── src/endo3reval/
│   ├── data.py                 # SCARED dataset8/9 与数值帧 ID 发现
│   ├── endo3r.py               # 官方 demo.py 子进程、环境/权重预检
│   ├── vda.py                  # 官方 VDA 对齐与指标
│   ├── pipeline.py             # preflight/infer/evaluate/all 编排
│   └── cli.py                  # 命令行入口
└── tests/                      # 不加载模型的适配层回归测试
```

第三方源码、数据、权重、预测和输出均由 `.gitignore` 排除。

## 服务器准备

服务器已有按照 Endo3R README 配好的 conda 环境，Python 为 `3.9.25`。本项目不会
新建或升级该环境，也不要求 sudo。先通过 HTTPS 获取代码：

```bash
git clone https://github.com/C2H5O/endo3reval.git
cd endo3reval
```

如果官方 Endo3R 尚未放在其他位置，可同样使用 HTTPS：

```bash
git clone https://github.com/wrld/Endo3R.git external/Endo3R
```

复制配置并填写真实的 SCARED、Endo3R 和 checkpoint 路径：

```bash
cp configs/scared.example.json configs/scared.local.json
```

默认配置假设：

```text
external/Endo3R/demo.py
external/Endo3R/checkpoints/endo3r.pth
external/Endo3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

`configs/scared.local.json` 被忽略，不会上传服务器路径。若当前 Endo3R checkout
还需要 `raft-things.pth`，把它的相对路径加入 `required_runtime_files` 即可。

## 数据结构

发现器支持 `dataset8`、`dataset_8` 等目录名，以及 `keyframe_0`、`key_frame_0`
等 keyframe 名。示例：

```text
SCARED_ROOT/
├── dataset8/
│   └── keyframe_0/
│       └── data/
│           ├── left_rectified/       # 或 left/
│           │   ├── 000000.png
│           │   └── ...
│           └── depthmap_rectified/
│               ├── 000000.npy
│               └── ...
└── dataset9/
```

默认按 `left_rectified -> left` 选择 RGB，GT 按 `0.001` 从毫米转换到米。没有 RGB
或 GT 的 keyframe 会记录为 skipped，不会阻断其他可用序列。

## 运行

先激活已配好的官方环境：

```bash
conda activate endo3r
python --version  # 必须是 3.9.25
```

预检会在加载大模型前验证 Python、CUDA、NumPy/OpenCV resize ABI、官方入口和每个
PyTorch checkpoint 的 ZIP CRC：

```bash
bash scripts/run_scared.sh --stage preflight
```

单序列 smoke test：

```bash
bash scripts/run_scared.sh --stage all --limit-sequences 1
```

完整运行 dataset8/9：

```bash
bash scripts/run_scared.sh --stage all
```

也可拆分推理和评估：

```bash
bash scripts/run_scared.sh --stage infer
bash scripts/run_scared.sh --stage evaluate
```

已有完整预测会自动复用。显式重跑时使用：

```bash
bash scripts/run_scared.sh --stage all --force-inference
```

## 输出

默认保存到 `outputs/scared_endo3r_vda/`：

```text
predictions/                 # Endo3R 官方 depth/*.npy
logs/                        # 每个序列的完整 stdout/stderr
.runtime/preflight.json      # Python/CUDA/权重校验信息
run_manifest.json            # 可恢复进度与失败原因
evaluation_vda.json          # 逐序列及总体 VDA 指标
```

总体指标与官方 `eval.py` 一致，使用各序列指标的算术平均。

## 本地开发验证

不需要下载模型即可测试数据适配、命令编排和 VDA 数值路径：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest
```

服务器无需 GitHub CLI。后续同步只需 HTTPS：

```bash
git pull --ff-only origin main
```

详细说明见 [服务器同步](docs/SERVER_SYNC.md)。

## License

本适配项目使用 Apache-2.0。Endo3R、Video Depth Anything、模型权重和数据集仍受各自
上游许可约束；详见 [NOTICE](NOTICE)。

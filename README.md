<h2 align="center">
  <b>VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation</b>
  <br>
  <b><i>CVPR 2025</i></b>

<div align="center">
    <a href="https://arxiv.org/abs/2503.07135" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://hanzhic.github.io/vidbot-project/" target="_blank">
    <img src="https://img.shields.io/badge/Page-VidBot-blue" alt="Project Page"/></a>
    <a href="https://www.youtube.com/watch?v=lfI6M1perfQ" target="_blank">
    <img src="https://img.shields.io/badge/Video-YouTube-red"></a>
</div>
</h2>

<div align="center">
    <img src="assets/teaser.png" alt="VidBot Teaser" width="1000">
</div>

This is the official repository of [**VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation**](https://arxiv.org/abs/2503.07135). For more details, please check our [**project website**](https://hanzhic.github.io/vidbot-project/).



## Installation and Setup

The repository now targets modern PyTorch/Lightning and Python 3.12+.

### 0) System prerequisites

Install these command-line tools first:

- `git`
- `wget`
- `unzip`

On macOS, you can install missing tools with Homebrew, for example:

```bash
brew install wget
```

### 1) Clone the repository

```bash
git clone https://github.com/HanzhiC/vidbot.git
cd vidbot
```

### 2) Create environment and install Python dependencies

We recommend `uv` (uses `pyproject.toml` + `uv.lock` in this repo).

```bash
# Create local venv (Python 3.12 recommended)
uv venv --python 3.12

# Install project dependencies
uv sync
```

All commands below use `uv run ...` so they run inside `.venv` without manual activation.
If you are using an activated venv instead, just drop the `uv run` prefix.

If you prefer standard venv + pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 3) Download pretrained checkpoints and demo data

```bash
uv run sh scripts/download_ckpt_testdata.sh
```

This downloads:

- `datasets/vidbot_data_demo`
- `datasets/epickitchens_traj_demo`
- `pretrained/*` model checkpoints

### 4) Quick demo (no third-party modules required)

Run end-to-end inference on demo frames using pre-saved detections:

```bash
uv run bash scripts/test_demo.sh
```

This is the fastest way to verify your setup.

On macOS, `scripts/test_demo.sh` uses `shuf`.
If you do not have it, install GNU coreutils:

```bash
brew install coreutils
```

Then either:

- run with `gshuf` available in your PATH, or
- run the demo commands manually (copy from `scripts/test_demo.sh`).

### 5) Optional: install third-party modules (detector, segmentation, depth, grasp)

If you want open-vocabulary detection, segmentation, external depth models, and optional GraspNet:

```bash
uv run sh scripts/prepare_third_party_modules.sh
```

This script initializes required submodules and installs what it can automatically.

Important: third-party modules are installed as extra editable packages.
If you run `uv sync` again later, these extras may be removed; just re-run the setup script.


## Full Pipeline (Your Own Data)

Use this section to run VidBot on your own RGB-D or RGB data.

### Step 1) Prepare dataset folder

Place your dataset under `datasets/` with this structure:

```text
YOUR_DATASET_NAME/
├── camera_intrinsic.json
├── color/
│   ├── 000000.png  (or .jpg)
│   ├── 000001.png
│   └── ...
└── depth/          (optional if you will estimate depth)
    ├── 000000.png
    ├── 000001.png
    └── ...
```

`camera_intrinsic.json` format:

```json
{
  "width": 1280,
  "height": 720,
  "intrinsic_matrix": [
    fx, 0, 0,
    0, fy, 0,
    cx, cy, 1
  ]
}
```

Notes:

- Depth PNGs are expected in **millimetres** (`uint16`), same convention as demo data.
- `1280x720` is recommended.

### Step 2) (Optional) Estimate camera intrinsics

If you do not already have intrinsics, calibrate from checkerboard images:

```bash
uv run python scripts/estimate_instrinsics.py \
  --src calib_frames \
  --board_size 9 6 \
  --square_size 25 \
  --pixel_pitch 1.4 \
  --output calibration.json
```

Then convert/export the result into `camera_intrinsic.json` format above.

This matrix needs to be transposed before being linearized for the vidbot format.

### Step 3) (Optional) Estimate depth from RGB frames

If your dataset does not contain `depth/*.png`, estimate depth first.

Depth Anything V3 (recommended default):

```bash
uv run python scripts/estimate_depth.py \
  --dataset YOUR_DATASET_NAME \
  --model dav3
```

Metric3D:

```bash
uv run python scripts/estimate_depth.py \
  --dataset YOUR_DATASET_NAME \
  --model metric3d
```

Output folders:

- `depth_dav3/` for DAv3
- `depth_m3d/` for Metric3D

### Step 4) Run affordance inference

#### A) Full pipeline with detector + segmentation (requires Installation Step 5 above)

```bash
uv run python demos/infer_affordance.py \
  --config ./config/test_config.yaml \
  --dataset YOUR_DATASET_NAME \
  --frame 0 \
  --instruction "open cabinet" \
  --object cabinet \
  --depth_model auto \
  --visualize
```

#### B) Run from cached detections/results (no detector needed)

```bash
uv run python demos/infer_affordance.py \
  --dataset YOUR_DATASET_NAME \
  --frame 0 \
  --instruction "open cabinet" \
  --load_results \
  --depth_model auto \
  --visualize
```

Optional flags:

- `--use_graspnet` to use learned grasp detection (if installed)
- `--skip_coarse_stage` and/or `--skip_fine_stage` for ablations
- `--no_save` to disable writing output files

### Step 5) Inspect and visualize outputs

Generated artifacts:

- `datasets/YOUR_DATASET_NAME/scene_meta/*.npz`
- `datasets/YOUR_DATASET_NAME/prediction/*.npz`

Visualization helpers:

```bash
# Pipeline sanity view (RGB, depth, point cloud, intrinsics overlay)
uv run python scripts/visualize_pipeline.py --dataset YOUR_DATASET_NAME --frames 0

# Prediction summary figures from saved NPZ results
uv run python scripts/visualize_results.py --dataset YOUR_DATASET_NAME --frame 0
```


## Citation

**If you find our work useful, please cite:** 

```bibtex
@article{chen2025vidbot,
    author    = {Chen, Hanzhi and Sun, Boyang and Zhang, Anran and Pollefeys, Marc and Leutenegger, Stefan},
    title     = {{VidBot}: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference},
    year      = {2025},
}
```

## Acknowledgement
Our codebase is built upon [TRACE](https://github.com/nv-tlabs/trace). Partial code is borrowed from [ConvONet](https://github.com/autonomousvision/convolutional_occupancy_networks), [afford-motion](https://github.com/afford-motion/afford-motion) and [rq-vae-transformer
](https://github.com/kakaobrain/rq-vae-transformer). Thanks for their great contribution!

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

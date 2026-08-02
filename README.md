# Evaluating the Segment Anything Model (SAM) on Aerial Imagery

CS-7900 (AI in Graphics and Vision) - Wright State University
Michael Sloan | sloan.40@wright.edu

Evaluates Meta AI's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) zero-shot segmentation performance on aerial imagery using the [DOTA](https://captain-whu.github.io/DOTA/) / [iSAID](https://captain-whu.github.io/iSAID/) validation set (458 images, 116,649 ground-truth objects). This repository contains the code referenced in the accompanying project paper; see the paper for full methodology, results, and analysis.

## Repository Structure

```
.
├── sam_miou.py                       # Evaluation pipeline: runs SAM automatic mask generation across the validation set and computes mIoU against iSAID ground truth
├── generate_visualizations.py        # Generates the mask-overlay figures used in the paper/presentation for a single given image
├── parse_results.py                  # Parses miou_results.json and prints summary statistics
├── sam_validation_test.ipynb         # Exploratory notebook validating SAM's prompted and automatic mask generation on a sample image
├── miou_results.json                 # Raw output of sam_miou.py — per-image and per-category results for the full 458-image run
├── figures/                          # P2085.png (best) and P1557.png (worst) with SAM's automatic masks overlaid
├── requirements.txt                  # Python dependencies
└── README.md
```

## Setup

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Developed and ran locally on an Apple M1 Max MacBook Pro using PyTorch's MPS backend (`device = "mps"` in `sam_miou.py` and `generate_visualizations.py`). On a CUDA-capable machine, change `device` to `"cuda"` in both files.

**NOTE:** SAM has a bug with their `mps` support. In order for their source to run on Mac you must open this file in your `.venv`: `.venv/src/segment-anything/segment_anything/automatic_mask_generator.py`
1. Go to line 277: `in_points = torch.as_tensor(transformed_points, device=self.predictor.device)`
2. Change line 277 to this: `in_points = torch.as_tensor(transformed_points.astype(np.float32), device=self.predictor.device)`

### 2. SAM Model Checkpoint

Download the ViT-H checkpoint from Meta's [Segment Anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints):

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Place it at `../segment-anything/sam_vit_h_4b8939.pth` relative to this repo, matching the path expected by `sam_miou.py` and `generate_visualizations.py`.

### 3. Dataset

Download the DOTA validation images and iSAID validation annotations:

- [DOTA dataset](https://captain-whu.github.io/DOTA/dataset.html)
- [iSAID dataset](https://captain-whu.github.io/iSAID/dataset.html)

The dataset is **not included in this repository** (size and licensing). Place the downloaded files at:

```
../validation_dataset/images_v1/                     # DOTA validation images (.png)
../validation_dataset/Annotations/iSAID_val.json     # iSAID validation annotations
```

relative to this repo, matching the hardcoded paths in `sam_miou.py`.

## Usage

**Run the full evaluation:**
```bash
python sam_miou.py
```
Runs SAM automatic mask generation across the validation set and writes results to `miou_results.json`. Took ~23 hours locally on an M1 Max.

**Generate a visualization for a specific image (expected image directory is ../validation_dataset/images_v1/):**
```bash
python generate_visualizations.py <filename> [output_path]
# e.g. python generate_visualizations.py P1557.png figures/P1557.png
```

**View summary statistics:**
```bash
python parse_results.py [path_to_miou_results.json]
```
Prints the overall mIoU, per-category mIoU, and best/worst-performing images — the numbers reported in the paper.

## Notes

- Mask colors in `generate_visualizations.py` are randomly assigned per run for visual distinction only (no semantic meaning).
- SAM inference on MPS may have minor run-to-run nondeterminism; aggregate mIoU should be consistent, but exact mask boundaries may vary slightly.
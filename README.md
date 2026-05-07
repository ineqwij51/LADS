This folder contains the code kept for public release of the LADS paper:

`Dyadic Imitation Modeling with Lag-Aware Dual-Stream Framework for Autism Classification`

## Contents

| File | Purpose |
| --- | --- |
| `Diffusion_new.py` | Main LADS training/evaluation script used by the final main and ablation runs. |
| `Diffusion_new_patched.py` | Later patched LADS script with DTW/no-LAA visualization and regularization-related utilities. |
| `Diffusion_3.py` | Dataset, split, and helper code imported by the LADS scripts. |
| `classwise_viz.py` | Class-wise alignment/lag visualization helper imported by LADS scripts. |
| `Baselines.py` | Baseline model training script for VGG-style CNN, ResNet, PatchTST, BlockGCN-style input, DIM-style model, etc. |
| `run_ablation.py` | Ablation runner used to launch the paper ablation variants. |
| `run_regularization_comparison.py` | Regularization comparison helper. |
| `splits_subjects_template.json` | Subject-independent 5-fold split template. |
| `setup_baseline_env.sh` | Baseline environment setup helper. |

## Data

The original local dataset was moved outside the release code folder:

```text
../results/data/autism_multimodal_dataset_20250726.pkl
```

Put the dataset at a local path and pass it with `--data_path`.

Expected sample feature keys follow the paper format:

```text
exp/skeleton:    (T, 132)
exp/sparse_flow: (T-1, 66)
exp/dense_flow:  (T-1, 128)
exp/heatmap:     (T, 833)
sub/skeleton:    (T, 132)
sub/sparse_flow: (T-1, 66)
sub/dense_flow:  (T-1, 128)
sub/heatmap:     (T, 833)
label
subject_id
```

## Environment

Core dependencies used by the scripts:

```text
python >= 3.8
torch
numpy
matplotlib
scikit-learn
tqdm
seaborn
einops
pandas
```

Optional baseline dependencies may include `tsai` for PatchTST.

## Reproduce LADS Main Results

Run from this directory:

```bash
python Diffusion_new.py \
  --data_path ../results/data/autism_multimodal_dataset_20250726.pkl \
  --features skeleton,dense_flow,heatmap,sparse_flow \
  --lag_windows 12 \
  --use_diff 1 \
  --epochs 60 \
  --batch_size 8 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --dropout 0.1 \
  --seed 42 \
  --use_d3_splits 1 \
  --splits_json splits_subjects_template.json \
  --root_out ./runs_lads_main
```

## Reproduce Ablation Results
Example runner command:

```bash
python run_ablation.py \
  --main_cmd "python Diffusion_new.py --root_out {out} --features {feature} --seed {seed} --splits_json splits_subjects_template.json {extra}" \
  --out_root ./ablation_exps \
  --features skeleton \
  --seeds 42 \
  --sets_csv "baseline,-diff,-laca,pool_avg"
```

## Regularization Comparison

The regularization helper is:

```bash
python run_regularization_comparison.py
```

## Baselines

`Baselines.py` uses an internal `args` dictionary near the bottom of the file. Set `feature_type`, `model`, `save_dir`, and `data_path` there, then run:

```bash
python Baselines.py
```

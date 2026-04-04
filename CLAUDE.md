# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python **3.9 only** (`requires-python = ">=3.9,<3.10"`). The venv is managed with `uv`.

```bash
uv sync                  # install / sync dependencies
source .venv/bin/activate
```

## Commands

```bash
# Full training run (fold1 → train, fold2 → val, fold3 → test/held out)
python main.py train

# Quick smoke test (caps each fold at N samples, skips disk cache)
python main.py train --max-samples 20

# Train without class-aware head
python main.py train --no-classes

# Inference on a single image
python main.py predict path/to/image.png
python main.py predict path/to/image.png -o output.tiff

# Installed entry point (same as python main.py)
stardist-skripsi train
stardist-skripsi predict path/to/image.png
```

There is no test suite.

## Architecture

Four files, no packages:

| File | Role |
|------|------|
| `main.py` | argparse CLI router — delegates to `train.train()` or `predict.predict()` |
| `data.py` | PanNuke loading, mask processing, disk cache |
| `train.py` | StarDist model config, training loop |
| `predict.py` | Single-image inference and result summarisation |

### Data flow

**PanNuke** (HuggingFace `RationAI/PanNuke`) has three folds. `load_fold_cached` in `data.py`:

1. **First run** — streams fold from HuggingFace one sample at a time (`streaming=True`), converts PIL instance masks → stacked int32 label map, normalises the RGB image, writes processed arrays to `data/<fold_name>/X.npy`, `Y.npy`, `C.json`.
2. **Subsequent runs** — loads `.npy` files with `mmap_mode='r'` (lazy, pages from disk on demand) and splits into a Python list of per-image views. This is how lazy loading is achieved without a PyTorch DataLoader — StarDist's internal sampler indexes these views and the OS only keeps accessed pages in RAM.

`--max-samples` bypasses the cache entirely (partial data is never written to disk).

### StarDist class-aware training

StarDist's class-aware mode takes a **dict per image** `{instance_id: class_id}` (1-indexed, background = 0), not a per-pixel map. `build_class_dict` in `data.py` produces this. `build_class_map` (per-pixel) exists but is not used in the training path.

`train.py` passes class data via `model.train(..., classes=C_tr)` and `validation_data=(X_val, Y_val, C_val)`.

### Key details

- `np.float = float` at the top of `train.py` is a required compatibility shim for older TensorFlow/csbdeep that reference the removed `np.float` alias.
- Trained model is saved to `models/pannuke_stardist/` (StarDist HDF5 + JSON config).
- `build_instance_map` always receives `size=(H, W)` from the actual image — never rely on the default `(256, 256)` since it will raise a confusing numpy shape error if a mask differs.
- Cache validity is checked by the presence of `X.npy` + `Y.npy`. If `use_classes=True` but `C.json` is absent (cache was built with `--no-classes`), `load_fold_cached` re-downloads automatically.
- `train_n_val_patches` must be set explicitly (256). The default `None` makes StarDist pre-generate one patch per validation image as a single batch before epoch 1 — for 2500 val images this allocates ~12 GB at once, causing OOM on 32 GB machines.
- `train_sample_cache=False` is required for datasets this large. The default `True` caches valid patch-centre indices for all training images; StarDist's own docs say to disable it for large datasets.

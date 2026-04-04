"""Train a StarDist 2D model on the PanNuke dataset."""

from __future__ import annotations

import argparse

import numpy as np
from stardist import fill_label_holes
from stardist.models import Config2D, StarDist2D

np.float = float  # NOQA

from data import CELL_TYPES, load_fold_cached

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_RAYS = 32
USE_CLASSES = True
N_CLASSES = len(CELL_TYPES)  # 5
MODEL_NAME = "pannuke_stardist"
MODEL_BASEDIR = "models"
TRAIN_EPOCHS = 200
STEPS_PER_EPOCH = 100
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
USE_GPU = True  # set True when CUDA is available


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fill_holes(Y: list) -> list:
    return [fill_label_holes(y) for y in Y]


# ── Model builder ─────────────────────────────────────────────────────────────


def build_model(use_classes: bool = USE_CLASSES) -> StarDist2D:
    conf = Config2D(
        n_rays=N_RAYS,
        n_channel_in=3,  # RGB input
        grid=(2, 2),  # output stride
        n_classes=N_CLASSES if use_classes else None,
        use_gpu=USE_GPU,
        train_epochs=TRAIN_EPOCHS,
        train_steps_per_epoch=STEPS_PER_EPOCH,
        train_batch_size=BATCH_SIZE,
        train_learning_rate=LEARNING_RATE,
        # Default (None) uses one patch per val image — for 2500 val images this
        # pre-generates a ~12 GB batch before the first epoch, causing OOM.
        train_n_val_patches=256,
        # Disable patch-centre index cache; StarDist docs recommend False for
        # large datasets (5000 train images × cached indices ≈ ~1 GB over time).
        train_sample_cache=False,
    )
    return StarDist2D(conf, name=MODEL_NAME, basedir=MODEL_BASEDIR)


# ── Training entry point ──────────────────────────────────────────────────────


def train(
    use_classes: bool = USE_CLASSES,
    max_samples: int | None = None,
) -> None:
    # ── Load data ──
    # fold1 = train, fold2 = validation, fold3 = test (held out)
    print("Loading fold1 (train)...")
    X_tr, Y_tr, C_tr = load_fold_cached("fold1", use_classes=use_classes, max_samples=max_samples)

    print("Loading fold2 (validation)...")
    X_val, Y_val, C_val = load_fold_cached(
        "fold2", use_classes=use_classes, max_samples=max_samples
    )

    Y_tr = _fill_holes(Y_tr)
    Y_val = _fill_holes(Y_val)

    if not use_classes:
        C_tr = None

    # ── Build model ──
    model = build_model(use_classes)
    print(
        f"\nModel '{MODEL_NAME}'  "
        f"rays={N_RAYS}  "
        f"classes={model.config.n_classes}  "
        f"epochs={TRAIN_EPOCHS}\n"
    )

    # ── Train ──
    if use_classes:
        # Pass class maps via `classes` kwarg; validation data as 3-tuple
        model.train(
            X_tr,
            Y_tr,
            validation_data=(X_val, Y_val, C_val),
            classes=C_tr,
        )
    else:
        model.train(
            X_tr,
            Y_tr,
            validation_data=(X_val, Y_val),
        )

    # ── Post-training threshold optimization ──
    print("\nOptimizing NMS thresholds on validation set...")
    model.optimize_thresholds(X_val, Y_val)

    print(f"\nDone. Model saved to {MODEL_BASEDIR}/{MODEL_NAME}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StarDist on PanNuke")
    parser.add_argument(
        "--no-classes",
        action="store_true",
        help="Disable class-aware training (train instance segmentation only)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit samples per fold for quick smoke tests (e.g. --max-samples 50)",
    )
    args = parser.parse_args()
    train(use_classes=not args.no_classes, max_samples=args.max_samples)

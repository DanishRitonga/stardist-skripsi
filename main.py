"""CLI entry point for stardist-skripsi."""
from __future__ import annotations

import argparse
import sys

from train import MODEL_BASEDIR, MODEL_NAME


def cmd_train(args: argparse.Namespace) -> None:
    from train import train
    train(use_classes=not args.no_classes, max_samples=args.max_samples)


def cmd_predict(args: argparse.Namespace) -> None:
    from predict import load_model, predict, summarize
    import numpy as np

    model = load_model(name=args.model_name, basedir=args.model_basedir)
    labels, details = predict(
        args.image,
        model=model,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
    )

    summarize(labels, details)

    if args.output:
        from tifffile import imwrite
        imwrite(args.output, labels.astype(np.uint16))
        print(f"Saved → {args.output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluate import evaluate, print_results

    results = evaluate(
        use_classes=not args.no_classes,
        max_samples=args.max_samples,
        iou_threshold=args.iou_threshold,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        model_name=args.model_name,
        model_basedir=args.model_basedir,
        benchmark=not args.no_benchmark,
    )
    print_results(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stardist-skripsi",
        description="StarDist nucleus segmentation on PanNuke",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    t = sub.add_parser("train", help="Train StarDist on PanNuke (fold1+fold2 → fold3 val)")
    t.add_argument(
        "--no-classes",
        action="store_true",
        help="Disable class-aware training",
    )
    def _positive_int(v: str) -> int:
        n = int(v)
        if n < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
        return n

    t.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Cap samples per fold for quick tests (e.g. --max-samples 50)",
    )

    # ── predict ──
    p = sub.add_parser("predict", help="Run inference on a single image")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--output", "-o", metavar="FILE", help="Save instance map as TIFF")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--model-basedir", default=MODEL_BASEDIR)
    p.add_argument("--prob-thresh", type=float, default=None)
    p.add_argument("--nms-thresh", type=float, default=None)

    # ── evaluate ──
    e = sub.add_parser("evaluate", help="Evaluate model on PanNuke fold3 (LSP-DETR protocol)")
    e.add_argument("--no-classes", action="store_true", help="Skip per-class mPQ")
    e.add_argument("--max-samples", type=_positive_int, default=None, metavar="N")
    e.add_argument("--iou-threshold", type=float, default=0.5)
    e.add_argument("--model-name", default=MODEL_NAME)
    e.add_argument("--model-basedir", default=MODEL_BASEDIR)
    e.add_argument("--prob-thresh", type=float, default=None)
    e.add_argument("--nms-thresh", type=float, default=None)
    e.add_argument("--no-benchmark", action="store_true", help="Skip inference timing")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()

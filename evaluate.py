"""Evaluate a trained StarDist model on PanNuke fold3 using mPQ and bPQ."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from data import CELL_TYPES, load_fold_cached
from predict import load_model
from train import MODEL_BASEDIR, MODEL_NAME


def compute_iou_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    n_true = int(y_true.max())
    n_pred = int(y_pred.max())

    if n_true == 0 or n_pred == 0:
        return np.zeros((n_true, n_pred), dtype=np.float64)

    y_true_flat = y_true.ravel().astype(np.int64)
    y_pred_flat = y_pred.ravel().astype(np.int64)

    combined = y_true_flat * (n_pred + 1) + y_pred_flat
    counts = np.bincount(combined, minlength=(n_true + 1) * (n_pred + 1))
    contingency = counts.reshape(n_true + 1, n_pred + 1)[1:, 1:]

    area_true = contingency.sum(axis=1)
    area_pred = contingency.sum(axis=0)
    union = area_true[:, None] + area_pred[None, :] - contingency

    return np.where(union > 0, contingency / union, 0.0)


def match_instances(
    iou_matrix: np.ndarray, iou_threshold: float = 0.5
) -> tuple[list[tuple[int, int, float]], int, int]:
    n_true, n_pred = iou_matrix.shape
    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for idx in np.argsort(-iou_matrix.ravel()):
        i, j = divmod(int(idx), n_pred)
        iou = iou_matrix[i, j]
        if iou < iou_threshold:
            break
        if i not in matched_true and j not in matched_pred:
            matches.append((i, j, float(iou)))
            matched_true.add(i)
            matched_pred.add(j)

    n_fp = n_pred - len(matched_pred)
    n_fn = n_true - len(matched_true)
    return matches, n_fp, n_fn


def _filter_instance_map(
    inst_map: np.ndarray, keep_ids: list[int]
) -> np.ndarray:
    filtered = np.zeros_like(inst_map)
    for new_id, old_id in enumerate(keep_ids, start=1):
        filtered[inst_map == old_id] = new_id
    return filtered


def _pq_from_accum(tp: int, fp: int, fn: int, iou_sum: float) -> float:
    denom = tp + 0.5 * fp + 0.5 * fn
    return iou_sum / denom if denom > 0 else 0.0


def evaluate(
    use_classes: bool = True,
    max_samples: int | None = None,
    iou_threshold: float = 0.5,
    prob_thresh: float | None = None,
    nms_thresh: float | None = None,
    model_name: str = MODEL_NAME,
    model_basedir: str = MODEL_BASEDIR,
) -> dict:
    print("Loading fold3 (test)...")
    X_test, Y_test, C_test = load_fold_cached(
        "fold3", use_classes=use_classes, max_samples=max_samples
    )

    model = load_model(name=model_name, basedir=model_basedir)

    bpq_tp = 0
    bpq_fp = 0
    bpq_fn = 0
    bpq_iou_sum = 0.0

    n_classes = len(CELL_TYPES)
    cls_tp = [0] * n_classes
    cls_fp = [0] * n_classes
    cls_fn = [0] * n_classes
    cls_iou_sum = [0.0] * n_classes

    for idx in tqdm(range(len(X_test)), desc="Evaluating", unit="img"):
        img = X_test[idx]
        y_true = Y_test[idx]
        class_dict = C_test[idx] if C_test is not None else None

        kwargs: dict = {}
        if prob_thresh is not None:
            kwargs["prob_thresh"] = prob_thresh
        if nms_thresh is not None:
            kwargs["nms_thresh"] = nms_thresh

        y_pred, details = model.predict_instances(img, **kwargs)

        iou_mat = compute_iou_matrix(y_true, y_pred)
        matches, n_fp, n_fn = match_instances(iou_mat, iou_threshold)
        bpq_tp += len(matches)
        bpq_fp += n_fp
        bpq_fn += n_fn
        bpq_iou_sum += sum(iou for _, _, iou in matches)

        if use_classes and class_dict is not None and "class_id" in details:
            pred_class_ids = details["class_id"]
            n_pred_instances = int(y_pred.max())

            for c in range(n_classes):
                gt_ids_c = [
                    inst_id
                    for inst_id, cls in class_dict.items()
                    if cls - 1 == c
                ]
                pred_ids_c = [
                    j + 1
                    for j in range(n_pred_instances)
                    if int(pred_class_ids[j]) - 1 == c
                ]

                if not gt_ids_c and not pred_ids_c:
                    continue

                y_true_c = _filter_instance_map(y_true, gt_ids_c)
                y_pred_c = _filter_instance_map(y_pred, pred_ids_c)

                iou_c = compute_iou_matrix(y_true_c, y_pred_c)
                matches_c, fp_c, fn_c = match_instances(iou_c, iou_threshold)

                cls_tp[c] += len(matches_c)
                cls_fp[c] += fp_c
                cls_fn[c] += fn_c
                cls_iou_sum[c] += sum(iou for _, _, iou in matches_c)

    bpq = _pq_from_accum(bpq_tp, bpq_fp, bpq_fn, bpq_iou_sum)
    results: dict = {"bPQ": bpq}

    if use_classes:
        per_class = {}
        for c in range(n_classes):
            pq_c = _pq_from_accum(cls_tp[c], cls_fp[c], cls_fn[c], cls_iou_sum[c])
            per_class[CELL_TYPES[c]] = pq_c

        active = [v for v in per_class.values() if v > 0 or (cls_tp[c] + cls_fp[c] + cls_fn[c]) > 0]
        mpq = float(np.mean(list(per_class.values()))) if per_class else 0.0
        results["mPQ"] = mpq
        results["per_class"] = per_class

    return results


def print_results(results: dict) -> None:
    print(f"\nbPQ: {results['bPQ']:.4f}")
    if "mPQ" in results:
        print(f"mPQ: {results['mPQ']:.4f}")
        print("\nPer-class PQ:")
        for name, pq in results["per_class"].items():
            print(f"  {name}: {pq:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate StarDist on PanNuke fold3")
    parser.add_argument("--no-classes", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, metavar="N")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-basedir", default=MODEL_BASEDIR)
    parser.add_argument("--prob-thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", type=float, default=None)
    args = parser.parse_args()

    results = evaluate(
        use_classes=not args.no_classes,
        max_samples=args.max_samples,
        iou_threshold=args.iou_threshold,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        model_name=args.model_name,
        model_basedir=args.model_basedir,
    )
    print_results(results)

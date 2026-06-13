"""Evaluate a trained StarDist model on PanNuke fold3 — LSP-DETR aligned metrics.

Computes metrics matching RayCastED / LSP-DETR's evaluation protocol:
  AJI, AP@0.5, AP@0.7, AP@0.9, AP@0.5:0.05:0.95,
  bPQ, bMPQ, mPQ, mMPQ,
  F1 (centroid, r=12), Precision, Recall,
  Inference Time.

All instance-matching metrics use Hungarian assignment (not greedy)
to match RayCastED exactly.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import center_of_mass
from tqdm import tqdm

from data import CELL_TYPES, TISSUE_TYPES, load_fold_cached
from predict import load_model
from train import MODEL_BASEDIR, MODEL_NAME

# PanNuke cell-type display names (matches RayCastED naming)
_NUCLEI_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Necrosis", "Epithelial"]


# ---------------------------------------------------------------------------
# Mask utilities (ported from RayCastED metrics.py)
# ---------------------------------------------------------------------------


def label_map_to_masks(label_map: np.ndarray) -> list[np.ndarray]:
    """Split an integer label map into a list of binary masks."""
    n = int(label_map.max())
    if n == 0:
        return []
    H, W = label_map.shape
    masks = []
    for i in range(1, n + 1):
        m = (label_map == i).astype(np.uint8)
        if m.sum() > 0:
            masks.append(m)
    return masks


def label_map_to_centroids(label_map: np.ndarray) -> np.ndarray:
    """Compute centroids (row, col) for each instance in a label map.

    Returns shape (N, 2) in (y, x) order.
    """
    n = int(label_map.max())
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    labels = range(1, n + 1)
    coms = center_of_mass(label_map, label_map, index=labels)
    return np.array(coms, dtype=np.float64)  # (N, 2) — (row, col) = (y, x)


def resolve_mask_overlaps(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Resolve overlapping masks using largest-first priority.

    Processes masks from largest area to smallest. Each pixel is assigned to
    the largest mask that covers it, removing it from smaller overlapping masks.
    This matches LSP-DETR's post-processing for fair comparison.
    """
    if len(masks) == 0:
        return masks

    n = len(masks)
    h, w = masks[0].shape
    stack = np.stack(masks)  # (N, H, W)
    areas = stack.sum(axis=(1, 2))
    sorted_indices = np.argsort(-areas)  # largest first

    occupied = np.zeros((h, w), dtype=bool)
    resolved = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]

    for idx in sorted_indices:
        resolved[idx] = stack[idx] & ~occupied
        occupied |= resolved[idx].astype(bool)

    return resolved


def _mask_iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise mask IoU between predictions and GT."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)

    pred_stack = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)
    gt_stack = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)

    intersection = pred_stack @ gt_stack.T
    pred_area = pred_stack.sum(axis=1, keepdims=True)
    gt_area = gt_stack.sum(axis=1, keepdims=True)
    union = pred_area + gt_area.T - intersection

    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)


def _compute_pq_masked(pred_masks, gt_masks, iou_threshold=0.5, mask=None):
    """Compute PQ with optional foreground mask (bMPQ / mMPQ style)."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    if mask is not None:
        pred_masks = [m & mask for m in pred_masks]
        gt_masks = [m & mask for m in gt_masks]

    iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold

    tp = valid.sum()
    fp = n_pred - tp
    fn = n_gt - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(iou_matrix[row_ind[valid], col_ind[valid]].mean()) if tp > 0 else 0.0
    pq = sq * dq
    return float(pq), float(sq), float(dq)


def _compute_ap(recall, precision):
    """Compute average precision from recall and precision arrays."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def compute_aji(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_threshold: float = 0.5) -> float:
    """Compute Aggregated Jaccard Index for one image.

    AJI = sum_intersections / (sum_unions + unmatched_pred_area + unmatched_gt_area)

    Uses Hungarian matching at iou_threshold to find optimal pred<->gt pairs.
    """
    if len(gt_masks) == 0:
        return 0.0
    if len(pred_masks) == 0:
        return 0.0

    iou_matrix = _mask_iou_matrix(pred_masks, gt_masks)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold
    match_pred = set(row_ind[valid].tolist())
    match_gt = set(col_ind[valid].tolist())

    total_intersection = 0.0
    total_union = 0.0
    for r, c in zip(row_ind[valid], col_ind[valid]):
        p = pred_masks[r].astype(np.float64)
        g = gt_masks[c].astype(np.float64)
        total_intersection += (p * g).sum()
        total_union += (p + g - p * g).sum()

    for i in range(len(pred_masks)):
        if i not in match_pred:
            total_union += pred_masks[i].astype(np.float64).sum()

    for j in range(len(gt_masks)):
        if j not in match_gt:
            total_union += gt_masks[j].astype(np.float64).sum()

    if total_union == 0:
        return 0.0

    return total_intersection / total_union


# ---------------------------------------------------------------------------
# Per-image result extraction (adapts StarDist outputs to metric pipeline)
# ---------------------------------------------------------------------------


def _extract_image_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    details: dict,
    class_dict: dict[int, int] | None,
) -> dict:
    """Extract per-image data in the format expected by compute_metrics_streaming.

    Converts StarDist label maps to mask lists + centroid/conf/class arrays.
    """
    H, W = y_true.shape

    # --- GT ---
    gt_masks = label_map_to_masks(y_true)
    gt_centroids = label_map_to_centroids(y_true)  # (N_gt, 2) in (y, x)
    n_gt = len(gt_masks)

    # GT class IDs: from class_dict {instance_id: class_id (1-indexed)}
    # class_dict keys map to label_map IDs. But some instance IDs might have
    # been dropped (empty masks), so we need to remap.
    gt_cls = np.zeros(n_gt, dtype=int)
    if class_dict is not None:
        # Re-derive class for each surviving mask.
        # GT masks are ordered by instance ID (1..max), skipping empty ones.
        gt_mask_ids = []
        for i in range(1, int(y_true.max()) + 1):
            if (y_true == i).any():
                gt_mask_ids.append(i)
        for new_idx, orig_id in enumerate(gt_mask_ids):
            cls_val = class_dict.get(orig_id, 0)
            gt_cls[new_idx] = cls_val - 1 if cls_val > 0 else 0  # 0-indexed for eval
    else:
        gt_cls = np.zeros(n_gt, dtype=int)

    # --- Predictions ---
    pred_masks = label_map_to_masks(y_pred)
    pred_masks = resolve_mask_overlaps(pred_masks)

    n_pred = len(pred_masks)
    # StarDist details['points'] are in (y, x) order, shape (n_pred, 2)
    points = details.get("points", np.zeros((0, 2)))
    if len(points) != n_pred:
        # Fallback: compute centroids from label map
        pred_centroids = label_map_to_centroids(y_pred)
    else:
        pred_centroids = np.asarray(points[:n_pred], dtype=np.float64)

    pred_confs = np.asarray(details.get("prob", np.ones(n_pred)), dtype=np.float32)
    if len(pred_confs) != n_pred:
        pred_confs = np.ones(n_pred, dtype=np.float32)

    pred_cls_ids = details.get("class_id", None)
    if pred_cls_ids is not None and len(pred_cls_ids) >= n_pred:
        pred_cls = np.asarray(pred_cls_ids[:n_pred], dtype=int)
        pred_cls = np.where(pred_cls > 0, pred_cls - 1, 0)  # to 0-indexed
    else:
        pred_cls = np.zeros(n_pred, dtype=int)

    return {
        "gt_masks": gt_masks,
        "pred_masks": pred_masks,
        "gt_centroids": gt_centroids,
        "pred_centroids": pred_centroids,
        "gt_cls": gt_cls,
        "pred_cls": pred_cls,
        "pred_confs": pred_confs,
        "imgsz_h": H,
        "imgsz_w": W,
    }


# ---------------------------------------------------------------------------
# Streaming metric computation (ported from RayCastED compute_metrics_streaming)
# ---------------------------------------------------------------------------


def compute_metrics_streaming(results: list[dict], num_classes: int) -> dict:
    """Compute all metrics in a single pass, one image at a time.

    Peak memory = O(max_masks_per_image * H * W).
    """
    iou_thresholds = sorted(set(round(x, 2) for x in np.arange(0.5, 1.0, 0.05)))

    # Accumulators
    aji_scores: list[float] = []
    bpq_scores: list[float] = []
    bmpq_scores: list[float] = []
    class_pq: dict[int, list[float]] = {c: [] for c in range(num_classes)}
    class_mpq: dict[int, list[float]] = {c: [] for c in range(num_classes)}
    centroid_tp = 0
    centroid_fp = 0
    centroid_fn = 0
    tissue_aji: dict[int, list[float]] = {t: [] for t in range(19)}
    tissue_bpq: dict[int, list[float]] = {t: [] for t in range(19)}
    tissue_mpq: dict[int, dict[int, list[float]]] = {
        t: {c: [] for c in range(num_classes)} for t in range(19)
    }

    # AP accumulators — per class, per threshold
    class_set: set[int] = set()
    for r in results:
        class_set.update(r["gt_cls"].tolist())
        class_set.update(r["pred_cls"].tolist())

    ap_stats: dict[int, dict] = {}
    for cls_id in sorted(class_set):
        ap_stats[cls_id] = {
            t: {"tp": [], "fp": [], "conf": [], "n_gt": 0} for t in iou_thresholds
        }

    for i, r in enumerate(results):
        imgsz_h = r["imgsz_h"]
        imgsz_w = r["imgsz_w"]
        pred_masks = r["pred_masks"]
        gt_masks = r["gt_masks"]
        pred_cls = r["pred_cls"]
        gt_cls = r["gt_cls"]
        pred_confs = r["pred_confs"]
        pred_centroids = r["pred_centroids"]
        gt_centroids = r["gt_centroids"]

        if i == 0:
            print(
                f"  First image: {len(pred_masks)} preds, {len(gt_masks)} GT, "
                f"size={imgsz_h}x{imgsz_w}",
                flush=True,
            )

        # --- AJI ---
        aji_val = compute_aji(pred_masks, gt_masks)
        aji_scores.append(aji_val)

        # --- bPQ / bMPQ ---
        if len(pred_masks) > 0:
            pred_binary = np.stack(pred_masks).max(axis=0).astype(np.uint8)
        else:
            pred_binary = np.zeros((imgsz_h, imgsz_w), dtype=np.uint8)

        if len(gt_masks) > 0:
            gt_binary = np.stack(gt_masks).max(axis=0).astype(np.uint8)
        else:
            gt_binary = np.zeros((imgsz_h, imgsz_w), dtype=np.uint8)

        bpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary])
        bpq_scores.append(bpq)

        if gt_binary.sum() > 0:
            fg = gt_binary > 0
            bmpq, _, _ = _compute_pq_masked([pred_binary], [gt_binary], mask=fg)
        else:
            bmpq = 0.0
        bmpq_scores.append(bmpq)

        # --- mPQ / mMPQ ---
        class_pq_img = [0.0] * num_classes
        gt_idx_counts = [0] * num_classes
        pred_idx_counts = [0] * num_classes
        for cls_id in range(num_classes):
            pred_idx = [j for j, c in enumerate(pred_cls) if c == cls_id]
            gt_idx = [j for j, c in enumerate(gt_cls) if j < len(gt_cls) and c == cls_id]
            gt_idx_counts[cls_id] = len(gt_idx)
            pred_idx_counts[cls_id] = len(pred_idx)

            pred_cls_masks = [pred_masks[j] for j in pred_idx]
            gt_cls_masks = [gt_masks[j] for j in gt_idx]

            pq, _, _ = _compute_pq_masked(pred_cls_masks, gt_cls_masks)
            class_pq[cls_id].append(pq)
            class_pq_img[cls_id] = pq

            if len(gt_masks) > 0:
                gt_any = np.stack(gt_masks).max(axis=0).astype(np.uint8)
                fg = gt_any > 0
                mpq, _, _ = _compute_pq_masked(pred_cls_masks, gt_cls_masks, mask=fg)
            else:
                mpq = 0.0
            class_mpq[cls_id].append(mpq)

        # --- AP (per-class Hungarian matching) ---
        for cls_id in sorted(class_set):
            cls_pred_masks = [
                pred_masks[j]
                for j in range(len(pred_masks))
                if j < len(pred_cls) and pred_cls[j] == cls_id
            ]
            cls_gt_masks = [
                gt_masks[j]
                for j in range(len(gt_masks))
                if j < len(gt_cls) and gt_cls[j] == cls_id
            ]
            cls_confs = pred_confs[pred_cls == cls_id]

            n_pred_cls = len(cls_pred_masks)
            n_gt_cls = len(cls_gt_masks)

            for t in iou_thresholds:
                ap_stats[cls_id][t]["n_gt"] += n_gt_cls

            if n_pred_cls > 0 and n_gt_cls > 0:
                iou_mat = _mask_iou_matrix(cls_pred_masks, cls_gt_masks)
                row_ind, col_ind = linear_sum_assignment(-iou_mat)
                matched_iou = iou_mat[row_ind, col_ind]
            else:
                row_ind = np.array([], dtype=int)
                matched_iou = np.array([], dtype=float)

            for t in iou_thresholds:
                if n_gt_cls > 0:
                    valid = matched_iou >= t
                    matched_pred = set(row_ind[valid].tolist())
                else:
                    matched_pred = set()

                for pi in range(n_pred_cls):
                    is_tp = pi in matched_pred
                    ap_stats[cls_id][t]["conf"].append(
                        float(cls_confs[pi]) if pi < len(cls_confs) else 0.0
                    )
                    ap_stats[cls_id][t]["tp"].append(is_tp)
                    ap_stats[cls_id][t]["fp"].append(not is_tp)

        # --- Centroid F1 ---
        n_pred = len(pred_centroids)
        n_gt = len(gt_centroids)
        if n_pred > 0 and n_gt > 0:
            dist_matrix = np.linalg.norm(
                gt_centroids[:, None, :] - pred_centroids[None, :, :], axis=2
            )
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            tp = int((dist_matrix[row_ind, col_ind] <= 12).sum())
        else:
            tp = 0
        centroid_tp += tp
        centroid_fp += n_pred - tp
        centroid_fn += n_gt - tp

        # Per-tissue tracking
        tissue = int(r.get("tissue", 0))
        if tissue < 19 and aji_val > -1:
            tissue_aji[tissue].append(aji_val)
            tissue_bpq[tissue].append(bpq)
            for cls_id in range(num_classes):
                if gt_idx_counts[cls_id] > 0 or pred_idx_counts[cls_id] > 0:
                    tissue_mpq[tissue][cls_id].append(class_pq_img[cls_id])

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(results)} images", flush=True)

    # --- Aggregate ---

    mean_aji = float(np.mean(aji_scores)) if aji_scores else 0.0
    mean_bpq = float(np.mean(bpq_scores)) if bpq_scores else 0.0
    mean_bmpq = float(np.mean(bmpq_scores)) if bmpq_scores else 0.0

    mpq_values = []
    mmpq_values = []
    for c in range(num_classes):
        valid_pq = [v for v in class_pq[c] if v > 0]
        valid_mpq = [v for v in class_mpq[c] if v > 0]
        if valid_pq:
            mpq_values.append(float(np.mean(valid_pq)))
        if valid_mpq:
            mmpq_values.append(float(np.mean(valid_mpq)))
    mean_mpq = float(np.mean(mpq_values)) if mpq_values else 0.0
    mean_mmpq = float(np.mean(mmpq_values)) if mmpq_values else 0.0

    # AP
    ap_results = {}
    for t in iou_thresholds:
        aps = []
        for cls_id in sorted(class_set):
            stats = ap_stats[cls_id][t]
            n_gt = stats["n_gt"]
            if n_gt == 0:
                continue
            confs = np.array(stats["conf"])
            tps = np.array(stats["tp"])
            fps = np.array(stats["fp"])
            if len(confs) == 0:
                aps.append(0.0)
                continue
            order = np.argsort(-confs)
            tps = tps[order]
            fps = fps[order]
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / n_gt
            aps.append(_compute_ap(recall, precision))
        ap_results[t] = {"AP": float(np.mean(aps)) if aps else 0.0}

    # Centroid F1
    prec = centroid_tp / (centroid_tp + centroid_fp) if (centroid_tp + centroid_fp) > 0 else 0.0
    rec = centroid_tp / (centroid_tp + centroid_fn) if (centroid_tp + centroid_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "aji": mean_aji,
        "bpq": mean_bpq,
        "bmpq": mean_bmpq,
        "mpq": mean_mpq,
        "mmpq": mean_mmpq,
        "ap": ap_results,
        "centroid": {"precision": prec, "recall": rec, "f1": f1},
        "tissue_aji": tissue_aji,
        "tissue_bpq": tissue_bpq,
        "tissue_mpq": tissue_mpq,
    }


# ---------------------------------------------------------------------------
# Diagnostic breakdowns (ported from RayCastED)
# ---------------------------------------------------------------------------


def _diagnose_tissue(results, metrics):
    """Per-tissue centroid F1 + AJI/bPQ/mPQ."""
    _per_group_metrics(results, TISSUE_TYPES, "tissue", "Tissue Type Breakdown", metrics)


def _diagnose_nuclei(results, num_classes):
    """Per-class centroid F1 metrics."""
    return _per_nuclei_class_metrics(results, num_classes, _NUCLEI_NAMES)


def _per_group_metrics(results, names, key, title, metrics=None):
    """Compute per-group centroid F1 + optional AJI/bPQ/mPQ."""
    n_groups = len(names)
    tp = [0] * n_groups
    fp = [0] * n_groups
    fn = [0] * n_groups
    n_gt = [0] * n_groups
    n_pred = [0] * n_groups
    seen = [set() for _ in range(n_groups)]

    for ri, r in enumerate(results):
        g = int(r.get(key, 0))
        if g >= n_groups:
            continue
        seen[g].add(ri)
        gt_c = r["gt_centroids"]
        pred_c = r["pred_centroids"]
        n_gt_g = len(gt_c)
        n_pred_g = len(pred_c)
        n_gt[g] += n_gt_g
        n_pred[g] += n_pred_g
        if n_gt_g == 0:
            fp[g] += n_pred_g
            continue
        if n_pred_g == 0:
            fn[g] += n_gt_g
            continue
        dist = np.linalg.norm(gt_c[:, None] - pred_c[None, :], axis=2)
        ri_, ci_ = linear_sum_assignment(dist)
        t = int((dist[ri_, ci_] <= 12).sum())
        tp[g] += t
        fp[g] += n_pred_g - t
        fn[g] += n_gt_g - t

    t_aji = metrics.get("tissue_aji", {}) if metrics else {}
    t_bpq = metrics.get("tissue_bpq", {}) if metrics else {}
    t_mpq = metrics.get("tissue_mpq", {}) if metrics else {}
    has_mask = bool(metrics)

    sep = "=" * (85 if has_mask else 60)
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    col = ["Group", "Imgs", "Prec", "Recall", "F1"]
    hdr_fmt = "  {:<14} {:>5} {:>7} {:>7} {:>7}"
    row_fmt = "  {:<14} {:>5} {:>7.4f} {:>7.4f} {:>7.4f}"
    if has_mask:
        col += ["AJI", "bPQ", "mPQ"]
        hdr_fmt += " {:>7} {:>7} {:>7}"
        row_fmt += " {:>7.4f} {:>7.4f} {:>7.4f}"
    print(hdr_fmt.format(*col))
    sep2 = "  " + "-" * 14 + " " + "-" * 5 + " " + "-" * 7 + " " + "-" * 7 + " " + "-" * 7
    if has_mask:
        sep2 += " " + "-" * 7 + " " + "-" * 7 + " " + "-" * 7
    print(sep2)

    for g in range(n_groups):
        if n_gt[g] == 0:
            continue
        prec = tp[g] / (tp[g] + fp[g]) if (tp[g] + fp[g]) else 0
        rec = tp[g] / (tp[g] + fn[g]) if (tp[g] + fn[g]) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        vals = [names[g], len(seen[g]), prec, rec, f1]
        if has_mask:
            aji_arr = np.array(t_aji.get(g)) if t_aji.get(g) else np.array([])
            bpq_arr = np.array(t_bpq.get(g)) if t_bpq.get(g) else np.array([])
            aji_m = float(np.mean(aji_arr)) if len(aji_arr) else 0.0
            bpq_m = float(np.mean(bpq_arr)) if len(bpq_arr) else 0.0
            mpq_vals = [
                np.mean([x for x in v if x > 0])
                for v in t_mpq.get(g, {}).values()
                if v and any(x > 0 for x in v)
            ]
            mpq_m = float(np.mean(mpq_vals)) if mpq_vals else 0.0
            vals += [aji_m, bpq_m, mpq_m]
        print(row_fmt.format(*vals))

    tot_tp = sum(tp)
    tot_fp = sum(fp)
    tot_fn = sum(fn)
    p_t = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0
    r_t = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0
    f_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) else 0
    vals = ["TOTAL", len(results), p_t, r_t, f_t]
    if has_mask:
        vals += [metrics["aji"], metrics["bpq"], metrics["mpq"]]
    print(row_fmt.format(*vals))

    if has_mask:
        per_t_aji = [float(np.mean(t_aji.get(g))) for g in range(n_groups) if t_aji.get(g)]
        per_t_bpq = [float(np.mean(t_bpq.get(g))) for g in range(n_groups) if t_bpq.get(g)]
        per_t_mpq = []
        for g in range(n_groups):
            v = t_mpq.get(g, {})
            if v:
                vp = [
                    np.mean([x for x in lst if x > 0])
                    for lst in v.values()
                    if lst and any(x > 0 for x in lst)
                ]
                if vp:
                    per_t_mpq.append(float(np.mean(vp)))
        am = np.mean(per_t_aji) if per_t_aji else 0
        as_ = np.std(per_t_aji) if len(per_t_aji) > 1 else 0
        bm = np.mean(per_t_bpq) if per_t_bpq else 0
        bs_ = np.std(per_t_bpq) if len(per_t_bpq) > 1 else 0
        mm = np.mean(per_t_mpq) if per_t_mpq else 0
        ms_ = np.std(per_t_mpq) if len(per_t_mpq) > 1 else 0
        print(f'  {"Avg":<14} {"":>5} {"":>7} {"":>7} {"":>7} {am:>7.4f} {bm:>7.4f} {mm:>7.4f}')
        print(f'  {"Std":<14} {"":>5} {"":>7} {"":>7} {"":>7} {as_:>7.4f} {bs_:>7.4f} {ms_:>7.4f}')
    print(sep)


def _per_nuclei_class_metrics(results, num_classes, names):
    """Per-class centroid F1."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for r in results:
        gt_c = r["gt_centroids"]
        pred_c = r["pred_centroids"]
        gt_cls = r["gt_cls"]
        pred_cls = r["pred_cls"]
        n_gt = len(gt_c)
        n_pred = len(pred_c)
        if n_gt == 0 or n_pred == 0:
            continue
        dist = np.linalg.norm(gt_c[:, None] - pred_c[None, :], axis=2)
        ri_, ci_ = linear_sum_assignment(dist)
        for ri, ci in zip(ri_, ci_):
            if dist[ri, ci] <= 12:
                if gt_cls[ri] == pred_cls[ci]:
                    tp[int(gt_cls[ri])] += 1

    print(f'\n{"=" * 70}')
    print(f"  Nuclei Class Breakdown (Centroid F1, class-matched)")
    print(f'{"=" * 70}')
    print(f'  {"Class":<14} {"Prec":>7} {"Recall":>7} {"F1":>7}')
    print(f'  {"-" * 14} {"-" * 7} {"-" * 7} {"-" * 7}')
    class_gt = [0] * num_classes
    class_pred = [0] * num_classes
    for r in results:
        for c in r["gt_cls"]:
            class_gt[int(c)] += 1
        for c in r["pred_cls"]:
            if int(c) < num_classes:
                class_pred[int(c)] += 1
    for c in range(num_classes):
        name = names[c] if c < len(names) else f"cls_{c}"
        prec = tp[c] / class_pred[c] if class_pred[c] else 0
        rec = tp[c] / class_gt[c] if class_gt[c] else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"  {name:<14} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")
    print(f'{"=" * 70}')


def _diagnose_recall(results, num_classes):
    """Break down recall by GT size bin, class, and nearest-prediction distance."""
    gt_areas_matched: list[float] = []
    gt_areas_unmatched: list[float] = []
    class_total = [0] * num_classes
    class_matched = [0] * num_classes
    unmatched_dists: list[float] = []

    for r in results:
        gt_centroids = r["gt_centroids"]
        pred_centroids = r["pred_centroids"]
        pred_confs = r["pred_confs"]
        pred_cls = r["pred_cls"]
        gt_cls = r["gt_cls"]
        gt_masks = r["gt_masks"]

        n_gt = len(gt_centroids)
        n_pred = len(pred_centroids)

        if n_gt == 0:
            continue

        gt_areas = np.array([float(m.sum()) for m in gt_masks])

        if n_pred > 0:
            dist_matrix = np.linalg.norm(
                gt_centroids[:, None, :] - pred_centroids[None, :, :], axis=2
            )
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            matched = np.zeros(n_gt, dtype=bool)
            matched_dists = dist_matrix[row_ind, col_ind]
            for ri, ci, d in zip(row_ind, col_ind, matched_dists):
                if d <= 12.0:
                    matched[ri] = True
        else:
            matched = np.zeros(n_gt, dtype=bool)

        for j in range(n_gt):
            cls_id = int(gt_cls[j])
            if cls_id < num_classes:
                class_total[cls_id] += 1
                if matched[j]:
                    class_matched[cls_id] += 1

            if matched[j]:
                gt_areas_matched.append(float(gt_areas[j]))
            else:
                gt_areas_unmatched.append(float(gt_areas[j]))

                if n_pred > 0:
                    dists_j = np.linalg.norm(pred_centroids - gt_centroids[j], axis=1)
                    nearest_idx = int(np.argmin(dists_j))
                    unmatched_dists.append(float(dists_j[nearest_idx]))
                else:
                    unmatched_dists.append(float("inf"))

    total_gt = sum(class_total)
    total_matched = sum(class_matched)
    recall = total_matched / total_gt if total_gt > 0 else 0.0

    print("\n" + "=" * 65)
    print("Recall Diagnosis")
    print("=" * 65)
    print(f"  Overall: {total_matched}/{total_gt} = {recall:.4f}")

    print(f'\n  {"Class":<18} {"Total":>8} {"Matched":>8} {"Recall":>8}')
    print(f'  {"-" * 18} {"-" * 8} {"-" * 8} {"-" * 8}')
    for c in range(num_classes):
        if class_total[c] > 0:
            r = class_matched[c] / class_total[c]
            name = _NUCLEI_NAMES[c] if c < len(_NUCLEI_NAMES) else f"class_{c}"
            print(f"  {name:<18} {class_total[c]:>8} {class_matched[c]:>8} {r:>8.4f}")

    if gt_areas_matched or gt_areas_unmatched:
        all_areas = np.array(gt_areas_matched + gt_areas_unmatched)
        if len(all_areas) > 0:
            p33 = np.percentile(all_areas, 33)
            p67 = np.percentile(all_areas, 67)
            bins = [
                ("Small (<P33)", lambda a: a < p33),
                ("Medium (P33-P67)", lambda a: (a >= p33) & (a < p67)),
                ("Large (>P67)", lambda a: a >= p67),
            ]
            print(f'\n  {"Size Bin":<20} {"Total":>8} {"Matched":>8} {"Recall":>8}')
            print(f'  {"-" * 20} {"-" * 8} {"-" * 8} {"-" * 8}')
            for label, cond in bins:
                t = sum(1 for a in gt_areas_matched + gt_areas_unmatched if cond(a))
                m = sum(1 for a in gt_areas_matched if cond(a))
                rec = m / t if t > 0 else 0.0
                print(f"  {label:<20} {t:>8} {m:>8} {rec:>8.4f}")

    if unmatched_dists:
        ud = np.array(unmatched_dists)
        finite = ud[np.isfinite(ud)]
        print(f"\n  Unmatched GT — Distance to nearest prediction:")
        print(
            f"    <5px (near miss): {int(np.sum(finite < 5))}/{len(unmatched_dists)} "
            f"({100 * np.sum(finite < 5) / len(unmatched_dists):.1f}%)"
        )
        print(
            f"    5-12px (drifted):  {int(np.sum((finite >= 5) & (finite <= 12)))}/{len(unmatched_dists)} "
            f"({100 * np.sum((finite >= 5) & (finite <= 12)) / len(unmatched_dists):.1f}%)"
        )
        print(
            f"    >12px (truly miss):{int(np.sum(finite > 12))}/{len(unmatched_dists)} "
            f"({100 * np.sum(finite > 12) / len(unmatched_dists):.1f}%)"
        )
        no_det = int(np.isinf(ud).sum())
        if no_det > 0:
            print(
                f"    No predictions:    {no_det}/{len(unmatched_dists)} "
                f"({100 * no_det / len(unmatched_dists):.1f}%)"
            )

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(
    use_classes: bool = True,
    max_samples: int | None = None,
    iou_threshold: float = 0.5,
    prob_thresh: float | None = None,
    nms_thresh: float | None = None,
    model_name: str = MODEL_NAME,
    model_basedir: str = MODEL_BASEDIR,
    benchmark: bool = True,
) -> dict:
    """Run inference on PanNuke fold3 and compute LSP-DETR aligned metrics.

    Returns a dict with all metric values + per-tissue / per-class breakdowns.
    """
    print("Loading fold3 (test)...")
    X_test, Y_test, C_test, T_test = load_fold_cached(
        "fold3",
        use_classes=use_classes,
        max_samples=max_samples,
        load_tissue=True,
    )

    model = load_model(name=model_name, basedir=model_basedir)

    n_classes = len(CELL_TYPES)
    results: list[dict] = []

    # Inference time benchmark
    inf_times: list[float] = []

    for idx in tqdm(range(len(X_test)), desc="Evaluating", unit="img"):
        img = X_test[idx]
        y_true = Y_test[idx]
        class_dict = C_test[idx] if C_test is not None else None
        tissue = T_test[idx] if T_test is not None else 0

        kwargs: dict = {}
        if prob_thresh is not None:
            kwargs["prob_thresh"] = prob_thresh
        if nms_thresh is not None:
            kwargs["nms_thresh"] = nms_thresh

        if benchmark:
            t0 = time.perf_counter()
            y_pred, details = model.predict_instances(img, **kwargs)
            inf_times.append((time.perf_counter() - t0) * 1000)
        else:
            y_pred, details = model.predict_instances(img, **kwargs)

        r = _extract_image_data(y_true, y_pred, details, class_dict)
        r["tissue"] = tissue
        results.append(r)

    n_pred_total = sum(len(r["pred_masks"]) for r in results)
    n_gt_total = sum(len(r["gt_masks"]) for r in results)
    print(
        f"  Processed {len(results)} images: {n_pred_total} predictions, {n_gt_total} GT",
        flush=True,
    )

    # --- Compute all metrics ---
    print("Computing metrics (streaming)...", flush=True)
    metrics = compute_metrics_streaming(results, num_classes=n_classes)

    ap_results = metrics["ap"]
    ap50 = ap_results.get(0.5, {}).get("AP", 0.0)
    ap70 = ap_results.get(0.7, {}).get("AP", 0.0)
    ap90 = ap_results.get(0.9, {}).get("AP", 0.0)
    ap50_95 = float(np.mean([ap_results[t]["AP"] for t in sorted(ap_results.keys())]))

    metrics["ap50"] = ap50
    metrics["ap70"] = ap70
    metrics["ap90"] = ap90
    metrics["ap50_95"] = ap50_95

    if inf_times:
        # Drop first image (warmup)
        if len(inf_times) > 1:
            metrics["inference_ms"] = float(np.mean(inf_times[1:]))
        else:
            metrics["inference_ms"] = float(np.mean(inf_times))

    metrics["n_pred_total"] = n_pred_total
    metrics["n_gt_total"] = n_gt_total
    metrics["n_images"] = len(results)

    # Stash results for diagnostics
    metrics["_results"] = results

    return metrics


def print_results(metrics: dict) -> None:
    """Print results in the same format as RayCastED eval_pannuke.py."""
    results = metrics.pop("_results", None)
    n_classes = len(CELL_TYPES)
    print("\n" + "=" * 60, flush=True)
    print("PanNuke Fold3 Evaluation Results (LSP-DETR Protocol)")
    print("=" * 60)
    print(f'{"Metric":<25} {"Value":>12}')
    print("-" * 37)
    print(f'{"AJI":<25} {metrics["aji"]:>12.4f}')
    print(f'{"AP@0.5":<25} {metrics["ap50"]:>12.4f}')
    print(f'{"AP@0.7":<25} {metrics["ap70"]:>12.4f}')
    print(f'{"AP@0.9":<25} {metrics["ap90"]:>12.4f}')
    print(f'{"AP@0.5:0.05:0.95":<25} {metrics["ap50_95"]:>12.4f}')
    print(f'{"bPQ":<25} {metrics["bpq"]:>12.4f}')
    print(f'{"bMPQ":<25} {metrics["bmpq"]:>12.4f}')
    print(f'{"mPQ":<25} {metrics["mpq"]:>12.4f}')
    print(f'{"mMPQ":<25} {metrics["mmpq"]:>12.4f}')
    f12 = metrics["centroid"]
    print(f'{"F1 (centroid, r=12)":<25} {f12["f1"]:>12.4f}')
    print(f'{"Precision (centroid)":<25} {f12["precision"]:>12.4f}')
    print(f'{"Recall (centroid)":<25} {f12["recall"]:>12.4f}')
    if "inference_ms" in metrics:
        print(f'{"Inference Time (ms/img)":<25} {metrics["inference_ms"]:>12.2f}')
    print("=" * 37)

    print(f'\nImages evaluated: {metrics["n_images"]}')
    print(f'Total predictions: {metrics["n_pred_total"]}')
    print(f'Total GT instances: {metrics["n_gt_total"]}')

    # --- Tissue-origin breakdown ---
    if results is not None:
        try:
            _diagnose_tissue(results, metrics)
            _diagnose_nuclei(results, n_classes)
        except Exception as exc:
            print(f"\n[DIAG ERROR] {exc}", flush=True)

        try:
            _diagnose_recall(results, num_classes=n_classes)
        except Exception as exc:
            import traceback

            print(f"\n[DIAG ERROR] _diagnose_recall failed: {exc}", flush=True)
            traceback.print_exc()


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
    parser.add_argument("--no-benchmark", action="store_true", help="Skip inference timing")
    args = parser.parse_args()

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

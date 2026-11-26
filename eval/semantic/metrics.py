from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass
class MetricResult:
    iou: np.ndarray
    accuracy: np.ndarray
    gt_support: np.ndarray
    pred_support: np.ndarray
    true_positive: np.ndarray
    split_metrics: Dict[str, Dict[str, float]]
    pixel_accuracy: float


def update_confusion(
    confusion: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> int:
    """
    Accumulate predictions into a confusion matrix.

    Convention:
        rows    = ground-truth classes
        columns = predicted classes

    Returns:
        Number of valid evaluated pixels.
    """
    if gt.shape != pred.shape:
        raise ValueError(
            f"GT/prediction shape mismatch: {gt.shape} vs {pred.shape}"
        )

    expected_shape = (num_classes, num_classes)
    if confusion.shape != expected_shape:
        raise ValueError(
            f"Confusion matrix must have shape {expected_shape}, "
            f"got {confusion.shape}"
        )

    gt = np.asarray(gt)
    pred = np.asarray(pred)

    valid = (
        (gt != ignore_index)
        & (gt >= 0)
        & (gt < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )

    num_valid = int(valid.sum())

    if num_valid == 0:
        return 0

    gt_valid = gt[valid].astype(np.int64, copy=False)
    pred_valid = pred[valid].astype(np.int64, copy=False)

    # Encode each (GT, prediction) pair as:
    #
    #     encoded = gt * num_classes + pred
    #
    # so that reshape gives:
    #
    #     confusion[gt_class, predicted_class]
    encoded = gt_valid * num_classes + pred_valid

    counts = np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    confusion += counts.astype(confusion.dtype, copy=False)

    return num_valid


def compute_metrics(
    confusion: np.ndarray,
    splits: Optional[Mapping[str, Sequence[int]]] = None,
) -> MetricResult:
    """
    Compute semantic-segmentation metrics from a confusion matrix.

    Convention:
        rows    = ground-truth classes
        columns = predicted classes

    Macro metrics (mIoU and mAcc) are averaged only over classes
    that have ground-truth support.

    FmIoU is weighted by the number of GT pixels belonging to
    each class.
    """
    confusion = np.asarray(confusion)

    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(
            f"Confusion matrix must be square, got shape {confusion.shape}"
        )

    num_classes = confusion.shape[0]
    confusion_f = confusion.astype(np.float64, copy=False)

    true_positive = np.diag(confusion_f)
    gt_support = confusion_f.sum(axis=1)
    pred_support = confusion_f.sum(axis=0)

    union = gt_support + pred_support - true_positive

    # Classes absent from the GT should not contribute to macro averages.
    iou = np.full(num_classes, np.nan, dtype=np.float64)
    accuracy = np.full(num_classes, np.nan, dtype=np.float64)

    valid_gt = gt_support > 0

    iou[valid_gt] = (
        true_positive[valid_gt] / union[valid_gt]
    )

    accuracy[valid_gt] = (
        true_positive[valid_gt] / gt_support[valid_gt]
    )

    # "all" always exists.
    all_splits: Dict[str, Sequence[int]] = {
        "all": list(range(num_classes))
    }

    if splits is not None:
        for name, indices in splits.items():
            if name == "all":
                raise ValueError(
                    "'all' is reserved and is created automatically."
                )
            all_splits[name] = indices

    split_metrics: Dict[str, Dict[str, float]] = {}

    for split_name, indices in all_splits.items():
        ids = np.asarray(list(indices), dtype=np.int64)

        if ids.size == 0:
            raise ValueError(f"Split '{split_name}' contains no classes.")

        if np.any(ids < 0) or np.any(ids >= num_classes):
            raise ValueError(
                f"Split '{split_name}' contains class IDs outside "
                f"[0, {num_classes - 1}]: {ids.tolist()}"
            )

        split_gt_support = gt_support[ids]
        split_iou = iou[ids]
        split_accuracy = accuracy[ids]

        present = split_gt_support > 0
        classes_with_gt = int(present.sum())

        if classes_with_gt > 0:
            miou = float(np.mean(split_iou[present]))
            macc = float(np.mean(split_accuracy[present]))

            weights = (
                split_gt_support[present]
                / split_gt_support[present].sum()
            )

            fmiou = float(
                np.sum(weights * split_iou[present])
            )
        else:
            miou = float("nan")
            macc = float("nan")
            fmiou = float("nan")

        split_metrics[split_name] = {
            "mIoU": miou,
            "FmIoU": fmiou,
            "mAcc": macc,
            "classes_with_gt": classes_with_gt,
            "num_classes": int(ids.size),
        }

    total_pixels = confusion_f.sum()

    if total_pixels > 0:
        pixel_accuracy = float(
            true_positive.sum() / total_pixels
        )
    else:
        pixel_accuracy = float("nan")

    return MetricResult(
        iou=iou,
        accuracy=accuracy,
        gt_support=gt_support,
        pred_support=pred_support,
        true_positive=true_positive,
        split_metrics=split_metrics,
        pixel_accuracy=pixel_accuracy,
    )


def merge_confusions(confusions):
    """
    Sum multiple confusion matrices into a single global matrix.
    """
    confusions = list(confusions)

    if not confusions:
        raise ValueError("No confusion matrices were provided.")

    first = np.asarray(confusions[0])
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError(
            f"Confusion matrix must be square, got {first.shape}"
        )

    merged = np.zeros_like(first)

    for i, confusion in enumerate(confusions):
        confusion = np.asarray(confusion)

        if confusion.shape != first.shape:
            raise ValueError(
                f"Confusion matrix {i} has shape {confusion.shape}, "
                f"expected {first.shape}"
            )

        merged += confusion

    return merged
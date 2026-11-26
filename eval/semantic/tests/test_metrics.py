import numpy as np

from eval.semantic.metrics import (
    update_confusion,
    compute_metrics,
    merge_confusions,
)


def test_semantic_metrics_basic():
    gt = np.array([0, 0, 1, 1, 2, 255])
    pred = np.array([0, 1, 1, 1, 0, 2])

    confusion = np.zeros((3, 3), dtype=np.uint64)

    num_valid = update_confusion(
        confusion=confusion,
        gt=gt,
        pred=pred,
        num_classes=3,
        ignore_index=255,
    )

    assert num_valid == 5

    expected_confusion = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 0],
    ], dtype=np.uint64)

    assert np.array_equal(confusion, expected_confusion)

    metrics = compute_metrics(confusion)

    # Class 0:
    # TP=1, GT=2, Pred=2
    # IoU = 1 / (2 + 2 - 1) = 1/3
    assert np.isclose(metrics.iou[0], 1.0 / 3.0)

    # Class 1:
    # TP=2, GT=2, Pred=3
    # IoU = 2 / (2 + 3 - 2) = 2/3
    assert np.isclose(metrics.iou[1], 2.0 / 3.0)

    # Class 2:
    # TP=0, GT=1, Pred=0
    # IoU = 0 / 1 = 0
    assert np.isclose(metrics.iou[2], 0.0)

    # Class accuracies
    assert np.isclose(metrics.accuracy[0], 0.5)
    assert np.isclose(metrics.accuracy[1], 1.0)
    assert np.isclose(metrics.accuracy[2], 0.0)

    expected_miou = (1/3 + 2/3 + 0.0) / 3
    expected_macc = (0.5 + 1.0 + 0.0) / 3
    expected_pixel_acc = 3 / 5

    assert np.isclose(metrics.split_metrics["all"]["mIoU"], expected_miou)
    assert np.isclose(metrics.split_metrics["all"]["mAcc"], expected_macc)
    assert np.isclose(metrics.pixel_accuracy, expected_pixel_acc)


def test_merge_confusions():
    c1 = np.array([
        [1, 0],
        [0, 1],
    ], dtype=np.uint64)

    c2 = np.array([
        [2, 1],
        [1, 3],
    ], dtype=np.uint64)

    merged = merge_confusions([c1, c2])

    expected = np.array([
        [3, 1],
        [1, 4],
    ], dtype=np.uint64)

    assert np.array_equal(merged, expected)
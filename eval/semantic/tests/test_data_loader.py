from pathlib import Path

import numpy as np
import cv2
from eval.semantic.datasets.replica import ReplicaDataset
from eval.semantic.datasets.scannet import ScanNetDataset


def test_replica_dataset():
    dataset = ReplicaDataset(
        Path("/workspace/data/Replica")
    )

    frames = dataset.list_frames("office0")

    assert len(frames) > 0

    print("\nReplica first frame:", frames[0])

    gt = dataset.load_semantic_gt(frames[0])

    print("Replica GT shape:", gt.shape)
    print("Replica unique IDs:", np.unique(gt))

    assert gt.ndim == 2

    unique_ids = np.unique(gt)

    assert np.all(
        (unique_ids == 255)
        | ((unique_ids >= 0) & (unique_ids < 51))
    )


def test_scannet_dataset():
    dataset = ScanNetDataset(
        root=Path("/workspace/data/scannet"),
        label_file=Path(
            "/workspace/eval/scannetv2-labels.modified.tsv"
        ),
    )

    frames = dataset.list_frames("scene0011_00")

    assert len(frames) > 0

    print("\nScanNet number of frames:", len(frames))
    print("ScanNet first frame:", frames[0])

    gt = dataset.load_semantic_gt(frames[0])

    print("ScanNet GT shape:", gt.shape)
    print("ScanNet unique dense IDs:", np.unique(gt))

    assert gt.ndim == 2

    unique_ids = np.unique(gt)

    assert np.all(
        (unique_ids == 255)
        | ((unique_ids >= 0) & (unique_ids < 20))
    )

    # Make sure the mapping didn't accidentally ignore everything.
    assert np.any(gt != 255)
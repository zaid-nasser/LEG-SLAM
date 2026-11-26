from pathlib import Path

import numpy as np

from eval.reconstruction.datasets.replica import (
    ReplicaReconstructionDataset,
)
from eval.reconstruction.datasets.scannet import (
    ScanNetReconstructionDataset,
)

from eval.reconstruction.evaluator import ( 
    prepare_rgb_gt,
)

def test_prepare_rgb_gt_resizes_to_render_resolution():
    rgb = np.zeros((968, 1296, 3), dtype=np.uint8)

    prepared = prepare_rgb_gt(
        rgb,
        target_width=648,
        target_height=484,
    )

    assert prepared.shape == (484, 648, 3)

def test_replica_dataset():
    dataset = ReplicaReconstructionDataset(
        Path("/workspace/data/Replica")
    )

    frames = dataset.list_frames(
        "office0"
    )

    assert len(frames) > 0

    frame = frames[0]

    assert frame.frame_id == 0

    rgb = dataset.load_rgb_gt(frame)

    assert rgb.ndim == 3
    assert rgb.shape[2] == 3
    assert rgb.dtype == np.float32
    assert 0.0 <= rgb.min()
    assert rgb.max() <= 1.0

    depth = dataset.load_depth_gt_m(
        frame
    )

    assert depth.shape == rgb.shape[:2]
    assert depth.dtype == np.float32

    poses = dataset.load_gt_poses(
        "office0"
    )

    assert frame.frame_id in poses
    assert poses[frame.frame_id].shape == (4, 4)


def test_scannet_dataset():
    dataset = ScanNetReconstructionDataset(
        Path("/workspace/data/scannet")
    )

    frames = dataset.list_frames(
        "scene0011_01"
    )

    assert len(frames) > 0

    frame = frames[0]

    assert frame.frame_id == 0

    rgb = dataset.load_rgb_gt(frame)

    assert rgb.ndim == 3
    assert rgb.shape[2] == 3
    assert rgb.dtype == np.float32
    assert 0.0 <= rgb.min()
    assert rgb.max() <= 1.0

    assert (
        dataset.load_depth_gt_m(frame)
        is None
    )

    assert (
        dataset.load_gt_poses(
            "scene0011_00"
        )
        is None
    )
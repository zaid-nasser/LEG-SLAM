import numpy as np

from eval.reconstruction.metrics import (
    compute_depth_l1_cm,
    compute_first_pose_aligned_translation_metrics,
    compute_psnr,
    compute_ssim,
)


def test_identical_rgb():
    image = np.full(
        (16, 16, 3),
        0.5,
        dtype=np.float32,
    )

    assert np.isinf(
        compute_psnr(image, image)
    )

    assert np.isclose(
        compute_ssim(image, image),
        1.0,
    )


def test_depth_l1():
    gt = np.array(
        [[1.0, 2.0]],
        dtype=np.float32,
    )

    pred = np.array(
        [[1.1, 1.8]],
        dtype=np.float32,
    )

    result = compute_depth_l1_cm(
        gt,
        pred,
    )

    assert np.isclose(
        result,
        15.0,
    )


def test_identical_trajectory():
    poses = np.repeat(
        np.eye(4)[None],
        3,
        axis=0,
    )

    result = (
        compute_first_pose_aligned_translation_metrics(
            poses,
            poses,
        )
    )

    assert np.isclose(
        result.rmse_cm,
        0.0,
    )

    assert np.isclose(
        result.mean_cm,
        0.0,
    )
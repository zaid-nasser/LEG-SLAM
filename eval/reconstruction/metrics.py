from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from skimage.metrics import structural_similarity as structural_similarity


@dataclass(frozen=True)
class RGBMetrics:
    psnr: float
    ssim: float
    lpips: float


@dataclass(frozen=True)
class TrajectoryMetrics:
    rmse_cm: float
    mean_cm: float


def compute_psnr(
    gt: np.ndarray,
    pred: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Compute PSNR between two floating-point images.

    Expected input range is [0, data_range].
    """
    if gt.shape != pred.shape:
        raise ValueError(
            f"Image shape mismatch: gt={gt.shape}, pred={pred.shape}"
        )

    gt = gt.astype(np.float64, copy=False)
    pred = pred.astype(np.float64, copy=False)

    mse = np.mean((gt - pred) ** 2)

    if mse == 0:
        return float("inf")

    return float(
        20.0 * np.log10(
            data_range / np.sqrt(mse)
        )
    )


def compute_ssim(
    gt: np.ndarray,
    pred: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Compute RGB structural similarity.
    """
    if gt.shape != pred.shape:
        raise ValueError(
            f"Image shape mismatch: gt={gt.shape}, pred={pred.shape}"
        )

    if gt.ndim != 3 or gt.shape[2] != 3:
        raise ValueError(
            f"Expected RGB images [H,W,3], got {gt.shape}"
        )

    return float(
        structural_similarity(
            gt,
            pred,
            channel_axis=2,
            data_range=data_range,
        )
    )


@torch.inference_mode()
def compute_lpips(
    gt: np.ndarray,
    pred: np.ndarray,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Compute LPIPS for RGB images in [0,1].

    The LPIPS model should be constructed with normalize=True.
    """
    if gt.shape != pred.shape:
        raise ValueError(
            f"Image shape mismatch: gt={gt.shape}, pred={pred.shape}"
        )

    gt_tensor = (
        torch.from_numpy(gt)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    pred_tensor = (
        torch.from_numpy(pred)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    return float(
        lpips_model(
            gt_tensor,
            pred_tensor,
        ).item()
    )


def compute_rgb_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> RGBMetrics:
    """
    Compute PSNR, SSIM and LPIPS.

    Both images must be RGB float arrays in [0,1].
    """
    return RGBMetrics(
        psnr=compute_psnr(
            gt,
            pred,
            data_range=1.0,
        ),
        ssim=compute_ssim(
            gt,
            pred,
            data_range=1.0,
        ),
        lpips=compute_lpips(
            gt,
            pred,
            lpips_model,
            device,
        ),
    )


def compute_depth_l1_cm(
    gt_depth_m: np.ndarray,
    pred_depth_m: np.ndarray,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
) -> float:
    """
    Mean absolute depth error in centimeters.

    Inputs are depth maps in meters.
    """
    if gt_depth_m.shape != pred_depth_m.shape:
        raise ValueError(
            f"Depth shape mismatch: "
            f"gt={gt_depth_m.shape}, pred={pred_depth_m.shape}"
        )

    gt = gt_depth_m.astype(np.float64, copy=False)
    pred = pred_depth_m.astype(np.float64, copy=False)

    valid = (
        np.isfinite(gt)
        & np.isfinite(pred)
        & (gt > min_depth_m)
        & (gt < max_depth_m)
        & (pred > min_depth_m)
        & (pred < max_depth_m)
    )

    if not np.any(valid):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                gt[valid] - pred[valid]
            )
        )
        * 100.0
    )


def compute_first_pose_aligned_translation_metrics(
    gt_poses: np.ndarray,
    est_poses: np.ndarray,
) -> TrajectoryMetrics:
    """
    Preserve the trajectory metric currently used by replica_test.py.

    Both arrays must have shape [N,4,4].

    Estimated poses are aligned to GT using the first pose, after which
    translation-component RMSE and mean absolute translation-component
    error are reported in centimeters.
    """
    gt_poses = np.asarray(gt_poses, dtype=np.float64)
    est_poses = np.asarray(est_poses, dtype=np.float64)

    if gt_poses.shape != est_poses.shape:
        raise ValueError(
            f"Trajectory shape mismatch: "
            f"gt={gt_poses.shape}, est={est_poses.shape}"
        )

    if (
        gt_poses.ndim != 3
        or gt_poses.shape[1:] != (4, 4)
    ):
        raise ValueError(
            f"Expected trajectories [N,4,4], got {gt_poses.shape}"
        )

    if len(gt_poses) == 0:
        raise ValueError("Trajectory is empty")

    alignment = (
        np.linalg.inv(est_poses[0])
        @ gt_poses[0]
    )

    est_aligned = np.asarray(
        [
            alignment @ pose
            for pose in est_poses
        ]
    )

    translation_error_cm = (
        gt_poses[:, :3, 3]
        - est_aligned[:, :3, 3]
    ) * 100.0

    return TrajectoryMetrics(
        rmse_cm=float(
            np.sqrt(
                np.mean(
                    translation_error_cm ** 2
                )
            )
        ),
        mean_cm=float(
            np.mean(
                np.abs(
                    translation_error_cm
                )
            )
        ),
    )
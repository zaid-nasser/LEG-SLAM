from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import (
    LearnedPerceptualImagePatchSimilarity,
)
from tqdm import tqdm

from .datasets.base import ReconstructionDatasetAdapter
from .metrics import (
    compute_depth_l1_cm,
    compute_first_pose_aligned_translation_metrics,
    compute_rgb_metrics,
)

from ..common.gaussian_scene import (
    camera_frame_id,
    camera_pose_matrix,
    load_gaussian_scene,
    make_minicam,
)
from ..render import render



def prepare_rgb_gt(
    rgb: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """
    Bring a ground-truth RGB image to the reconstruction/render resolution.

    LEG-SLAM may operate at a resolution different from the raw dataset.
    Reconstruction metrics must compare GT and render at the same resolution.
    """
    rgb = np.asarray(rgb)

    if rgb.shape[:2] == (target_height, target_width):
        return rgb

    return cv2.resize(
        rgb,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _finite_or_none(
    value: float | None,
) -> float | None:
    if value is None:
        return None

    value = float(value)

    return (
        value
        if np.isfinite(value)
        else None
    )


def _mean_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    finite = [
        float(value)
        for value in values
        if np.isfinite(value)
    ]

    if not finite:
        return None

    return float(
        np.mean(finite)
    )


def _prepare_rendered_rgb(
    rendered_image: torch.Tensor,
) -> np.ndarray:
    """
    Convert renderer output [3,H,W] to RGB float32 [H,W,3].

    The uint8 round-trip intentionally preserves the behavior of the
    previous Replica/ScanNet reconstruction evaluators.
    """
    if rendered_image.ndim != 3:
        raise ValueError(
            f"Expected rendered image [3,H,W], "
            f"got {tuple(rendered_image.shape)}"
        )

    image = (
        rendered_image
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    image_u8 = np.uint8(
        np.clip(
            image * 255.0,
            0,
            255,
        )
    )

    return (
        image_u8.astype(np.float32)
        / 255.0
    )


def evaluate_scene(
    scene_name: str,
    dataset: ReconstructionDatasetAdapter,
    recon_root: Path,
    device: torch.device,
    lpips_model: torch.nn.Module,
    frame_stride: int = 1,
    out_root: Path | None = None,
    save_renderings: bool = False,
) -> dict:

    if frame_stride <= 0:
        raise ValueError(
            "frame_stride must be > 0"
        )

    frames = dataset.list_frames(
        scene_name,
        every_n_frame=1,
    )

    frames_by_id = {
        frame.frame_id: frame
        for frame in frames
    }

    if not frames_by_id:
        raise RuntimeError(
            f"No dataset frames found for {scene_name}"
        )

    gaussian_scene = load_gaussian_scene(
        recon_root=recon_root,
        scene_name=scene_name,
    )

    gt_pose_map = dataset.load_gt_poses(
        scene_name
    )

    background = torch.zeros(
        3,
        dtype=torch.float32,
        device=device,
    )

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    depth_l1_values: list[float] = []

    gt_poses: list[np.ndarray] = []
    est_poses: list[np.ndarray] = []

    frames_used = 0

    scene_out = None
    color_pred_dir = None
    depth_pred_dir = None

    if (
        save_renderings
        and out_root is not None
    ):
        scene_out = (
            Path(out_root)
            / scene_name
        )

        color_pred_dir = (
            scene_out
            / "color_pred"
        )

        depth_pred_dir = (
            scene_out
            / "depth_pred"
        )

        color_pred_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        depth_pred_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    for camera_idx, camera_param in enumerate(
        tqdm(
            gaussian_scene.cameras,
            desc=scene_name,
        )
    ):
        if camera_idx % frame_stride != 0:
            continue

        frame_id = camera_frame_id(
            camera_param,
            camera_idx,
        )

        frame = frames_by_id.get(
            frame_id
        )

        if frame is None:
            raise FileNotFoundError(
                f"Camera frame {frame_id} exists in cameras.json "
                f"but not in dataset adapter for scene "
                f"{scene_name}"
            )

        gt_rgb = dataset.load_rgb_gt (frame)

        target_width = int(camera_param["width"])
        target_height = int(camera_param["height"])

        gt_rgb = prepare_rgb_gt(
            gt_rgb,
            target_width=target_width,
            target_height=target_height,
        )

        camera = make_minicam (camera_param)

        with torch.inference_mode():
            render_result = render(
                camera,
                gaussian_scene.gaussians,
                background,
            )

        if "rendered_image" not in render_result:
            raise KeyError(
                "Renderer output does not contain "
                "'rendered_image'"
            )

        pred_rgb = _prepare_rendered_rgb(
            render_result["rendered_image"]
        )

        if gt_rgb.shape != pred_rgb.shape:
            raise ValueError(
                f"RGB resolution mismatch for "
                f"{scene_name}:{frame_id}: "
                f"GT={gt_rgb.shape}, "
                f"render={pred_rgb.shape}"
            )

        rgb_metrics = compute_rgb_metrics(
            gt=gt_rgb,
            pred=pred_rgb,
            lpips_model=lpips_model,
            device=device,
        )

        psnr_values.append(
            rgb_metrics.psnr
        )

        ssim_values.append(
            rgb_metrics.ssim
        )

        lpips_values.append(
            rgb_metrics.lpips
        )

        # ----------------------------------------------------------
        # Depth evaluation, when defined by the dataset adapter
        # ----------------------------------------------------------

        gt_depth_m = dataset.load_depth_gt_m(
            frame
        )

        if gt_depth_m is not None:
            if "rendered_depth" not in render_result:
                raise KeyError(
                    "Dataset requires depth evaluation, "
                    "but renderer output does not contain "
                    "'rendered_depth'"
                )

            pred_depth_m = (
                render_result["rendered_depth"]
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            pred_depth_m = (
                dataset.prepare_depth_prediction_m(
                    pred_depth_m
                )
            )

            if (
                gt_depth_m.shape
                != pred_depth_m.shape
            ):
                raise ValueError(
                    f"Depth resolution mismatch for "
                    f"{scene_name}:{frame_id}: "
                    f"GT={gt_depth_m.shape}, "
                    f"render={pred_depth_m.shape}"
                )

            depth_l1 = compute_depth_l1_cm(
                gt_depth_m,
                pred_depth_m,
            )

            depth_l1_values.append(
                depth_l1
            )

        # ----------------------------------------------------------
        # Trajectory evaluation, when available
        # ----------------------------------------------------------
        if gt_pose_map is not None:

            gt_pose = gt_pose_map.get(
                frame_id
            )

            if gt_pose is not None:
                gt_poses.append(
                    np.asarray(
                        gt_pose,
                        dtype=np.float64,
                    )
                )

                est_poses.append(
                    camera_pose_matrix(
                        camera_param
                    )
                )

        # ----------------------------------------------------------
        # Optional saved predictions
        # ----------------------------------------------------------

        if color_pred_dir is not None:
            pred_bgr = cv2.cvtColor(
                np.uint8(
                    np.clip(
                        pred_rgb * 255.0,
                        0,
                        255,
                    )
                ),
                cv2.COLOR_RGB2BGR,
            )

            cv2.imwrite(
                str(
                    color_pred_dir
                    / f"{frame_id:06d}.png"
                ),
                pred_bgr,
            )

        if (
            depth_pred_dir is not None
            and gt_depth_m is not None
        ):
            # Save depth in meters as .npy to avoid introducing
            # another dataset-specific encoding into generic code.
            np.save(
                depth_pred_dir
                / f"{frame_id:06d}.npy",
                pred_depth_m,
            )

        frames_used += 1

        del render_result

    if frames_used == 0:
        raise RuntimeError(
            f"No frames evaluated for {scene_name}"
        )

    trajectory = None

    if gt_pose_map is not None:
        trajectory_result = (
            compute_first_pose_aligned_translation_metrics(
                np.stack(gt_poses),
                np.stack(est_poses),
            )
        )

        trajectory = {
            "rmse_cm": _finite_or_none(
                trajectory_result.rmse_cm
            ),
            "mean_cm": _finite_or_none(
                trajectory_result.mean_cm
            ),
        }

    summary = {
        "scene": scene_name,
        "frames_used": frames_used,
        "rgb": {
            "psnr": _mean_or_none(
                psnr_values
            ),
            "ssim": _mean_or_none(
                ssim_values
            ),
            "lpips": _mean_or_none(
                lpips_values
            ),
        },
        "depth": (
            {
                "l1_cm": _mean_or_none(
                    depth_l1_values
                ),
            }
            if depth_l1_values
            else None
        ),
        "trajectory": trajectory,
    }

    if out_root is not None:
        scene_out = (
            Path(out_root)
            / scene_name
        )

        scene_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            scene_out
            / "metrics.json"
        ).open("w") as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

    return summary


def evaluate_dataset(
    scenes: Sequence[str],
    dataset: ReconstructionDatasetAdapter,
    recon_root: Path,
    device: torch.device,
    frame_stride: int = 1,
    out_root: Path | None = None,
    save_renderings: bool = False,
) -> dict:

    lpips_model = (
        LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
            normalize=True,
        )
        .to(device)
        .eval()
    )

    scene_summaries = []

    for scene_name in scenes:

        print()
        print("=" * 80)
        print(
            f"Evaluating reconstruction: "
            f"{scene_name}"
        )
        print("=" * 80)

        scene_summary = evaluate_scene(
            scene_name=scene_name,
            dataset=dataset,
            recon_root=recon_root,
            device=device,
            lpips_model=lpips_model,
            frame_stride=frame_stride,
            out_root=out_root,
            save_renderings=save_renderings,
        )

        scene_summaries.append(
            scene_summary
        )

    def collect(
        *keys: str,
    ) -> list[float]:

        values = []

        for scene in scene_summaries:
            value = scene

            for key in keys:
                if value is None:
                    break

                value = value.get(key)

            if (
                value is not None
                and np.isfinite(value)
            ):
                values.append(
                    float(value)
                )

        return values

    scene_mean = {
        "rgb": {
            "psnr": _mean_or_none(
                collect(
                    "rgb",
                    "psnr",
                )
            ),
            "ssim": _mean_or_none(
                collect(
                    "rgb",
                    "ssim",
                )
            ),
            "lpips": _mean_or_none(
                collect(
                    "rgb",
                    "lpips",
                )
            ),
        },
        "depth": {
            "l1_cm": _mean_or_none(
                collect(
                    "depth",
                    "l1_cm",
                )
            ),
        },
        "trajectory": {
            "rmse_cm": _mean_or_none(
                collect(
                    "trajectory",
                    "rmse_cm",
                )
            ),
            "mean_cm": _mean_or_none(
                collect(
                    "trajectory",
                    "mean_cm",
                )
            ),
        },
        "num_scenes": len(
            scene_summaries
        ),
    }

    final_summary = {
        "scene_mean": scene_mean,
        "scenes": scene_summaries,
    }

    if out_root is not None:
        out_root = Path(
            out_root
        )

        out_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            out_root
            / "summary.json"
        ).open("w") as f:
            json.dump(
                final_summary,
                f,
                indent=2,
            )

    return final_summary
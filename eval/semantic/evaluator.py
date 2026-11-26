from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .datasets.base import DatasetAdapter, FrameRecord
from .metrics import (
    MetricResult,
    compute_metrics,
    merge_confusions,
    update_confusion,
)
from .predictors.base import SemanticPredictor
from .protocols import EvaluationProtocol
from .taxonomies import Taxonomy

from ..common.gaussian_scene import (
    camera_frame_id,
    load_gaussian_scene,
    make_minicam,
)
from ..render import render

from .refinement import RefinementPipeline

import torch.nn.functional as F

def _finite_or_none(value: float):
    value = float(value)
    return value if np.isfinite(value) else None


def metric_result_to_dict(
    result: MetricResult,
    class_names: Sequence[str],
) -> dict:
    per_class = {}

    for class_id, class_name in enumerate(class_names):
        per_class[class_name] = {
            "id": class_id,
            "IoU": _finite_or_none(result.iou[class_id]),
            "accuracy": _finite_or_none(result.accuracy[class_id]),
            "gt_support": int(result.gt_support[class_id]),
            "pred_support": int(result.pred_support[class_id]),
            "true_positive": int(result.true_positive[class_id]),
        }

    split_metrics = {}

    for split_name, metrics in result.split_metrics.items():
        split_metrics[split_name] = {
            key: (
                _finite_or_none(value)
                if isinstance(value, (float, np.floating))
                else int(value)
                if isinstance(value, (int, np.integer))
                else value
            )
            for key, value in metrics.items()
        }

    return {
        "pixel_accuracy": _finite_or_none(result.pixel_accuracy),
        "splits": split_metrics,
        "per_class": per_class,
    }

def _make_semantic_palette(
    num_classes: int,
) -> np.ndarray:
    """
    Deterministic high-contrast RGB palette.

    Colors are only for visualization; they have no effect
    on evaluation or semantic class IDs.
    """
    palette = np.zeros(
        (num_classes, 3),
        dtype=np.uint8,
    )

    for class_id in range(num_classes):
        value = class_id + 1
        bit = 0

        while value:
            palette[class_id, 0] |= (
                (value >> 0) & 1
            ) << (7 - bit)

            palette[class_id, 1] |= (
                (value >> 1) & 1
            ) << (7 - bit)

            palette[class_id, 2] |= (
                (value >> 2) & 1
            ) << (7 - bit)

            value >>= 3
            bit += 1

    return palette


def _colorize_semantic_mask(
    labels: np.ndarray,
    palette: np.ndarray,
    ignore_index: int,
) -> np.ndarray:
    """
    Convert dense semantic IDs [H,W] into RGB visualization.
    Ignored pixels remain black.
    """
    labels = np.asarray(labels)

    vis = np.zeros(
        (*labels.shape, 3),
        dtype=np.uint8,
    )

    valid = (
        (labels != ignore_index)
        & (labels >= 0)
        & (labels < len(palette))
    )

    vis[valid] = palette[
        labels[valid]
    ]

    return vis



def refine_label_image(
    labels: np.ndarray,
    min_area: int = 100,
    max_fill_area: int = 5000,
    ignore_index: int = 255,
) -> np.ndarray:
    """Remove small per-class connected components from a predicted ID mask.

    Adapted from the original LEG-SLAM connected-component cleanup.
    Every dense class ID, including 0, is valid. Temporarily removed pixels
    use -1; unresolved regions are restored to their original class.

    Only the *prediction* is refined. Ground truth must not be altered.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected [H,W] labels, got {labels.shape}")
    if min_area < 1 or max_fill_area < 1:
        raise ValueError("min_area and max_fill_area must be positive")

    original = labels.astype(np.int32, copy=False)
    refined = original.copy()
    removed = np.zeros(original.shape, dtype=bool)

    # Identify small islands separately for every semantic class.
    for class_id in np.unique(original):
        if int(class_id) == ignore_index:
            continue
        binary = (original == class_id).astype(np.uint8)
        num_components, components, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for idx in range(1, num_components):
            if stats[idx, cv2.CC_STAT_AREA] < min_area:
                removed[components == idx] = True

    if not np.any(removed):
        return labels.copy()

    refined[removed] = -1
    kernel = np.ones((3, 3), dtype=np.uint8)
    num_gaps, gaps = cv2.connectedComponents(
        removed.astype(np.uint8), connectivity=8
    )

    # Replace each sufficiently small removed region with its predominant
    # surviving boundary class, without ever interpreting class 0 as empty.
    for idx in range(1, num_gaps):
        region = gaps == idx
        if int(region.sum()) > max_fill_area:
            continue
        dilated = cv2.dilate(region.astype(np.uint8), kernel, iterations=1).astype(bool)
        border = dilated & ~region
        neighbors = refined[border]
        neighbors = neighbors[(neighbors >= 0) & (neighbors != ignore_index)]
        if neighbors.size:
            new_class = int(np.bincount(neighbors).argmax())
            refined[region] = new_class

    # Avoid inventing a class for isolated or oversized unresolved regions.
    unresolved = refined == -1
    refined[unresolved] = original[unresolved]
    return refined.astype(labels.dtype, copy=False)

def evaluate_scene(
    scene_name: str,
    dataset: DatasetAdapter,
    taxonomy: Taxonomy,
    semantic_predictor: SemanticPredictor,
    recon_root: Path,
    device: torch.device,
    frame_stride: int = 1,
    splits: Optional[Dict[str, np.ndarray]] = None,
    out_root: Optional[Path] = None,
    save_masks: bool = False,
    refinement: Optional[RefinementPipeline] = None,
    compute_cross_entropy: bool = False,
    component_refinement: bool = False,
) -> tuple[np.ndarray, dict]:

    num_classes = len(taxonomy.class_names)

    # Standard benchmark behaviour:
    # every taxonomy class is a candidate.
    candidate_ids = list(range(num_classes))

    if save_masks and out_root is None:
        raise ValueError(
            "save_masks=True requires out_root"
        )

    palette = _make_semantic_palette(
        num_classes
    )

    pred_ids_dir = None
    pred_vis_dir = None
    gt_ids_dir = None
    gt_vis_dir = None
    aux_root = None

    if save_masks:
        scene_out = (
            Path(out_root)
            / scene_name
        )

        pred_ids_dir = (
            scene_out / "pred_ids"
        )
        pred_vis_dir = (
            scene_out / "pred_vis"
        )
        gt_ids_dir = (
            scene_out / "gt_ids"
        )
        gt_vis_dir = (
            scene_out / "gt_vis"
        )
        aux_root = scene_out / "aux"

        for directory in (
            pred_ids_dir,
            pred_vis_dir,
            gt_ids_dir,
            gt_vis_dir,
            aux_root,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        with (
            scene_out / "classes.json"
        ).open("w") as f:
            json.dump(
                {
                    str(class_id): class_name
                    for class_id, class_name
                    in enumerate(
                        taxonomy.class_names
                    )
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Dataset frames
    # ------------------------------------------------------------------

    frames = dataset.list_frames(
        scene_name,
        every_n_frame=1,
    )

    frames_by_id: Dict[int, FrameRecord] = {
        frame.frame_id: frame
        for frame in frames
    }

    if not frames_by_id:
        raise RuntimeError(
            f"No dataset frames found for scene {scene_name}"
        )

    # ------------------------------------------------------------------
    # Gaussian reconstruction
    # ------------------------------------------------------------------

    gaussian_scene = load_gaussian_scene(
        recon_root=recon_root,
        scene_name=scene_name,
    )

    background = torch.zeros(
        3,
        dtype=torch.float32,
        device=device,
    )

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.uint64,
    )

    frames_used = 0
    valid_pixels = 0
    ignored_pixels = 0
    loss_sum = 0.0

    # ------------------------------------------------------------------
    # Camera loop
    # ------------------------------------------------------------------

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

        frame = frames_by_id.get(frame_id)

        if frame is None:
            raise FileNotFoundError(
                f"Camera frame {frame_id} exists in cameras.json "
                f"but not in dataset adapter for scene {scene_name}"
            )

        # --------------------------------------------------------------
        # GT
        # --------------------------------------------------------------

        gt = dataset.load_semantic_gt(frame)

        # --------------------------------------------------------------
        # Render semantic Gaussian features
        # --------------------------------------------------------------

        camera = make_minicam(camera_param)

        with torch.inference_mode():

            render_result = render(
                camera,
                gaussian_scene.gaussians,
                background,
            )

            if "rendered_lf" not in render_result:
                raise KeyError(
                    "Renderer output does not contain 'rendered_lf'"
                )

            rendered_lf = render_result["rendered_lf"].detach()

            prediction = semantic_predictor.predict(
                rendered_lf,
                candidate_ids=candidate_ids,
            )

        if refinement is None:

            pred_tensor = prediction.labels

        else:

            if prediction.class_scores is None:
                raise RuntimeError(
                    "Semantic refinement requires full class scores"
                )

            class_scores = prediction.class_scores

            depth_gt = dataset.load_depth_gt(
                frame
            )

            score_hw = tuple(
                class_scores.shape[-2:]
            )

            if depth_gt.shape != score_hw:
                raise RuntimeError(
                    "GT depth / semantic-score resolution mismatch: "
                    f"depth={depth_gt.shape}, "
                    f"scores={score_hw}, "
                    f"scene={scene_name}, "
                    f"frame={frame_id}"
                )

            depth_tensor = torch.as_tensor(
                depth_gt,
                dtype=class_scores.dtype,
                device=class_scores.device,
            )

            refined_scores = refinement.refine(
                class_scores=class_scores,
                resources={
                    "depth": depth_tensor,
                },
            )

            pred_tensor = (
                semantic_predictor.labels_from_scores(
                    refined_scores,
                    candidate_ids=candidate_ids,
                )
            )

        pred = (
            pred_tensor
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )

        # Optional class-ID post-processing: AFTER prediction/score refinement,
        # BEFORE masks are saved and confusion/IoU are computed.
        # Never post-process the GT.
        if component_refinement:
            pred = refine_label_image(
                pred,
                min_area=100,
                max_fill_area=5000,
                ignore_index=taxonomy.ignore_index,
            )

        # --------------------------------------------------------------
        # Match GT resolution to rendered resolution if needed
        # --------------------------------------------------------------

        if gt.shape != pred.shape:
            gt = cv2.resize(
                gt,
                (pred.shape[1], pred.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if compute_cross_entropy:

            class_scores = (
                prediction.class_scores
            )

            if class_scores is None:
                raise RuntimeError(
                    "Cross-entropy evaluation "
                    "requires prediction.class_scores"
                )

            if class_scores.ndim != 3:
                raise RuntimeError(
                    "Expected class_scores "
                    "[C,H,W], got "
                    f"{tuple(class_scores.shape)}"
                )

            if (
                class_scores.shape[0]
                != num_classes
            ):
                raise RuntimeError(
                    "Cross-entropy requires all "
                    f"{num_classes} class logits, "
                    f"got "
                    f"{class_scores.shape[0]}"
                )

            if (
                tuple(
                    class_scores.shape[-2:]
                )
                != tuple(gt.shape)
            ):
                raise RuntimeError(
                    "Probe-logit / resized-GT "
                    "shape mismatch: "
                    f"logits="
                    f"{tuple(class_scores.shape)}, "
                    f"GT={gt.shape}"
                )

            gt_tensor = torch.as_tensor(
                gt,
                dtype=torch.long,
                device=class_scores.device,
            )

            frame_loss_sum = (
                F.cross_entropy(
                    class_scores.unsqueeze(0),
                    gt_tensor.unsqueeze(0),
                    ignore_index=(
                        taxonomy.ignore_index
                    ),
                    reduction="sum",
                )
            )

            loss_sum += float(
                frame_loss_sum.item()
            )

        if save_masks:

            frame_name = (
                f"{frame_id:06d}.png"
            )

            # ----------------------------------------------------------
            # Dense class-index masks
            #
            # pred:
            #   ScanNet -> values 0..19
            #   Replica -> values 0..50
            #
            # GT ignore pixels are stored as 255.
            # ----------------------------------------------------------

            pred_ids = pred.astype(
                np.uint8,
                copy=False,
            )

            gt_ids = gt.copy()

            gt_ids[
                gt_ids == taxonomy.ignore_index
            ] = 255

            gt_ids = gt_ids.astype(
                np.uint8,
                copy=False,
            )

            cv2.imwrite(
                str(
                    pred_ids_dir
                    / frame_name
                ),
                pred_ids,
            )

            cv2.imwrite(
                str(
                    gt_ids_dir
                    / frame_name
                ),
                gt_ids,
            )

            # ----------------------------------------------------------
            # Colored visualization masks
            # ----------------------------------------------------------

            pred_vis = _colorize_semantic_mask(
                pred,
                palette,
                taxonomy.ignore_index,
            )

            gt_vis = _colorize_semantic_mask(
                gt,
                palette,
                taxonomy.ignore_index,
            )

            cv2.imwrite(
                str(
                    pred_vis_dir
                    / frame_name
                ),
                cv2.cvtColor(
                    pred_vis,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            cv2.imwrite(
                str(
                    gt_vis_dir
                    / frame_name
                ),
                cv2.cvtColor(
                    gt_vis,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            # ----------------------------------------------------------
            # Predictor-specific diagnostic maps
            # ----------------------------------------------------------

            if prediction.aux:
                for aux_name, aux_tensor in prediction.aux.items():
                    aux_array = (
                        aux_tensor
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    # Query coverage is image-independent. Save it once
                    # for the scene rather than duplicating it per frame.
                    if aux_array.ndim == 1:
                        aux_path = (
                            aux_root
                            / f"{aux_name}.npy"
                        )
                        if not aux_path.exists():
                            np.save(aux_path, aux_array)
                        continue

                    if aux_array.ndim != 2:
                        continue

                    aux_dir = aux_root / aux_name
                    aux_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    stem = f"{frame_id:06d}"
                    np.save(
                        aux_dir / f"{stem}.npy",
                        aux_array,
                    )

                    if aux_name in {
                        "closed_head_labels",
                        "closed_proto_labels",
                    }:
                        aux_vis = aux_array.astype(
                            np.uint8,
                            copy=False,
                        )
                    elif aux_name == "head_proto_agreement":
                        aux_vis = (
                            aux_array.astype(np.uint8)
                            * 255
                        )
                    else:
                        aux_float = aux_array.astype(
                            np.float32,
                            copy=False,
                        )
                        if aux_name == "prototype_knownness":
                            # Cosine knownness has range [-1,1].
                            aux_float = (aux_float + 1.0) * 0.5

                        aux_float = np.clip(
                            aux_float,
                            0.0,
                            1.0,
                        )
                        aux_vis = np.round(
                            aux_float * 255.0
                        ).astype(np.uint8)

                    cv2.imwrite(
                        str(aux_dir / f"{stem}.png"),
                        aux_vis,
                    )

        # --------------------------------------------------------------
        # Confusion
        # --------------------------------------------------------------

        valid = update_confusion(
            confusion=confusion,
            gt=gt,
            pred=pred,
            num_classes=num_classes,
            ignore_index=taxonomy.ignore_index,
        )

        frames_used += 1
        valid_pixels += valid
        ignored_pixels += int(
            np.sum(gt == taxonomy.ignore_index)
        )

        del render_result
        del rendered_lf
        del prediction
        if refinement is not None:
            del depth_tensor
            del refined_scores

    if frames_used == 0:
        raise RuntimeError(
            f"No frames evaluated for {scene_name}"
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    metrics = compute_metrics(
        confusion,
        splits=splits,
    )

    summary = {
        "scene": scene_name,
        "frames_used": frames_used,
        "valid_pixels": valid_pixels,
        "ignored_pixels": ignored_pixels,

        "cross_entropy_sum": (
            loss_sum if compute_cross_entropy else None),

        "cross_entropy": (
            loss_sum / valid_pixels
            if (compute_cross_entropy and valid_pixels > 0) else None),

        "metrics": metric_result_to_dict(
            metrics,
            taxonomy.class_names,
        ),
    }

    # ------------------------------------------------------------------
    # Save scene result
    # ------------------------------------------------------------------

    if out_root is not None:

        scene_out = Path(out_root) / scene_name
        scene_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            scene_out / "confusion.npy",
            confusion,
        )

        with (
            scene_out / "metrics.json"
        ).open("w") as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

    return confusion, summary


def evaluate_protocol(
    protocol: EvaluationProtocol,
    dataset: DatasetAdapter,
    taxonomy: Taxonomy,
    semantic_predictor: SemanticPredictor,
    recon_root: Path,
    device: torch.device,
    splits: Optional[Dict[str, np.ndarray]] = None,
    out_root: Optional[Path] = None,
    save_masks: bool = False,
    refinement: Optional[RefinementPipeline] = None,
    compute_cross_entropy: bool = False,
    component_refinement: bool = False,
) -> dict:

    scene_confusions = []
    scene_summaries = []

    for scene_name in protocol.scenes:

        print()
        print("=" * 80)
        print(f"Evaluating: {scene_name}")
        print("=" * 80)

        confusion, summary = evaluate_scene(
            scene_name=scene_name,
            dataset=dataset,
            taxonomy=taxonomy,
            semantic_predictor=semantic_predictor,
            recon_root=recon_root,
            device=device,
            frame_stride=protocol.frame_stride,
            splits=splits,
            out_root=out_root,
            save_masks=save_masks,
            refinement=refinement,
            compute_cross_entropy=(compute_cross_entropy),
            component_refinement=component_refinement,
        )

        scene_confusions.append(confusion)
        scene_summaries.append(summary)

    # ------------------------------------------------------------------
    # Pooled/global result
    # ------------------------------------------------------------------

    pooled_confusion = merge_confusions(
        scene_confusions
    )

    pooled_metrics = compute_metrics(
        pooled_confusion,
        splits=splits,
    )

    pooled_summary = metric_result_to_dict(
        pooled_metrics,
        taxonomy.class_names,
    )

    if compute_cross_entropy:

        pooled_loss_sum = sum(
            float(
                scene[
                    "cross_entropy_sum"
                ]
            )
            for scene in scene_summaries
        )

        pooled_valid_pixels = sum(
            int(
                scene[
                    "valid_pixels"
                ]
            )
            for scene in scene_summaries
        )

        pooled_summary[
            "cross_entropy"
        ] = (
            pooled_loss_sum
            / pooled_valid_pixels
            if pooled_valid_pixels > 0
            else None
        )

        # Existing metric code calls this FmIoU.
        # It is mathematically the same fwIoU used
        # by the direct probe evaluator.
        pooled_summary[
            "fwIoU"
        ] = pooled_summary[
            "splits"
        ][
            "all"
        ][
            "FmIoU"
        ]

    # ------------------------------------------------------------------
    # Scene-mean result
    #
    # Deliberately kept separate from pooled/global confusion.
    # ------------------------------------------------------------------

    scene_mious = []
    scene_maccs = []
    scene_fmious = []
    scene_pixel_accs = []
    scene_cross_entropies = []

    for scene in scene_summaries:

        metrics = scene["metrics"]

        all_metrics = metrics["splits"]["all"]

        if all_metrics["mIoU"] is not None:
            scene_mious.append(all_metrics["mIoU"])

        if all_metrics["mAcc"] is not None:
            scene_maccs.append(all_metrics["mAcc"])

        if all_metrics["FmIoU"] is not None:
            scene_fmious.append(all_metrics["FmIoU"])

        if metrics["pixel_accuracy"] is not None:
            scene_pixel_accs.append(metrics["pixel_accuracy"])

        if scene["cross_entropy"] is not None:
            scene_cross_entropies.append(scene["cross_entropy"])

    scene_mean = {
        "mIoU": (
            float(np.mean(scene_mious))
            if scene_mious
            else None
        ),
        "mAcc": (
            float(np.mean(scene_maccs))
            if scene_maccs
            else None
        ),
        "FmIoU": (
            float(np.mean(scene_fmious))
            if scene_fmious
            else None
        ),
        "pixel_accuracy": (
            float(np.mean(scene_pixel_accs))
            if scene_pixel_accs
            else None
        ),
        "num_scenes": len(scene_summaries),
    }

    final_summary = {
        "protocol": protocol.name,
        "dataset": protocol.dataset,
        "taxonomy": taxonomy.name,
        "predictor": semantic_predictor.metadata(),

        # Sum all pixels first, then compute metrics.
        "pooled": pooled_summary,

        # Compute each scene independently, then average.
        "scene_mean": scene_mean,

        "scenes": scene_summaries,
    }

    if out_root is not None:

        out_root = Path(out_root)
        out_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            out_root / "confusion_pooled.npy",
            pooled_confusion,
        )

        with (
            out_root / "summary.json"
        ).open("w") as f:
            json.dump(
                final_summary,
                f,
                indent=2,
            )

    return final_summary
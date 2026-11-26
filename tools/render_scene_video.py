from __future__ import annotations

"""
Create a four-panel video from a finished LEG-SLAM Gaussian reconstruction.

Panels:
    Real RGB | Gaussian-rendered RGB
    RGB visualization of 3 scene-wide PCA axes | Replica51 dual semantic mask

The PCA is fit once to *rendered compressed feature vectors* sampled from the
scene. The same basis and color limits are used for every frame.

Run from the LEG-SLAM repository root:
    python -m tools.render_scene_video --help

Default: render ALL frames with tracked poses from CameraTrajectory_TUM.txt.
The saved cameras.json supplies intrinsics and aligns the exported TUM poses
with the Gaussian map frame. --camera-mode keyframes reproduces old behavior.

Semantic predictions are computed directly from the rendered Gaussian features
using the existing RADIO dual predictor and Replica51 taxonomy. The optional
connected-component cleanup is identical to eval.semantic.evaluator.
No GT semantic annotation or semantic evaluation is needed.
"""

import argparse
from pathlib import Path

from scipy.spatial.transform import Rotation

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval.common.gaussian_scene import load_gaussian_scene, make_minicam
from eval.render import render

from eval.semantic.taxonomies import REPLICA51_TAXONOMY
from eval.semantic.predictors.factory import load_feature_model_registry
from eval.semantic.predictors.radio_v4 import RadioDualSemanticPredictor
from eval.semantic.evaluator import (
    _make_semantic_palette,
    _colorize_semantic_mask,
    refine_label_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a LEG-SLAM scene to a 2x2 MP4.")
    p.add_argument("--result", type=Path, required=True,
                   help="Scene output directory containing experiment/ply/")
    p.add_argument("--sequence", type=Path, required=True,
                   help="Original dataset sequence directory")
    p.add_argument("--dataset", choices=("mmpr", "scannet", "replica"),
                   required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Output MP4 filename")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--panel-width", type=int, default=640)
    p.add_argument("--camera-mode", choices=("full", "keyframes"), default="full",
                   help="full = all tracked TUM poses; keyframes = saved cameras.json")
    p.add_argument("--trajectory", type=Path, default=None,
                   help="Default: <result>/CameraTrajectory_TUM.txt")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Use every Nth tracked frame / saved keyframe")
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 means all available views after striding")
    p.add_argument("--pca-sample-views", type=int, default=32)
    p.add_argument("--pca-pixels-per-view", type=int, default=2048)
    # Same pretrained RADIO dual inference as eval semantic --protocol replica51.
    p.add_argument("--feature-model", default="radio_v4_so400m")
    p.add_argument("--feature-registry", type=Path,
                   default=Path("cfg/feature_models.json"))
    p.add_argument("--probe-checkpoint", type=Path, required=True)
    p.add_argument("--prototype-file", type=Path, required=True)
    p.add_argument("--fusion-beta", type=float, default=1.0)
    p.add_argument("--fusion-tau-ov", type=float, default=0.5)
    p.add_argument("--fusion-tau-cs", type=float, default=1.0)
    p.add_argument("--semantic-z-threshold", type=float, default=2.2,
                   help="Query-relative SigLIP text similarity z-score gate")
    p.add_argument("--radio-chunk-size", type=int, default=32768)
    p.add_argument("--semantic-refinement", choices=("components", "none"),
                   default="components",
                   help="Replica51 semantic postprocessing; default components")
    p.add_argument("--save-semantic-masks", type=Path, default=None,
                   help="Optional output directory for raw class-ID and colored masks")
    return p.parse_args()


def get_original_rgb(cam: dict, sequence: Path, dataset: str) -> Path:
    """Resolve original RGB from cameras.json's img_name, never guess index."""
    original = str(cam.get("img_name", "") or "")
    filename = Path(original).name if original else ""
    candidates: list[Path] = []

    if original:
        candidates.append(Path(original))
    if filename:
        if dataset == "mmpr":
            candidates.append(sequence / "cam0" / "data" / filename)
        elif dataset == "scannet":
            candidates.append(sequence / "color" / filename)
        elif dataset == "replica":
            candidates.append(sequence / "results" / filename)

    for path in candidates:
        if path.is_file():
            return path

    # Some cameras.json files store the frame ID but not an image filename.
    # For ScanNet, ID-based lookup is unambiguous if these files exist.
    if dataset == "scannet" and "id" in cam:
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = sequence / "color" / (str(cam["id"]) + ext)
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "Cannot match saved camera to an RGB frame.\n"
        f"Camera img_name={original!r}, id={cam.get('id')!r}\n"
        f"Sequence={sequence}\n"
        f"Tried: {[str(c) for c in candidates]}\n"
        "Check img_name in experiment/ply/cameras.json."
    )


def list_input_rgb(sequence: Path, dataset: str) -> list[Path]:
    """Mirror the C++ runner's RGB sequence order (frame i has tframe=i)."""
    if dataset == "mmpr":
        color_dir = sequence / "cam0" / "data"
        depth_dir = sequence / "depth" / "data"
        if not depth_dir.is_dir():
            raise FileNotFoundError(depth_dir)
        depth_names = {p.stem for p in depth_dir.glob("*.npy")}
        rgb = [p for p in color_dir.glob("*.png") if p.stem in depth_names]
        rgb.sort(key=lambda p: int(p.stem))
        # The MMPR C++ reader processes only exact timestamp-matched pairs.
    elif dataset == "scannet":
        color_dir = sequence / "color"
        rgb = [p for p in color_dir.iterdir()
               if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        rgb.sort(key=lambda p: int(p.stem))
    elif dataset == "replica":
        color_dir = sequence / "results"
        rgb = [p for p in color_dir.iterdir()
               if p.is_file() and p.name.startswith("frame")]
        rgb.sort(key=lambda p: p.name)  # matches LoadImagesReplica C++
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not rgb:
        raise RuntimeError(f"No input RGB images found for {dataset} in {sequence}")
    if len({p.name for p in rgb}) != len(rgb):
        raise RuntimeError("Duplicate input RGB filenames")
    return rgb


def load_tum_trajectory(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read timestamp tx ty tz qx qy qz qw. Here timestamp = C++ frame index."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing full tracked trajectory: {path}")
    trajectory = {}
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 TUM fields")
            values = np.asarray([float(v) for v in fields], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{line_no}: nonfinite trajectory value")
            time = values[0]
            frame_index = int(round(time))
            if abs(time - frame_index) > 1e-3:
                raise ValueError(
                    f"{path}:{line_no}: expected integer-indexed timestamps; got {time}. "
                    "This script assumes the C++ tracking call uses tframe=ni."
                )
            if frame_index in trajectory:
                raise ValueError(f"Repeated trajectory frame index: {frame_index}")
            quaternion = values[4:8]  # x,y,z,w, standard TUM convention
            if np.linalg.norm(quaternion) < 1e-8:
                raise ValueError(f"{path}:{line_no}: invalid zero quaternion")
            trajectory[frame_index] = (
                Rotation.from_quat(quaternion).as_matrix(), values[1:4]
            )
    if not trajectory:
        raise RuntimeError(f"Empty TUM trajectory: {path}")
    return trajectory


def align_tum_to_gaussian(
    trajectory: dict[int, tuple[np.ndarray, np.ndarray]],
    saved_cameras: list[dict],
    ordered_rgb: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Determine T_gaussian_from_tum using common saved keyframe poses.

    Both are T_wc camera-to-world poses. ORB-SLAM3's TUM exporter may
    normalize the trajectory's world frame to its first keyframe, while
    GaussianMapper exports poses in its own map frame.
    """
    filename_to_index = {p.name: i for i, p in enumerate(ordered_rgb)}
    correspondences = []
    for cam in saved_cameras:
        filename = Path(str(cam.get("img_name", "") or "")).name
        index = filename_to_index.get(filename)
        if index is None:
            # Fallback for cameras.json versions without img_name.
            try:
                index = int(cam["id"])
            except (KeyError, ValueError, TypeError):
                continue
        if index not in trajectory:
            continue
        r_tum, t_tum = trajectory[index]
        r_json = np.asarray(cam["rotation"], dtype=np.float64)
        t_json = np.asarray(cam["position"], dtype=np.float64)
        if r_json.shape != (3, 3) or t_json.shape != (3,):
            raise ValueError(f"Invalid camera pose in cameras.json for frame {index}")
        correspondences.append((r_json, t_json, r_tum, t_tum))

    if not correspondences:
        raise RuntimeError(
            "No shared frames between cameras.json and CameraTrajectory_TUM.txt. "
            "Check that both files belong to this reconstruction, and that "
            "the C++ runner used tframe=ni."
        )

    align_rotations = np.stack([
        r_json @ r_tum.T
        for r_json, _, r_tum, _ in correspondences
    ])
    u, _, vt = np.linalg.svd(align_rotations.mean(axis=0))
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    r_align = u @ correction @ vt
    offsets = np.stack([
        t_json - r_align @ t_tum
        for _, t_json, _, t_tum in correspondences
    ])
    t_align = np.median(offsets, axis=0)

    position_errors = np.asarray([
        np.linalg.norm(r_align @ t_tum + t_align - t_json)
        for _, t_json, _, t_tum in correspondences
    ])
    angle_errors = np.asarray([
        np.rad2deg(Rotation.from_matrix(r_json.T @ (r_align @ r_tum)).magnitude())
        for r_json, _, r_tum, _ in correspondences
    ])
    pos_error = float(np.median(position_errors))
    rot_error = float(np.median(angle_errors))
    print(
        f"[Poses] Matched {len(correspondences)} keyframes for map alignment; "
        f"median disagreement: {pos_error:.3f} m, {rot_error:.2f} deg"
    )
    if pos_error > 0.5 or rot_error > 20:
        raise RuntimeError(
            "TUM trajectory is not sufficiently consistent with saved Gaussian "
            "camera poses. Verify frame indexing / trajectory world frame before rendering."
        )
    if pos_error > 0.05 or rot_error > 3:
        print("[Poses] WARNING: Nontrivial trajectory-vs-map pose disagreement.")
    return r_align, t_align


def build_full_views(
    scene, sequence: Path, dataset: str, result: Path,
    trajectory_path: Path | None,
) -> list[tuple[dict, Path]]:
    """Use only frames having a saved *tracked* pose, no invented interpolation."""
    ordered_rgb = list_input_rgb(sequence, dataset)
    trajectory = load_tum_trajectory(
        trajectory_path if trajectory_path is not None
        else result / "CameraTrajectory_TUM.txt"
    )
    r_align, t_align = align_tum_to_gaussian(
        trajectory, scene.cameras, ordered_rgb
    )

    template = scene.cameras[0]  # same intrinsics as all scene frames
    views = []
    for frame_id in sorted(trajectory):
        if not (0 <= frame_id < len(ordered_rgb)):
            continue
        r_tum, t_tum = trajectory[frame_id]
        cam = dict(template)
        cam.update({
            "id": frame_id,
            "img_name": str(ordered_rgb[frame_id]),
            "rotation": (r_align @ r_tum).tolist(),
            "position": (r_align @ t_tum + t_align).tolist(),
        })
        views.append((cam, ordered_rgb[frame_id]))
    if not views:
        raise RuntimeError("No trajectory poses could be matched to input RGB frames")
    print(
        f"[Views] Full tracked poses: {len(trajectory)}; source RGB frames: "
        f"{len(ordered_rgb)}; renderable matched frames: {len(views)}"
    )
    if len(views) < len(ordered_rgb):
        print(
            f"[Views] NOTE: {len(ordered_rgb) - len(views)} input frames have no "
            "matching saved tracked pose and will not be rendered."
        )
    return views


def render_one(camera_params: dict, gaussian_scene, background: torch.Tensor):
    camera = make_minicam(camera_params)
    with torch.no_grad():
        outputs = render(camera, gaussian_scene.gaussians, background)
    required = ("rendered_image", "rendered_lf", "rendered_depth")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise KeyError(f"Renderer is missing outputs: {missing}")
    return outputs


def sample_features(
    lf: torch.Tensor, pixels_per_view: int, rng: np.random.Generator
) -> np.ndarray:
    if lf.ndim != 3:
        raise ValueError(f"Expected rendered_lf [K,H,W], got {lf.shape}")
    k, h, w = lf.shape
    n = min(pixels_per_view, h * w)
    indices = rng.choice(h * w, size=n, replace=False)
    index_gpu = torch.as_tensor(indices, device=lf.device, dtype=torch.long)
    # Gather only sampled feature vectors before moving to CPU.
    sampled = lf.detach().reshape(k, h * w).index_select(1, index_gpu)
    sampled = sampled.T.float().cpu().numpy()
    valid = np.isfinite(sampled).all(axis=1)
    valid &= np.linalg.norm(sampled, axis=1) > 1e-7
    return sampled[valid]


def fit_pca_from_rendered_views(
    cameras: list[dict], scene, background: torch.Tensor,
    views_to_sample: int, pixels_per_view: int
):
    rng = np.random.default_rng(42)
    sample_indices = np.unique(np.linspace(
        0, len(cameras) - 1,
        min(len(cameras), views_to_sample),
        dtype=int
    ))
    chunks = []
    for i in tqdm(sample_indices, desc="PCA: sampling rendered views"):
        outputs = render_one(cameras[int(i)], scene, background)
        chunks.append(sample_features(outputs["rendered_lf"], pixels_per_view, rng))
        del outputs
    chunks = [x for x in chunks if len(x)]
    if not chunks:
        raise RuntimeError("No valid Gaussian feature vectors were sampled.")

    data = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
    if data.shape[0] < 4 or data.shape[1] < 3:
        raise RuntimeError(
            f"Not enough pixels/features to fit 3D PCA: {data.shape}"
        )
    mean = data.mean(axis=0)
    centered = data - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T

    # Fix arbitrary PCA axis sign so re-runs are more reproducible.
    for c in range(3):
        dominant = int(np.argmax(np.abs(basis[:, c])))
        if basis[dominant, c] < 0:
            basis[:, c] *= -1

    projected = centered @ basis
    lower = np.percentile(projected, 1, axis=0)
    upper = np.percentile(projected, 99, axis=0)
    upper = np.maximum(upper, lower + 1e-6)
    explained = (
        singular_values[:3] ** 2 /
        np.maximum(np.sum(singular_values ** 2), 1e-15)
    )
    print(f"[PCA] Fit on {len(data)} pixels, D={data.shape[1]}")
    print(f"[PCA] Top-3 explained fractions: {explained.round(4).tolist()}")
    return (mean.astype(np.float32), basis.astype(np.float32),
            lower.astype(np.float32), upper.astype(np.float32), explained)


def feature_visualization(
    rendered_lf: torch.Tensor, target_size: tuple[int, int],
    mean: np.ndarray, basis: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Return a panel-size BGR image; fixed PCA basis and 1-99% limits."""
    width, height = target_size
    low_res = F.interpolate(
        rendered_lf.detach().unsqueeze(0).float(),
        size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(0)
    k, h, w = low_res.shape
    vectors = low_res.permute(1, 2, 0).reshape(-1, k).cpu().numpy()
    invalid = ~np.isfinite(vectors).all(axis=1)
    invalid |= (np.linalg.norm(np.nan_to_num(vectors), axis=1) <= 1e-7)
    projected = (np.nan_to_num(vectors) - mean) @ basis
    rgb = np.clip(
        (projected - lower) / (upper - lower),
        0.0, 1.0
    ).reshape(h, w, 3)
    rgb = np.uint8(np.round(rgb * 255.0))
    rgb.reshape(-1, 3)[invalid] = 0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def rgb_from_render(
    tensor: torch.Tensor, target_size: tuple[int, int]
) -> np.ndarray:
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected rendered_image [3,H,W], got {tensor.shape}")
    rgb = tensor.detach().float().permute(1, 2, 0).cpu().numpy()
    rgb = (np.clip(np.nan_to_num(rgb), 0.0, 1.0) * 255).round().astype(np.uint8)
    return cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        target_size, interpolation=cv2.INTER_AREA
    )


def put_title(img: np.ndarray, title: str, frame_text: str = "") -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 32), (25, 25, 25), -1)
    cv2.putText(img, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.59, (255, 255, 255), 1, cv2.LINE_AA)
    if frame_text:
        tw = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0]
        cv2.putText(img, frame_text, (max(4, img.shape[1] - tw - 12), 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 215, 215), 1, cv2.LINE_AA)
    return img


def build_replica51_dual_predictor(args) -> RadioDualSemanticPredictor:
    """Construct the SAME dual RADIO model class used by semantic evaluation.

    Bypass only factory routing, not the model or its decision rule, so that
    this video's semantic z-threshold can follow the current dual predictor.
    """
    registry = args.feature_registry.resolve()
    models = load_feature_model_registry(registry)
    if args.feature_model not in models:
        raise ValueError(
            f"Missing {args.feature_model!r} in {registry}; "
            f"available: {sorted(models)}"
        )

    return RadioDualSemanticPredictor(
        class_names=REPLICA51_TAXONOMY.class_names,
        feature_model_id=args.feature_model,
        model_cfg=models[args.feature_model],
        chunk_size=args.radio_chunk_size,
        device="cuda",
        repo_root=registry.parent.parent,
        probe_checkpoint=args.probe_checkpoint.resolve(),
        prototype_path=args.prototype_file.resolve(),
        fusion_beta=args.fusion_beta,
        tau_ov=args.fusion_tau_ov,
        tau_cs=args.fusion_tau_cs,
        use_query_coverage=True,
        semantic_z_threshold=args.semantic_z_threshold,
    )


def semantic_from_render(
    rendered_lf: torch.Tensor,
    predictor: RadioDualSemanticPredictor,
    candidate_ids: list[int],
    palette: np.ndarray,
    target_size: tuple[int, int],
    apply_components: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense Replica51 IDs [H,W] and panel-size BGR semantic colors."""
    with torch.inference_mode():
        prediction = predictor.predict(rendered_lf, candidate_ids=candidate_ids)
    labels = prediction.labels.detach().cpu().numpy().astype(np.uint8)

    if apply_components:
        labels = refine_label_image(
            labels,
            min_area=100,
            max_fill_area=5000,
            ignore_index=REPLICA51_TAXONOMY.ignore_index,
        )

    visual_rgb = _colorize_semantic_mask(
        labels, palette, REPLICA51_TAXONOMY.ignore_index
    )
    visual_bgr = cv2.cvtColor(visual_rgb, cv2.COLOR_RGB2BGR)
    visual_bgr = cv2.resize(visual_bgr, target_size, interpolation=cv2.INTER_NEAREST)
    return labels, visual_bgr


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("LEG-SLAM Gaussian rasterizer requires CUDA.")
    if args.frame_stride <= 0 or args.panel_width < 64 or args.fps <= 0:
        raise ValueError("frame-stride, fps and panel-width must be positive.")
    if args.max_frames < 0:
        raise ValueError("max-frames must be >= 0")
    if args.pca_sample_views < 1 or args.pca_pixels_per_view < 1:
        raise ValueError("PCA sample counts must be positive.")
    if args.radio_chunk_size <= 0:
        raise ValueError("--radio-chunk-size must be positive")

    result = args.result.resolve()
    sequence = args.sequence.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not sequence.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {sequence}")

    scene = load_gaussian_scene(result.parent, result.name)
    if args.camera_mode == "full":
        views = build_full_views(
            scene, sequence, args.dataset, result, args.trajectory
        )
    else:
        saved = sorted(scene.cameras, key=lambda c: int(c.get("id", 0)))
        views = [(cam, get_original_rgb(cam, sequence, args.dataset)) for cam in saved]
        print(f"[Views] Using {len(views)} saved Gaussian keyframes")

    views = views[::args.frame_stride]
    if args.max_frames:
        views = views[:args.max_frames]
    if not views:
        raise RuntimeError("No renderable views after striding/max-frames")

    cameras = [cam for cam, _ in views]
    first_img = views[0][1]
    original_w = int(cameras[0]["width"])
    original_h = int(cameras[0]["height"])
    panel_w = args.panel_width + args.panel_width % 2
    panel_h = max(2, int(round(panel_w * original_h / original_w)))
    panel_h += panel_h % 2
    size = (panel_w, panel_h)
    print(f"[Video] {len(views)} views; first RGB: {first_img}")
    print(f"[Video] 2x2 output: {2 * panel_w} x {2 * panel_h}, {args.fps} fps")

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    predictor = build_replica51_dual_predictor(args)
    candidate_ids = list(range(len(REPLICA51_TAXONOMY.class_names)))
    palette = _make_semantic_palette(len(candidate_ids))
    print(f"[Semantic] Replica51: {len(candidate_ids)} queries, "
          f"refinement={args.semantic_refinement}")

    mean, basis, lower, upper, explained = fit_pca_from_rendered_views(
        cameras, scene, background, args.pca_sample_views, args.pca_pixels_per_view
    )
    np.savez_compressed(
        output_path.with_suffix(".pca.npz"),
        mean=mean, basis=basis, lower=lower, upper=upper,
        explained_variance_ratio=explained,
    )

    masks_root = args.save_semantic_masks
    if masks_root is not None:
        import json
        masks_root = masks_root.resolve()
        (masks_root / "pred_ids").mkdir(parents=True, exist_ok=True)
        (masks_root / "pred_vis").mkdir(parents=True, exist_ok=True)
        (masks_root / "classes.json").write_text(json.dumps({
            str(idx): name for idx, name in enumerate(REPLICA51_TAXONOMY.class_names)
        }, indent=2))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (2 * panel_w, 2 * panel_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    preview = None
    try:
        for cam, rgb_path in tqdm(views, desc="Rendering MP4"):
            original = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if original is None:
                raise RuntimeError(f"Could not read RGB frame: {rgb_path}")
            original = cv2.resize(original, size, interpolation=cv2.INTER_AREA)

            outputs = render_one(cam, scene, background)
            rendered = rgb_from_render(outputs["rendered_image"], size)
            pca = feature_visualization(
                outputs["rendered_lf"], size, mean, basis, lower, upper
            )
            labels, semantic_vis = semantic_from_render(
                outputs["rendered_lf"], predictor, candidate_ids, palette,
                size, apply_components=(args.semantic_refinement == "components")
            )
            frame_name = rgb_path.stem
            if masks_root is not None:
                if not cv2.imwrite(str(masks_root / "pred_ids" / f"{frame_name}.png"), labels):
                    raise RuntimeError(f"Cannot save class-ID mask for {frame_name}")
                # Save at rendered resolution; video panel is resized for display.
                vis_native_rgb = _colorize_semantic_mask(
                    labels, palette, REPLICA51_TAXONOMY.ignore_index
                )
                vis_native_bgr = cv2.cvtColor(vis_native_rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(masks_root / "pred_vis" / f"{frame_name}.png"), vis_native_bgr):
                    raise RuntimeError(f"Cannot save color mask for {frame_name}")
            top = np.hstack((
                put_title(original, "Real RGB", frame_name),
                put_title(rendered, "Rendered RGB")
            ))
            bottom = np.hstack((
                put_title(pca, "Rendered features: 3 PCA components"),
                put_title(semantic_vis, "Replica51: RADIO dual + components" if args.semantic_refinement == "components"
                          else "Replica51: RADIO dual")
            ))
            composed = np.vstack((top, bottom))
            writer.write(composed)
            if preview is None:
                preview = composed.copy()
            del outputs

    finally:
        writer.release()
    if preview is not None:
        preview_path = output_path.with_suffix(".preview.png")
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Could not save preview: {preview_path}")
        print(f"Preview: {preview_path}")
    print(f"Video: {output_path}")
    print(f"Fixed PCA transform: {output_path.with_suffix('.pca.npz')}")
    if masks_root is not None:
        print(f"Predicted masks: {masks_root}")

if __name__ == "__main__":
    main()

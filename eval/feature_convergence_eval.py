#!/usr/bin/env python3
"""
Diagnose convergence of LEG-SLAM semantic Gaussian features against the
image-feature targets used during mapping.

The evaluator reports only four values:

  1. compressed cosine similarity
  2. compressed norm ratio ||rendered|| / ||target||
  3. reconstructed full-space cosine similarity
  4. reconstructed full-space norm ratio ||rendered_full|| / ||target_full||

Cosine similarity measures only vector direction (the cosine of the angle).
Norm ratio measures only relative magnitude.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.utils import MiniCam, focal2fov, get_world2view  # noqa: E402
from eval.gaussian_model import GaussianModel  # noqa: E402
from eval.render import render  # noqa: E402


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_registry(path: Path) -> Dict[str, dict]:
    with path.open("r") as f:
        root = json.load(f)
    models = root.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Registry has no 'models' object: {path}")
    return models


def parse_frame_index(path_or_name: str) -> int:
    stem = Path(str(path_or_name)).stem
    matches = re.findall(r"\d+", stem)
    if not matches:
        raise ValueError(f"Could not extract frame index from {path_or_name!r}")
    return int(matches[-1])


def find_rgb_path(
    scene_data: Path,
    camera_param: dict,
    camera_idx: int,
    dataset: str,
) -> Path:
    """Find the RGB frame corresponding to this cameras.json entry."""
    image_name = str(camera_param.get("img_name", f"frame{camera_idx:06d}.jpg"))
    frame_idx = parse_frame_index(image_name)

    if dataset == "replica":
        rgb_dir = scene_data / "results"
        candidates = [
            rgb_dir / f"frame{frame_idx:06d}.jpg",
            rgb_dir / f"frame{frame_idx:06d}.png",
            rgb_dir / f"frame{frame_idx:06d}.jpeg",
            rgb_dir / Path(image_name).name,
        ]
    elif dataset == "scannet":
        rgb_dir = scene_data / "color"
        candidates = [
            rgb_dir / f"{frame_idx}.jpg",
            rgb_dir / f"{frame_idx}.png",
            rgb_dir / f"{frame_idx}.jpeg",
            rgb_dir / f"{frame_idx:06d}.jpg",
            rgb_dir / f"{frame_idx:06d}.png",
            rgb_dir / f"{frame_idx:06d}.jpeg",
            rgb_dir / Path(image_name).name,
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find RGB frame {frame_idx} for dataset={dataset!r} "
        f"in {rgb_dir}"
    )


def nearest_supported_size(size: int, step: int) -> int:
    # Python round() uses ties-to-even, matching RADIO's Python implementation.
    return max(int(round(size / step)) * step, step)


class RadioTargetExtractor:
    def __init__(self, model_cfg: dict, device: torch.device):
        encoder_cfg = model_cfg.get("encoder", {})
        compressor_cfg = model_cfg.get("compressor", {})

        if encoder_cfg.get("type") != "RadioTorchScript":
            raise ValueError(
                "This diagnostic currently supports encoder.type="
                f"'RadioTorchScript', got {encoder_cfg.get('type')!r}"
            )
        if compressor_cfg.get("type") != "TorchPCA":
            raise ValueError(
                "This diagnostic currently supports compressor.type='TorchPCA', "
                f"got {compressor_cfg.get('type')!r}"
            )

        self.device = device
        self.resolution_step = int(encoder_cfg.get("resolution_step", 16))
        self.full_dim = int(encoder_cfg["embedding_size"])
        self.compressed_dim = int(compressor_cfg["output_dim"])

        encoder_path = resolve_repo_path(encoder_cfg["path"])
        compressor_path = resolve_repo_path(compressor_cfg["path"])

        # The C++ RADIO path needed graph-executor optimization disabled for
        # stability. Apply the same setting here when this PyTorch exposes it.
        if hasattr(torch._C, "_set_graph_executor_optimize"):
            torch._C._set_graph_executor_optimize(False)

        print(f"Loading RADIO TorchScript encoder: {encoder_path}")
        self.encoder = torch.jit.load(
            str(encoder_path), map_location=device
        ).eval()

        print(f"Loading RADIO TorchScript PCA: {compressor_path}")
        self.pca = torch.jit.load(
            str(compressor_path), map_location=device
        ).eval()

        pca_state = self.pca.state_dict()
        if "mean" not in pca_state or "components" not in pca_state:
            raise RuntimeError(
                "TorchPCA module does not expose mean/components buffers."
            )
        self.pca_mean = pca_state["mean"].detach().float().to(device)
        self.pca_components = (
            pca_state["components"][:self.compressed_dim]
            .detach()
            .float()
            .to(device)
        )

    @torch.inference_mode()
    def extract(
        self,
        image_path: Path,
        camera_height: Optional[int] = None,
        camera_width: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Return the compressed RADIO target used by mapping as [K,Hf,Wf].

        If the stored RGB resolution differs from the reconstruction camera
        resolution (common for ScanNet), resize RGB to the camera resolution
        first, then apply RADIO's normal nearest-supported-size preprocessing.
        """
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        x = x.to(self.device, dtype=torch.float32) / 255.0

        source_h, source_w = x.shape[-2:]
        if camera_height is not None and camera_width is not None:
            camera_height = int(camera_height)
            camera_width = int(camera_width)
            if (source_h, source_w) != (camera_height, camera_width):
                x = F.interpolate(
                    x,
                    size=(camera_height, camera_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

        h0, w0 = x.shape[-2:]
        h = nearest_supported_size(h0, self.resolution_step)
        w = nearest_supported_size(w0, self.resolution_step)

        if (h, w) != (h0, w0):
            x = F.interpolate(
                x,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        spatial = self.encoder(x)
        if spatial.ndim != 3 or spatial.shape[0] != 1:
            raise RuntimeError(
                f"Expected RADIO output [1,N,D], got {tuple(spatial.shape)}"
            )

        spatial = spatial[0].float()
        n, d = spatial.shape
        if d != self.full_dim:
            raise RuntimeError(
                f"RADIO feature dim mismatch: got {d}, expected {self.full_dim}"
            )

        hf = h // self.resolution_step
        wf = w // self.resolution_step
        if hf * wf != n:
            raise RuntimeError(
                f"RADIO grid mismatch: N={n}, expected {hf}x{wf}={hf*wf}"
            )

        compressed = self.pca(spatial, self.compressed_dim).float()
        if compressed.shape != (n, self.compressed_dim):
            raise RuntimeError(
                "Unexpected PCA output shape: "
                f"{tuple(compressed.shape)}, expected {(n, self.compressed_dim)}"
            )

        return (
            compressed
            .reshape(hf, wf, self.compressed_dim)
            .permute(2, 0, 1)
            .contiguous()
        )


def load_scene_model(
    recon_root: Path,
    scene_name: str,
) -> Tuple[GaussianModel, List[dict]]:
    scene_path = recon_root / scene_name
    ply_path = scene_path / "experiment" / "ply" / "point_cloud" / "point_cloud.ply"
    cameras_path = scene_path / "experiment" / "ply" / "cameras.json"

    if not ply_path.exists():
        raise FileNotFoundError(f"Missing point cloud: {ply_path}")
    if not cameras_path.exists():
        raise FileNotFoundError(f"Missing cameras.json: {cameras_path}")

    print(f"Reading Gaussian reconstruction: {ply_path}")
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_path))

    with cameras_path.open("r") as f:
        camera_params = json.load(f)

    if not camera_params:
        raise ValueError(f"No cameras found in {cameras_path}")

    return gaussians, camera_params


@torch.inference_mode()
def full_space_metrics(
    rendered: torch.Tensor,
    target: torch.Tensor,
    components: torch.Tensor,
    mean: torch.Tensor,
    valid: torch.Tensor,
    chunk_size: int = 32768,
) -> Dict[str, float]:
    """
    Reconstruct compressed PCA features back to the original feature space and
    report only:
      - cosine similarity
      - norm ratio ||rendered_full|| / ||target_full||
    """
    k, h, w = target.shape

    target_flat = target.permute(1, 2, 0).reshape(-1, k)
    rendered_flat = rendered.permute(1, 2, 0).reshape(-1, k)
    valid_flat = valid.reshape(-1)

    target_flat = target_flat[valid_flat]
    rendered_flat = rendered_flat[valid_flat]

    cosine_sum = 0.0
    norm_ratio_sum = 0.0
    count = 0

    for start in range(0, target_flat.shape[0], chunk_size):
        gt_c = target_flat[start:start + chunk_size]
        pred_c = rendered_flat[start:start + chunk_size]

        # PCA inverse:
        # X_hat = Y @ components + mean
        gt_full = torch.matmul(gt_c, components) + mean
        pred_full = torch.matmul(pred_c, components) + mean

        cosine = F.cosine_similarity(
            pred_full,
            gt_full,
            dim=-1,
            eps=1e-8,
        )

        gt_norm = torch.linalg.vector_norm(gt_full, dim=-1)
        pred_norm = torch.linalg.vector_norm(pred_full, dim=-1)

        norm_ratio = pred_norm / gt_norm.clamp_min(1e-8)

        cosine_sum += float(cosine.sum().item())
        norm_ratio_sum += float(norm_ratio.sum().item())
        count += cosine.numel()

    if count == 0:
        raise RuntimeError("No valid pixels for reconstructed-space comparison.")

    return {
        "cosine_similarity": cosine_sum / count,
        "norm_ratio": norm_ratio_sum / count,
    }


@torch.inference_mode()
def evaluate_scene(
    scene_name: str,
    data_root: Path,
    recon_root: Path,
    dataset: str,
    extractor: RadioTargetExtractor,
    every_n_frame: int,
    max_frames: Optional[int],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    scene_data = data_root / scene_name
    gaussians, camera_params = load_scene_model(recon_root, scene_name)

    background = torch.zeros(3, dtype=torch.float32, device=device)

    rows: List[Dict[str, object]] = []

    pooled_compressed_cosine_sum = 0.0
    pooled_compressed_norm_ratio_sum = 0.0
    pooled_reconstructed_cosine_sum = 0.0
    pooled_reconstructed_norm_ratio_sum = 0.0
    pooled_count = 0

    evaluated = 0

    for camera_idx, camera_param in enumerate(
        tqdm(camera_params, desc=f"{scene_name}: convergence")
    ):
        if camera_idx % every_n_frame != 0:
            continue
        if max_frames is not None and evaluated >= max_frames:
            break

        image_path = find_rgb_path(
            scene_data, camera_param, camera_idx, dataset=dataset
        )
        frame_idx = parse_frame_index(image_path.name)

        w = int(camera_param["width"])
        h = int(camera_param["height"])
        fovx = focal2fov(float(camera_param["fx"]), w)
        fovy = focal2fov(float(camera_param["fy"]), h)

        world_view_transform = get_world2view(
            np.array(camera_param["rotation"]),
            np.array(camera_param["position"]),
        )
        cam = MiniCam(w, h, fovx, fovy, world_view_transform)

        render_result = render(cam, gaussians, background)
        rendered_lf = render_result["rendered_lf"].detach().float()

        if rendered_lf.ndim != 3:
            raise RuntimeError(
                f"Expected rendered_lf [K,H,W], got {tuple(rendered_lf.shape)}"
            )

        target_lowres = extractor.extract(
            image_path,
            camera_height=h,
            camera_width=w,
        )

        # Match the C++ training path.
        target = F.interpolate(
            target_lowres.unsqueeze(0),
            size=rendered_lf.shape[-2:],
            mode="nearest",
        ).squeeze(0)

        if target.shape != rendered_lf.shape:
            raise RuntimeError(
                f"Target/rendered mismatch: {tuple(target.shape)} vs "
                f"{tuple(rendered_lf.shape)}"
            )

        # --------------------------------------------------------------
        # 1) COMPRESSED FEATURE SPACE
        # --------------------------------------------------------------
        target_norm = torch.linalg.vector_norm(target, dim=0)
        rendered_norm = torch.linalg.vector_norm(rendered_lf, dim=0)

        valid = (target_norm > 1e-8) & (rendered_norm > 1e-8)
        valid_count = int(valid.sum().item())
        total_count = int(valid.numel())

        if valid_count == 0:
            print(f"WARNING: frame {frame_idx} has no non-zero semantic pixels.")
            del render_result, rendered_lf, target_lowres, target
            continue

        # Cosine of the angle between the two compressed vectors.
        compressed_cosine_map = F.cosine_similarity(
            rendered_lf,
            target,
            dim=0,
            eps=1e-8,
        )

        # Pure magnitude ratio. Direction/angle plays no role here.
        compressed_norm_ratio_map = (
            rendered_norm / target_norm.clamp_min(1e-8)
        )

        compressed_cosine = float(
            compressed_cosine_map[valid].mean().item()
        )
        compressed_norm_ratio = float(
            compressed_norm_ratio_map[valid].mean().item()
        )

        # --------------------------------------------------------------
        # 2) RECONSTRUCTED FULL FEATURE SPACE
        # --------------------------------------------------------------
        reconstructed = full_space_metrics(
            rendered=rendered_lf,
            target=target,
            components=extractor.pca_components,
            mean=extractor.pca_mean,
            valid=valid,
        )

        reconstructed_cosine = reconstructed["cosine_similarity"]
        reconstructed_norm_ratio = reconstructed["norm_ratio"]

        pooled_compressed_cosine_sum += compressed_cosine * valid_count
        pooled_compressed_norm_ratio_sum += compressed_norm_ratio * valid_count
        pooled_reconstructed_cosine_sum += reconstructed_cosine * valid_count
        pooled_reconstructed_norm_ratio_sum += reconstructed_norm_ratio * valid_count
        pooled_count += valid_count

        row = {
            "frame": frame_idx,
            "image": str(image_path),
            "valid_pixels": valid_count,
            "total_pixels": total_count,
            "compressed_cosine_similarity": compressed_cosine,
            "compressed_norm_ratio": compressed_norm_ratio,
            "reconstructed_cosine_similarity": reconstructed_cosine,
            "reconstructed_norm_ratio": reconstructed_norm_ratio,
        }
        rows.append(row)
        evaluated += 1

        if evaluated <= 5 or evaluated % 10 == 0:
            print(
                f"\nframe {frame_idx:06d}: "
                f"compressed cosine={compressed_cosine:+.4f}, "
                f"compressed norm-ratio={compressed_norm_ratio:.4f}, "
                f"reconstructed cosine={reconstructed_cosine:+.4f}, "
                f"reconstructed norm-ratio={reconstructed_norm_ratio:.4f}"
            )

        del (
            render_result,
            rendered_lf,
            target_lowres,
            target,
            compressed_cosine_map,
            compressed_norm_ratio_map,
        )

    if not rows or pooled_count == 0:
        raise RuntimeError(f"No valid frames evaluated for scene {scene_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{scene_name}_feature_convergence.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    compressed_cosine = pooled_compressed_cosine_sum / pooled_count
    compressed_norm_ratio = pooled_compressed_norm_ratio_sum / pooled_count
    reconstructed_cosine = pooled_reconstructed_cosine_sum / pooled_count
    reconstructed_norm_ratio = pooled_reconstructed_norm_ratio_sum / pooled_count

    summary = {
        "scene": scene_name,
        "dataset": dataset,
        "frames_evaluated": len(rows),
        "valid_pixel_count": pooled_count,
        "compressed": {
            "cosine_similarity": compressed_cosine,
            "norm_ratio": compressed_norm_ratio,
        },
        "reconstructed": {
            "cosine_similarity": reconstructed_cosine,
            "norm_ratio": reconstructed_norm_ratio,
        },
        "csv": str(csv_path),
    }

    json_path = output_dir / f"{scene_name}_feature_convergence_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print(f"{scene_name} feature convergence")
    print("=" * 78)
    print(
        f"compressed:    cosine={compressed_cosine:+.6f}  "
        f"norm-ratio={compressed_norm_ratio:.6f}"
    )
    print(
        f"reconstructed: cosine={reconstructed_cosine:+.6f}  "
        f"norm-ratio={reconstructed_norm_ratio:.6f}"
    )
    print(f"valid pixels: {pooled_count:,}")
    print(f"per-frame CSV: {csv_path}")
    print(f"summary JSON: {json_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report cosine similarity and norm ratio in compressed and reconstructed RADIO feature spaces."
        )
    )
    parser.add_argument(
        "--feature-model",
        default="radio_v4_so400m",
        help="Feature model ID from cfg/feature_models.json.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "cfg" / "feature_models.json",
    )
    parser.add_argument(
        "--dataset",
        choices=["replica", "scannet"],
        default="replica",
        help=(
            "Dataset layout used to locate RGB images: "
            "Replica uses <scene>/results/frameXXXXXX.*, "
            "ScanNet uses <scene>/color/<frame_idx>.*."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/workspace/data/Replica"),
    )
    parser.add_argument(
        "--recon-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--every-n-frame",
        type=int,
        default=1,
        help="Evaluate every Nth reconstruction camera.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum evaluated frames per scene.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "feature_convergence_results",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.every_n_frame <= 0:
        raise ValueError("--every-n-frame must be >= 1")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    registry_path = args.registry.resolve()
    models = load_registry(registry_path)

    if args.feature_model not in models:
        raise ValueError(
            f"Unknown feature model {args.feature_model!r}. "
            f"Available: {sorted(models.keys())}"
        )

    model_cfg = models[args.feature_model]
    extractor = RadioTargetExtractor(model_cfg, device)

    args.data_root = args.data_root.resolve()
    args.recon_root = args.recon_root.resolve()
    args.output = args.output.resolve()

    all_summaries = {}
    for scene in args.scenes:
        all_summaries[scene] = evaluate_scene(
            scene_name=scene,
            data_root=args.data_root,
            recon_root=args.recon_root,
            dataset=args.dataset,
            extractor=extractor,
            every_n_frame=args.every_n_frame,
            max_frames=args.max_frames,
            output_dir=args.output,
            device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_path = args.output / "feature_convergence_summary.json"
    args.output.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w") as f:
        json.dump(
            {
                "feature_model": args.feature_model,
                "dataset": args.dataset,
                "data_root": str(args.data_root),
                "recon_root": str(args.recon_root),
                "scenes": all_summaries,
            },
            f,
            indent=2,
        )

    print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()

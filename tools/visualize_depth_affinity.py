from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.semantic.refinement import (
    DepthAffinityProvider,
    LocalAffinity,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--depth",
        type=Path,
        required=True,
        help="Replica depth PNG.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Depth affinity sigma in meters.",
    )

    parser.add_argument(
        "--dilations",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )

    parser.add_argument(
        "--lambda-refine",
        type=float,
        default=1.0,
        help=(
            "Used only for the neighbor-influence visualization."
        ),
    )

    parser.add_argument(
        "--depth-scale",
        type=float,
        default=6553.5,
        help="Replica depth scale.",
    )

    return parser.parse_args()


def paired_slices(
    height: int,
    width: int,
    dy: int,
    dx: int,
):
    """
    Return source/destination slices for:

        dst = src + (dy, dx)
    """

    if dy >= 0:
        src_y = slice(
            0,
            height - dy,
        )
        dst_y = slice(
            dy,
            height,
        )
    else:
        src_y = slice(
            -dy,
            height,
        )
        dst_y = slice(
            0,
            height + dy,
        )

    if dx >= 0:
        src_x = slice(
            0,
            width - dx,
        )
        dst_x = slice(
            dx,
            width,
        )
    else:
        src_x = slice(
            -dx,
            width,
        )
        dst_x = slice(
            0,
            width + dx,
        )

    return (
        src_y,
        src_x,
        dst_y,
        dst_x,
    )


def accumulate_affinity(
    affinity: LocalAffinity,
    height: int,
    width: int,
):
    """
    Convert edge-based affinities into per-pixel maps.

    Returns:
        total_sum:
            sum_j w_ij

        total_count:
            number of geometrically available neighbors

        dilation_sum:
            sum of affinities grouped by dilation

        dilation_count:
            number of neighbors grouped by dilation
    """

    device = affinity.edges[0].weights.device
    dtype = affinity.edges[0].weights.dtype

    total_sum = torch.zeros(
        (height, width),
        dtype=dtype,
        device=device,
    )

    total_count = torch.zeros(
        (height, width),
        dtype=dtype,
        device=device,
    )

    dilation_sum = defaultdict(
        lambda: torch.zeros(
            (height, width),
            dtype=dtype,
            device=device,
        )
    )

    dilation_count = defaultdict(
        lambda: torch.zeros(
            (height, width),
            dtype=dtype,
            device=device,
        )
    )

    for edge in affinity.edges:

        (
            src_y,
            src_x,
            dst_y,
            dst_x,
        ) = paired_slices(
            height=height,
            width=width,
            dy=edge.dy,
            dx=edge.dx,
        )

        weights = edge.weights

        # Each stored edge is undirected:
        # it contributes to both endpoint pixels.

        total_sum[
            src_y,
            src_x,
        ] += weights

        total_sum[
            dst_y,
            dst_x,
        ] += weights

        total_count[
            src_y,
            src_x,
        ] += 1

        total_count[
            dst_y,
            dst_x,
        ] += 1

        dilation = max(
            abs(edge.dy),
            abs(edge.dx),
        )

        dilation_sum[dilation][
            src_y,
            src_x,
        ] += weights

        dilation_sum[dilation][
            dst_y,
            dst_x,
        ] += weights

        dilation_count[dilation][
            src_y,
            src_x,
        ] += 1

        dilation_count[dilation][
            dst_y,
            dst_x,
        ] += 1

    return (
        total_sum,
        total_count,
        dilation_sum,
        dilation_count,
    )


def normalize_to_uint8(
    image: np.ndarray,
    min_value: float,
    max_value: float,
) -> np.ndarray:

    image = np.nan_to_num(
        image,
        nan=min_value,
        posinf=max_value,
        neginf=min_value,
    )

    image = np.clip(
        image,
        min_value,
        max_value,
    )

    if max_value <= min_value:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    image = (
        (image - min_value)
        / (max_value - min_value)
    )

    return np.round(
        image * 255.0
    ).astype(np.uint8)


def save_heatmap(
    path: Path,
    image: np.ndarray,
    min_value: float = 0.0,
    max_value: float = 1.0,
):
    gray = normalize_to_uint8(
        image,
        min_value=min_value,
        max_value=max_value,
    )

    heatmap = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO,
    )

    cv2.imwrite(
        str(path),
        heatmap,
    )


def main():

    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Load Replica depth
    # ----------------------------------------------------------

    depth_raw = cv2.imread(
        str(args.depth),
        cv2.IMREAD_UNCHANGED,
    )

    if depth_raw is None:
        raise FileNotFoundError(
            f"Could not read depth: {args.depth}"
        )

    if depth_raw.ndim != 2:
        raise RuntimeError(
            f"Expected [H,W] depth, got "
            f"{depth_raw.shape}"
        )

    depth_m = (
        depth_raw.astype(np.float32)
        / args.depth_scale
    )

    depth = torch.from_numpy(
        depth_m
    )

    height, width = depth.shape

    # ----------------------------------------------------------
    # Use exactly the same affinity provider as evaluation
    # ----------------------------------------------------------

    provider = DepthAffinityProvider(
        sigma_d=args.sigma,
        dilations=args.dilations,
    )

    affinity = provider.compute(
        {
            "depth": depth,
        }
    )

    (
        affinity_sum,
        neighbor_count,
        dilation_sum,
        dilation_count,
    ) = accumulate_affinity(
        affinity,
        height=height,
        width=width,
    )

    # ----------------------------------------------------------
    # Mean affinity per pixel
    # ----------------------------------------------------------

    mean_affinity = (
        affinity_sum
        / neighbor_count.clamp_min(1)
    )

    # ----------------------------------------------------------
    # Actual relative neighborhood contribution in our update
    #
    #     x_new =
    #       (s + lambda * sum(w*x))
    #       / (1 + lambda * sum(w))
    #
    # Relative denominator contribution of neighbors:
    #
    #     lambda * sum(w)
    #     -----------------
    #     1 + lambda * sum(w)
    #
    # 0 -> almost entirely original score
    # 1 -> almost entirely neighbors
    # ----------------------------------------------------------

    weighted_neighbor_sum = (
        args.lambda_refine
        * affinity_sum
    )

    neighbor_influence = (
        weighted_neighbor_sum
        / (1.0 + weighted_neighbor_sum)
    )

    # ----------------------------------------------------------
    # Save main maps
    # ----------------------------------------------------------

    affinity_sum_np = (
        affinity_sum.cpu().numpy()
    )

    mean_affinity_np = (
        mean_affinity.cpu().numpy()
    )

    neighbor_influence_np = (
        neighbor_influence.cpu().numpy()
    )

    depth_np = depth.cpu().numpy()

    valid_depth = (
        np.isfinite(depth_np)
        & (depth_np > 0)
    )

    if np.any(valid_depth):
        depth_max = float(
            np.percentile(
                depth_np[valid_depth],
                99,
            )
        )
    else:
        depth_max = 1.0

    save_heatmap(
        args.out_dir / "depth.png",
        depth_np,
        min_value=0.0,
        max_value=depth_max,
    )

    save_heatmap(
        args.out_dir / "affinity_mean.png",
        mean_affinity_np,
        min_value=0.0,
        max_value=1.0,
    )

    save_heatmap(
        args.out_dir / "neighbor_influence.png",
        neighbor_influence_np,
        min_value=0.0,
        max_value=1.0,
    )

    # For affinity sum, don't normalize each image independently.
    # The theoretical maximum is approximately the number of
    # available neighbors.
    max_neighbors = (
        8 * len(args.dilations)
    )

    save_heatmap(
        args.out_dir / "affinity_sum.png",
        affinity_sum_np,
        min_value=0.0,
        max_value=float(max_neighbors),
    )

    # ----------------------------------------------------------
    # Save one mean-affinity image per dilation
    # ----------------------------------------------------------

    for dilation in sorted(
        dilation_sum
    ):

        mean_dilation = (
            dilation_sum[dilation]
            / dilation_count[
                dilation
            ].clamp_min(1)
        )

        save_heatmap(
            args.out_dir
            / f"affinity_d{dilation}.png",
            mean_dilation.cpu().numpy(),
            min_value=0.0,
            max_value=1.0,
        )

    # ----------------------------------------------------------
    # Save raw numerical arrays too
    # ----------------------------------------------------------

    np.save(
        args.out_dir / "affinity_sum.npy",
        affinity_sum_np,
    )

    np.save(
        args.out_dir / "affinity_mean.npy",
        mean_affinity_np,
    )

    np.save(
        args.out_dir / "neighbor_influence.npy",
        neighbor_influence_np,
    )

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    print()
    print("Depth affinity visualization")
    print("----------------------------")
    print(
        f"Depth:       {args.depth}"
    )
    print(
        f"Resolution:  {width}x{height}"
    )
    print(
        f"Sigma:       {args.sigma} m"
    )
    print(
        f"Dilations:   {args.dilations}"
    )
    print(
        f"Lambda:      {args.lambda_refine}"
    )
    print(
        f"Edges:       {len(affinity.edges)} "
        f"stored / up to "
        f"{max_neighbors} neighbors per pixel"
    )

    valid = neighbor_count > 0

    print()
    print("Affinity statistics")
    print("-------------------")

    print(
        "Mean affinity: "
        f"{mean_affinity[valid].mean().item():.4f}"
    )

    print(
        "Mean affinity sum: "
        f"{affinity_sum[valid].mean().item():.4f}"
    )

    print(
        "Mean neighbor influence: "
        f"{neighbor_influence[valid].mean().item():.4f}"
    )

    print(
        "Median neighbor influence: "
        f"{neighbor_influence[valid].median().item():.4f}"
    )

    print()
    print(
        f"Saved to: {args.out_dir}"
    )


if __name__ == "__main__":
    main()
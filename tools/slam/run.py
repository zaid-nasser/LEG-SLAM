from __future__ import annotations

import argparse
from pathlib import Path

from .runner import (
    SlamRunConfig,
    run_slam,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LEG-SLAM",
    )

    parser.add_argument(
        "--sequence",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--orb-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--feature-registry", "--encoder-config",
        dest="encoder_config",
        type=Path,
        default=Path("cfg/feature_models.json"),
        help="Path to the JSON feature model registry.",
    )

    parser.add_argument(
        "--gaussian-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=Path(
            "ORB-SLAM3/Vocabulary/ORBvoc.txt"
        ),
    )

    parser.add_argument(
        "--executable",
        type=Path,
        default=Path(
            "bin/legs_slam_rgbd"
        ),
    )

    parser.add_argument(
        "--feature-model",
        default="radio_v4_so400m",
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Run even if the final "
            "point_cloud.ply already exists."
        ),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=("replica", "scannet", "mmpr"),
        help="Dataset layout used by the RGB-D sequence.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = SlamRunConfig(
        executable=args.executable,
        vocabulary=args.vocabulary,
        orb_config=args.orb_config,
        encoder_config=args.encoder_config,
        gaussian_config=args.gaussian_config,
        sequence=args.sequence,
        output=args.output,
        feature_model=args.feature_model,
        viewer=args.viewer,
        dataset = args.dataset,
    )

    result = run_slam(
        config,
        skip_existing=not args.force,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise SystemExit(
            result.returncode
        )

    print(
        f"Output: {result.output_dir}"
    )


if __name__ == "__main__":
    main()
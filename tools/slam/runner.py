from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SlamRunConfig:
    vocabulary: Path
    orb_config: Path
    encoder_config: Path
    gaussian_config: Path
    sequence: Path
    output: Path
    dataset: str

    executable: Path = Path("./bin/legs_slam_rgbd")
    feature_model: Optional[str] = None
    viewer: bool = False


@dataclass(frozen=True)
class SlamRunResult:
    returncode: int
    stdout: str
    output_dir: Path


def validate_mmpr_sequence(sequence: Path) -> None:
    """
    Validate MMPR RGB-D filenames and the first depth tensor.

    This checks the input structure only. Decoding .npy depth during SLAM
    must be implemented in the C++ legs_slam_rgbd dataset reader.
    """
    sequence = Path(sequence)
    rgb_dir = sequence / "cam0" / "data"
    depth_dir = sequence / "depth" / "data"

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"MMPR RGB directory not found: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"MMPR depth directory not found: {depth_dir}")

    rgb = {p.stem for p in rgb_dir.glob("*.png") if p.is_file()}
    depth = {p.stem for p in depth_dir.glob("*.npy") if p.is_file()}

    if not rgb or not depth:
        raise RuntimeError(
            f"MMPR has no RGB PNGs or depth NPYs: {sequence}"
        )

    if rgb != depth:
        rgb_only = sorted(rgb - depth)[:5]
        depth_only = sorted(depth - rgb)[:5]
        raise RuntimeError(
            "MMPR RGB and depth timestamp sets differ. "
            f"RGB-only examples: {rgb_only}; "
            f"depth-only examples: {depth_only}"
        )

    # Validate a representative depth array without reading all its pixels.
    import numpy as np

    first_timestamp = min(rgb, key=int)
    sample = np.load(
        depth_dir / f"{first_timestamp}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if sample.ndim != 2 or sample.dtype != np.float32:
        raise ValueError(
            "MMPR depth must be a 2D float32 array in meters; "
            f"got shape={sample.shape}, dtype={sample.dtype}"
        )

    print(
        f"MMPR validated: {len(rgb)} matching RGB-D pairs, "
        f"sample depth {tuple(sample.shape)} float32"
    )


def build_slam_command(
    config: SlamRunConfig,
) -> list[str]:
    if config.dataset == "mmpr":
        validate_mmpr_sequence(config.sequence)

    command = [
        str(config.executable),
        str(config.vocabulary),
        str(config.orb_config),
        str(config.encoder_config),
        str(config.gaussian_config),
        str(config.sequence),
        str(config.output),
        "viewer" if config.viewer else "no_viewer",
    ]

    if config.feature_model:
        command.extend(
            [
                "--feature-model",
                config.feature_model,
            ]
        )

    # Dataset selection does not depend on the feature model.
    command.extend(["--dataset", config.dataset])

    return command


def reconstruction_exists(
    output_dir: Path,
) -> bool:
    point_cloud = (
        Path(output_dir)
        / "experiment"
        / "ply"
        / "point_cloud"
        / "point_cloud.ply"
    )

    return point_cloud.is_file()


def run_slam(
    config: SlamRunConfig,
    *,
    skip_existing: bool = True,
    timeout: Optional[float] = None,
) -> SlamRunResult:

    config.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        skip_existing
        and reconstruction_exists(config.output)
    ):
        return SlamRunResult(
            returncode=0,
            stdout="Reconstruction already exists.",
            output_dir=config.output,
        )

    command = build_slam_command(config)

    print(
        "Running:",
        " ".join(command),
    )

    process = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return SlamRunResult(
        returncode=process.returncode,
        stdout=(
            process.stdout
            + process.stderr
        ),
        output_dir=config.output,
    )
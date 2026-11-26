from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import re
from ..gaussian_model import GaussianModel
from ..utils import MiniCam, focal2fov, get_world2view


def camera_frame_id(
    camera_param: dict,
    camera_idx: int,
) -> int:
    """
    Recover the dataset frame ID associated with one cameras.json entry.

    Supports names such as:
        frame000123.jpg
        123.jpg
        /path/to/color/123.jpg
    """
    image_name = str(
        camera_param.get(
            "img_name",
            camera_idx,
        )
    )

    stem = Path(image_name).stem

    matches = re.findall(r"\d+", stem)

    if not matches:
        return camera_idx

    return int(matches[-1])


def camera_pose_matrix(
    camera_param: dict,
) -> np.ndarray:
    """
    Convert a cameras.json entry into the 4x4 pose representation
    used by the existing LEG-SLAM evaluation code.
    """
    pose = np.eye(
        4,
        dtype=np.float64,
    )

    pose[:3, :3] = np.asarray(
        camera_param["rotation"],
        dtype=np.float64,
    )

    pose[:3, 3] = np.asarray(
        camera_param["position"],
        dtype=np.float64,
    )

    return pose


@dataclass
class GaussianScene:
    name: str
    gaussians: GaussianModel
    cameras: list[dict[str, Any]]


def load_camera_params(cameras_path: Path) -> list[dict]:
    cameras_path = Path(cameras_path)

    if not cameras_path.is_file():
        raise FileNotFoundError(
            f"Missing cameras.json: {cameras_path}"
        )

    with cameras_path.open("r") as f:
        cameras = json.load(f)

    if not cameras:
        raise ValueError(
            f"No cameras found in: {cameras_path}"
        )

    return cameras


def load_gaussian_scene(
    recon_root: Path,
    scene_name: str,
    sh_degree: int = 3,
) -> GaussianScene:
    """
    Expected layout:

        <recon_root>/<scene_name>/experiment/ply/
            cameras.json
            point_cloud/
                point_cloud.ply
    """

    scene_root = Path(recon_root) / scene_name
    ply_root = scene_root / "experiment" / "ply"

    point_cloud_path = (
        ply_root
        / "point_cloud"
        / "point_cloud.ply"
    )

    cameras_path = ply_root / "cameras.json"

    if not point_cloud_path.is_file():
        raise FileNotFoundError(
            f"Missing Gaussian point cloud: {point_cloud_path}"
        )

    cameras = load_camera_params(cameras_path)

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(str(point_cloud_path))

    return GaussianScene(
        name=scene_name,
        gaussians=gaussians,
        cameras=cameras,
    )


def make_minicam(camera_param: dict) -> MiniCam:
    """
    Convert one entry from cameras.json into LEG-SLAM's existing MiniCam.
    """

    width = int(camera_param["width"])
    height = int(camera_param["height"])

    fovx = focal2fov(
        float(camera_param["fx"]),
        width,
    )

    fovy = focal2fov(
        float(camera_param["fy"]),
        height,
    )

    world_view_transform = get_world2view(
        np.asarray(camera_param["rotation"]),
        np.asarray(camera_param["position"]),
    )

    return MiniCam(
        width,
        height,
        fovx,
        fovy,
        world_view_transform,
    )
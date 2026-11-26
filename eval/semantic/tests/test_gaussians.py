import json

import pytest
import torch

from eval.common.gaussian_scene import load_camera_params, make_minicam


def test_camera_loading(tmp_path):
    camera = {
        "img_name": "frame000000.jpg",
        "width": 640, "height": 480,
        "fx": 500.0, "fy": 500.0,
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "position": [0, 0, 0],
    }
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps([camera]))
    assert load_camera_params(path) == [camera]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by MiniCam")
def test_make_minicam():
    camera = {
        "width": 640, "height": 480,
        "fx": 500.0, "fy": 500.0,
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "position": [0, 0, 0],
    }
    cam = make_minicam(camera)
    assert cam.image_width == 640
    assert cam.image_height == 480

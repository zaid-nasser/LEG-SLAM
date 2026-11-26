"""Replica51 2D RADIOv4 PCA ablation (NO SLAM, NO Gaussian rendering).

Run at the LEG-SLAM repository root:

    python -m eval.semantic.evaluate_radio_2d_pca \
        --pca256 /workspace/encoder_models/YOUR_RADIO_256_COMPONENT_PCA.pt

The optional --pca256 argument overrides the path from cfg/feature_models.json.
The checkpoint must be the ORIGINAL TorchScript TorchPCA basis used to
compress RADIO features, with >=256 components. We use its first K principal
axes to make a strictly nested rank-K ablation, using a common PCA fit.

Uses all eight Replica scenes, every tenth frame, all 51 candidate classes,
the existing ReplicaDataset raw-ID mapping, Replica51 split definitions,
and exactly the existing SigLIP2-G text-prompt encoder/alignment head.

For each pixel, interpolate 2D RADIO backbone features to the RGB grid,
project and reconstruct for K=16,32,64,128,256, compare to uncompressed
features, then apply the SigLIP2-G head and cosine text matching.

Streaming the bilinear upsampling in chunks avoids materializing [1152,H,W].
Outputs summary.json, summary.csv, per-scene metrics and confusion matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval.common.torch_compat import (
    patch_torch_compiler_compat,
    patch_torch_hub_weights_only_compat,
    patch_torch_load_state_dict_compat,
)
from eval.semantic.datasets.replica import ReplicaDataset
from eval.semantic.metrics import compute_metrics, update_confusion
from eval.semantic.prompts import DEFAULT_PROMPT_SET, build_prompt_groups
from eval.semantic.taxonomies import REPLICA51_SPLITS, REPLICA51_TAXONOMY
from eval.semantic.predictors.radio_v4 import encode_radio_class_text_features


# One-purpose experiment: edit these if your dataset layout differs.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/workspace/data/Replica')
OUTPUT_ROOT = Path('/workspace/results/replica_radio_2d_pca_ablation')
REGISTRY_PATH = REPO_ROOT / 'cfg' / 'feature_models.json'
MODEL_ID = 'radio_v4_so400m'
SCENES = ('office0', 'office1','room0', 'room1')
STRIDE = 10
DIMS = (16, 32, 64, 128, 256, 'full_dim')
CHUNK_PIXELS = 8192  # Cap GPU memory for [pixels, 1152] feature matrices.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pca256', type=Path, default=None,
        help=('TorchScript TorchPCA checkpoint containing >=256 axes. '
              'Default: compressor path from cfg/feature_models.json, '
              'if that checkpoint actually contains 256 axes.'),
    )
    return parser.parse_args()


def get_model_config() -> dict:
    with REGISTRY_PATH.open('r') as f:
        registry = json.load(f)
    try:
        return registry['models'][MODEL_ID]
    except KeyError as exc:
        raise KeyError(f'Missing {MODEL_ID} in {REGISTRY_PATH}') from exc


def resolve_pca_path(cli_path: Path | None, model_cfg: dict) -> Path:
    if cli_path is not None:
        return cli_path.expanduser().resolve()
    path = Path(model_cfg['compressor']['path'])
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load_pca_256(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f'PCA checkpoint not found: {path}')
    module = torch.jit.load(str(path), map_location='cpu').eval()
    state = module.state_dict()
    if 'mean' not in state or 'components' not in state:
        raise RuntimeError(f'{path}: expected TorchPCA buffers mean and components.')
    mean = state['mean'].detach().float().reshape(-1)
    components = state['components'].detach().float()
    if components.ndim != 2 or components.shape[0] < 256:
        raise RuntimeError(
            f'{path}: requires >=256 PCA axes, got {tuple(components.shape)}. '
            'Supply a 256-axis TorchPCA via --pca256. A 64-axis checkpoint '
            'cannot be used to reconstruct 128/256 dimensions.'
        )
    if components.shape[1] != mean.numel():
        raise RuntimeError('TorchPCA mean/components dimension mismatch.')
    # Truncate the SAME ordered PCA basis for all rank-K experiments.
    return mean.to(device), components[:256].contiguous().to(device)


def make_radio(model_cfg: dict, device: torch.device):
    patch_torch_hub_weights_only_compat()
    patch_torch_compiler_compat()
    patch_torch_load_state_dict_compat()
    cfg = model_cfg.get('text_alignment', {})
    version = cfg.get('version', 'c-radio_v4-so400m')
    adaptor_name = cfg.get('adaptor', 'siglip2-g')
    model = torch.hub.load(
        'NVlabs/RADIO', 'radio_model', version=version,
        progress=True, skip_validation=True, trust_repo=True,
        adaptor_names=[adaptor_name],
    ).to(device).eval()
    adaptor = model.adaptors[adaptor_name]
    class_names = list(REPLICA51_TAXONOMY.class_names)
    text = encode_radio_class_text_features(
        adaptor=adaptor,
        prompt_groups=build_prompt_groups(class_names, DEFAULT_PROMPT_SET),
        device=device,
        expected_num_classes=len(class_names),
    )
    return model, adaptor, F.normalize(text.float(), dim=-1)


@torch.inference_mode()
def encode_image(image_bgr: np.ndarray, radio: torch.nn.Module,
                 device: torch.device) -> torch.Tensor:
    """Raw backbone RADIO features [1,1152,hp,wp], not the adaptor features."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    x = x.float().div_(255.0)
    h, w = x.shape[-2:]
    nearest = radio.get_nearest_supported_resolution(h, w)
    if tuple(nearest) != (h, w):
        x = F.interpolate(x, size=nearest, mode='bilinear', align_corners=False)
    output = radio(x, feature_fmt='NCHW')
    backbone = output['backbone'] if isinstance(output, dict) else output
    features = backbone.features if hasattr(backbone, 'features') else backbone[1]
    if features.ndim != 4 or features.shape[0] != 1:
        raise RuntimeError(f'Expected backbone spatial [1,D,hp,wp], got {features.shape}')
    return features.float()


@torch.inference_mode()
def iter_upsampled_features(feature_map: torch.Tensor, height: int, width: int):
    """Yield [N,D] pixel features equivalent to bilinear F.interpolate.

    The grid is defined at output-pixel centers with align_corners=False,
    and uses border padding, matching bilinear interpolate at boundaries.
    """
    n = height * width
    device = feature_map.device
    for start in range(0, n, CHUNK_PIXELS):
        end = min(start + CHUNK_PIXELS, n)
        idx = torch.arange(start, end, device=device)
        x = idx.remainder(width)
        y = torch.div(idx, width, rounding_mode='floor')
        grid = torch.stack(
            (2.0 * (x.float() + 0.5) / width - 1.0,
             2.0 * (y.float() + 0.5) / height - 1.0),
            dim=-1,
        ).reshape(1, -1, 1, 2)
        pixel_features = F.grid_sample(
            feature_map, grid, mode='bilinear',
            padding_mode='border', align_corners=False,
        )
        # [1,D,N,1] -> [N,D]; no full-resolution 1152-D map allocated.
        yield start, end, pixel_features.squeeze(0).squeeze(-1).T.contiguous()


@torch.inference_mode()
def classify(features: torch.Tensor, adaptor, text_embeddings: torch.Tensor) -> torch.Tensor:
    """Exactly the same RADIO SigLIP2-G score rule as RadioSiglipSemanticPredictor."""
    dtype = next(adaptor.head_mlp.parameters()).dtype
    aligned = adaptor.head_mlp(features.to(dtype)).float()
    aligned = F.normalize(aligned, dim=-1)
    return (aligned @ text_embeddings.T).argmax(dim=-1)


def json_metrics(metrics) -> dict:
    def safe(x):
        if x is None:
            return None
        value = float(x)
        return value if np.isfinite(value) else None
    parts = {}
    for name, row in metrics.split_metrics.items():
        parts[name] = {
            key: safe(v) if isinstance(v, (float, np.floating)) else int(v)
            for key, v in row.items()
        }
    return {'splits': parts, 'pixel_accuracy': safe(metrics.pixel_accuracy)}


@torch.inference_mode()
def evaluate_frame(image_bgr: np.ndarray, gt: np.ndarray,
                   radio, adaptor, text, mean, components, device):
    height, width = image_bgr.shape[:2]
    if gt.shape != (height, width):
        gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
    feature_map = encode_image(image_bgr, radio, device)
    if feature_map.shape[1] != mean.numel():
        raise RuntimeError(
            f'PCA/RADIO mismatch: backbone D={feature_map.shape[1]}, '
            f'checkpoint D={mean.numel()}.'
        )
    pred = {dim: np.empty(height * width, dtype=np.uint8) for dim in DIMS}
    cosine_sums = {dim: 0.0 for dim in DIMS}
    cosine_counts = {dim: 0 for dim in DIMS}

    for start, end, full in iter_upsampled_features(feature_map, height, width):
        centered = full - mean
        z_256 = centered @ components.T
        norms_original = torch.linalg.vector_norm(full, dim=-1)
        valid_original = torch.isfinite(full).all(dim=-1) & (norms_original > 1e-8)
        for dim in DIMS:
            if dim == 'full_dim':
                reconstructed = full
            else:
                reconstructed = z_256[:, :dim] @ components[:dim] + mean
            pred[dim][start:end] = classify(reconstructed, adaptor, text).byte().cpu().numpy()
            if dim == 'full_dim':
                # Identical original vectors: report exact cosine=1 for valid vectors.
                valid = valid_original
                cosine_sums[dim] += int(valid.sum().item())
                cosine_counts[dim] += int(valid.sum().item())
            else:
                norms_rec = torch.linalg.vector_norm(reconstructed, dim=-1)
                valid = valid_original & torch.isfinite(reconstructed).all(dim=-1) & (norms_rec > 1e-8)
                cosine = F.cosine_similarity(full, reconstructed, dim=-1)
                cosine_sums[dim] += float(cosine[valid].double().sum().item())
                cosine_counts[dim] += int(valid.sum().item())
    del feature_map
    return gt, {d: pred[d].reshape(height, width) for d in DIMS}, cosine_sums, cosine_counts


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This RADIOv4 ablation requires a CUDA device.')
    device = torch.device('cuda')
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f'Missing Replica dataset: {DATA_ROOT}')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model_cfg = get_model_config()
    pca_path = resolve_pca_path(args.pca256, model_cfg)
    print(f'Loading PCA: {pca_path}')
    mean, components = load_pca_256(pca_path, device)
    radio, adaptor, text = make_radio(model_cfg, device)
    data = ReplicaDataset(root=DATA_ROOT)
    num_classes = len(REPLICA51_TAXONOMY.class_names)
    if num_classes != 51:
        raise RuntimeError(f'Expected Replica51 classes, got {num_classes}')

    scene_results = []
    total_confusions = {d: np.zeros((51, 51), dtype=np.uint64) for d in DIMS}
    total_cos_sums = {d: 0.0 for d in DIMS}
    total_cos_counts = {d: 0 for d in DIMS}

    for scene in SCENES:
        frames = list(data.list_frames(scene, every_n_frame=STRIDE))
        if not frames:
            raise RuntimeError(f'No frames for {scene}')
        confusions = {d: np.zeros((51, 51), dtype=np.uint64) for d in DIMS}
        sums = {d: 0.0 for d in DIMS}
        counts = {d: 0 for d in DIMS}
        valid_pixels = 0
        for frame in tqdm(frames, desc=f'Replica 2D: {scene}', unit='frame'):
            image = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f'Cannot load RGB image: {frame.rgb_path}')
            gt_raw = data.load_semantic_gt(frame)
            gt, predictions, c_sums, c_counts = evaluate_frame(
                image, gt_raw, radio, adaptor, text, mean, components, device,
            )
            for d in DIMS:
                valid = update_confusion(
                    confusion=confusions[d], gt=gt, pred=predictions[d],
                    num_classes=num_classes,
                    ignore_index=REPLICA51_TAXONOMY.ignore_index,
                )
                if d == DIMS[0]:
                    valid_pixels += valid
                sums[d] += c_sums[d]
                counts[d] += c_counts[d]
        scene_summary = {'scene': scene, 'frames_used': len(frames), 'valid_pixels': valid_pixels,
                         'dimensions': {}}
        for d in DIMS:
            total_confusions[d] += confusions[d]
            total_cos_sums[d] += sums[d]
            total_cos_counts[d] += counts[d]
            score = compute_metrics(confusions[d], splits=REPLICA51_SPLITS)
            entry = json_metrics(score)
            entry['mean_cosine'] = sums[d] / counts[d] if counts[d] else None
            entry['cosine_vectors'] = counts[d]
            scene_summary['dimensions'][str(d)] = entry
            scene_out = OUTPUT_ROOT / scene / str(d)
            scene_out.mkdir(parents=True, exist_ok=True)
            np.save(scene_out / 'confusion.npy', confusions[d])
        scene_results.append(scene_summary)
        (OUTPUT_ROOT / scene / 'metrics.json').write_text(json.dumps(scene_summary, indent=2))
        print(f'Finished {scene}: {len(frames)} sampled RGB frames')

    summary = {'experiment': 'Replica51 raw 2D RADIOv4 PCA reconstruction + SigLIP2-G',
               'dataset_root': str(DATA_ROOT), 'pca_path': str(pca_path),
               'pca_strategy': 'single PCA fit, first K ordered components',
               'frame_stride': STRIDE, 'scenes': scene_results, 'dimensions': {}}
    rows = []
    for d in DIMS:
        pooled = json_metrics(compute_metrics(total_confusions[d], splits=REPLICA51_SPLITS))
        scene_mean = {}
        for metric in ('mIoU', 'mAcc', 'FmIoU'):
            valid = [r['dimensions'][str(d)]['splits']['all'][metric]
                     for r in scene_results
                     if r['dimensions'][str(d)]['splits']['all'][metric] is not None]
            scene_mean[metric] = float(np.mean(valid)) if valid else None
        cosine = total_cos_sums[d] / total_cos_counts[d] if total_cos_counts[d] else None
        summary['dimensions'][str(d)] = {
            'pooled': pooled, 'scene_mean': scene_mean,
            'mean_cosine': cosine, 'cosine_vectors': total_cos_counts[d],
        }
        np.save(OUTPUT_ROOT / f'confusion_pooled_{d}.npy', total_confusions[d])
        rows.append({
            'dimension': d, 'frames': sum(r['frames_used'] for r in scene_results),
            'cosine': cosine, 'pooled_mIoU': pooled['splits']['all']['mIoU'],
            'pooled_mAcc': pooled['splits']['all']['mAcc'],
            'pooled_FmIoU': pooled['splits']['all']['FmIoU'],
            'scene_mean_mIoU': scene_mean['mIoU'],
            'head_mIoU': pooled['splits']['head']['mIoU'],
            'common_mIoU': pooled['splits']['common']['mIoU'],
            'tail_mIoU': pooled['splits']['tail']['mIoU'],
        })
    (OUTPUT_ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (OUTPUT_ROOT / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('\nReplica51 RAW-2D PCA ablation')
    for row in rows:
        print(f"{str(row['dimension']):>8s} | cos={row['cosine']:.4f} | "
              f"pooled mIoU={row['pooled_mIoU']:.4f} | "
              f"mAcc={row['pooled_mAcc']:.4f} | FmIoU={row['pooled_FmIoU']:.4f}")
    print(f'\nSaved: {OUTPUT_ROOT / "summary.csv"}')


if __name__ == '__main__':
    main()

from __future__ import annotations

"""Render native-resolution, publication-ready LEG-SLAM qualitative frames.

Run at the LEG-SLAM repository root, with this file in tools/ and the existing
``tools/render_scene_video.py`` available (the supplied reference script).

Pipeline: FULL estimated TUM poses -> align to Gaussian world -> Gaussian RGB
and rendered compressed semantic features -> RADIOv4 dual Replica51 predictor
-> same optional connected-component refinement as semantic evaluation.

GT annotations recover the EXACT GT label-to-RGB lookup and GT color panel.
Optionally, class candidates can be GT-constrained per frame (default), per
scene, or the full Replica51 set. Irrespective of candidate mode, pixels with
undefined GT labels (ignore 255 or raw categories outside Replica51) and
pixels shown black in the GT color mask are ALWAYS blacked out in the exported
prediction, with predicted class ID set to 255. This is a GT-masked,
GT-constrained visualization, NOT a standard unrestricted semantic evaluation.
Unlike the video script, no synthetic palette fallback is allowed.
All output panels, masks, and PCA images are saved as PNG.

Color source (choose ONE):
  1) --gt-id-dir and --gt-color-dir: paired, pixel-aligned PNGs. By DEFAULT,
     GT IDs use REPLICA_RAW_CLASSES (raw dataset ordering). Each raw ID is
     translated by class NAME using RAW_ID_TO_DENSE51 before matching predicted
     Replica51 labels. Raw categories outside Replica51 are omitted; they must
     never shift or steal colors from the 51 evaluated categories. Use
     --gt-id-space replica51 only if GT labels have already been remapped.
  2) --gt-palette-json: a complete or sufficient {"class_id": [R,G,B]} table.
     Keys default to Replica51 indices; --gt-palette-id-space raw translates
     raw IDs to Replica51. --gt-id-dir is still required for per-pixel GT
     validity, and --gt-color-dir may additionally provide GT color panels.

Ignore predictions (ID 255) are shown in black by default, following the
semantic visualizer. Override with --ignore-rgb if needed; ignore is NOT a
Replica51 class, so an exact class-color comparison does not apply to it.

The optional GT images and label images are matched by the LAST numeric group
in each filename stem: frame000787.png, 787.png, frame_0787.png, etc.
Ambiguous/missing names are errors, rather than guesses.

From repository root:
  python -m tools.render_qualitative_images --help
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from eval.common.gaussian_scene import load_gaussian_scene
from eval.semantic.taxonomies import (
    REPLICA51_TAXONOMY,
    REPLICA_RAW_CLASSES,
    RAW_ID_TO_DENSE51,
    DENSE51_TO_RAW_ID,
)
from tools.render_scene_video import (
    build_full_views,
    build_replica51_dual_predictor,
    fit_pca_from_rendered_views,
    feature_visualization,
    render_one,
    rgb_from_render,
)
from eval.semantic.evaluator import refine_label_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--result', required=True, type=Path,
                   help='Experiment result directory, as for render_scene_video.py')
    p.add_argument('--sequence', required=True, type=Path,
                   help='Input sequence containing the original RGB frames')
    p.add_argument('--dataset', choices=('replica',), default='replica')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--frames', nargs='+', type=int, default=[787, 1190, 1476, 1980])
    p.add_argument('--trajectory', type=Path, default=None,
                   help='Default: <result>/CameraTrajectory_TUM.txt; never uses GT poses')

    # Exact GT colors, not a generated palette.
    p.add_argument('--gt-id-dir', type=Path, default=None,
                   help='Single-channel GT label PNGs; raw Replica IDs by default')
    p.add_argument('--gt-id-space', choices=('raw', 'replica51'), default='raw',
                   help='GT PNG ID coding: raw REPLICA_RAW_CLASSES (default) or already-remapped Replica51')
    p.add_argument('--gt-color-dir', type=Path, default=None,
                   help='Already-colored GT segmentation PNGs (exact desired RGB)')
    p.add_argument('--gt-palette-json', type=Path, default=None,
                   help='Alternative to paired GT: JSON {"0":[R,G,B], ...}')
    p.add_argument('--gt-palette-id-space', choices=('raw', 'replica51'), default='replica51',
                   help='Class ID coding used by the JSON palette (default: Replica51)')
    p.add_argument('--ignore-rgb', nargs=3, type=int, default=(0, 0, 0),
                   metavar=('R', 'G', 'B'),
                   help='RGB for predictions with ignore ID 255 (default: black, as evaluator display)')
    p.add_argument('--palette-sample-frames', type=int, default=32,
                   help='Reference paired GT views (plus requested frames) for class color coverage; 0=all')
    p.add_argument('--candidate-classes-source', choices=('frame_gt', 'scene_gt', 'all51'),
                   default='frame_gt',
                   help='Predict using Replica51 classes present in EACH frame GT (default), '
                        'classes in the recovered scene GT palette, or all 51 classes. '
                        'GT-constrained modes are for visualization, NOT standard evaluation.')

    # SAME predictor defaults as supplied tools/render_scene_video.py.
    p.add_argument('--feature-model', default='radio_v4_so400m')
    p.add_argument('--feature-registry', type=Path, default=Path('cfg/feature_models.json'))
    p.add_argument('--probe-checkpoint', type=Path, required=True)
    p.add_argument('--prototype-file', type=Path, required=True)
    p.add_argument('--fusion-beta', type=float, default=1.0)
    p.add_argument('--fusion-tau-ov', type=float, default=1.0)
    p.add_argument('--fusion-tau-cs', type=float, default=1.0)
    p.add_argument('--semantic-z-threshold', type=float, default=2.2)
    p.add_argument('--radio-chunk-size', type=int, default=32768)
    p.add_argument('--semantic-refinement', choices=('components', 'none'), default='components')

    p.add_argument('--pca-sample-views', type=int, default=32,
                   help='Evenly spaced tracked scene views used to fit a shared PCA basis')
    p.add_argument('--pca-pixels-per-view', type=int, default=2048)
    p.add_argument('--figure-panel-width', type=int, default=480,
                   help='Width of each panel in the optional contact sheet; raw PNGs are native res')
    return p.parse_args()


def image_index(directory: Path) -> dict[int, Path]:
    """Map final numeric sequence of a PNG filename to a frame ID; no ambiguity."""
    if not directory.is_dir():
        raise FileNotFoundError(f'GT directory not found: {directory}')
    index: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != '.png':
            continue
        m = re.search(r'(\d+)(?!.*\d)', path.stem)
        if m is None:
            continue
        frame_id = int(m.group(1))
        if frame_id in index:
            raise ValueError(f'Ambiguous GT frame {frame_id}: {index[frame_id]} and {path}')
        index[frame_id] = path
    if not index:
        raise ValueError(f'No numbered PNG images found in {directory}')
    return index


def read_color(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f'Unable to open colored GT image: {path}')
    if bgr.ndim != 3 or bgr.shape[2] not in (3, 4) or bgr.dtype != np.uint8:
        raise ValueError(f'Expected 8-bit RGB/RGBA colored GT PNG: {path}, got {bgr.shape}/{bgr.dtype}')
    if bgr.shape[2] == 4:
        if not np.all(bgr[:, :, 3] == 255):
            raise ValueError(f'GT contains transparency; exact RGB cannot be inferred: {path}')
        bgr = bgr[:, :, :3]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def validate_taxonomy_mapping() -> None:
    """Fail loudly if the repository taxonomy differs from the expected bijection."""
    names = REPLICA51_TAXONOMY.class_names
    if len(names) != 51 or len(DENSE51_TO_RAW_ID) != 51:
        raise ValueError('Expected exactly 51 classes and inverse raw-ID entries.')
    for dense_id, (name, raw_id) in enumerate(zip(names, DENSE51_TO_RAW_ID)):
        if (not 0 <= raw_id < len(REPLICA_RAW_CLASSES)
                or REPLICA_RAW_CLASSES[raw_id] != name
                or RAW_ID_TO_DENSE51.get(raw_id) != dense_id):
            raise ValueError(f'Inconsistent Replica raw/dense mapping for {name!r} at {dense_id}')


def read_ids(path: Path, id_space: str = 'raw') -> np.ndarray:
    """Read GT integer class IDs without assuming they equal prediction IDs."""
    ids = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if ids is None:
        raise RuntimeError(f'Unable to open GT IDs: {path}')
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f'GT class IDs must be SINGLE-CHANNEL integer PNG: {path}, got {ids.shape}')
    if id_space == 'raw':
        valid = set(range(len(REPLICA_RAW_CLASSES))) | {255}
    elif id_space == 'replica51':
        valid = set(range(len(REPLICA51_TAXONOMY.class_names))) | {255}
    else:
        raise ValueError(f'Unsupported GT class-ID space: {id_space}')
    unexpected = set(np.unique(ids).astype(int).tolist()) - valid
    if unexpected:
        raise ValueError(f'{path}: unexpected {id_space} GT IDs {sorted(unexpected)[:15]}. '
                         'Check the input dataset and --gt-id-space.')
    return ids


def gt_id_to_dense51(raw_or_dense_id: int, id_space: str) -> int | None:
    """Translate an annotated GT class to the predictor index by class name.

    Raw 'other', classes not in Replica51, and ignore=255 have no 51-class
    prediction counterpart. They are intentionally skipped when deriving the
    class palette; they are never treated as the same numeric dense ID.
    """
    if raw_or_dense_id == 255:
        return None
    if id_space == 'raw':
        return RAW_ID_TO_DENSE51.get(raw_or_dense_id)
    if id_space == 'replica51':
        return raw_or_dense_id if 0 <= raw_or_dense_id < 51 else None
    raise ValueError(f'Unsupported GT class-ID space: {id_space}')


def frame_gt_candidates(gt_ids: np.ndarray, id_space: str) -> list[int]:
    """Get globally indexed Replica51 candidate IDs from ONE frame's GT.

    Ignore ID 255 and raw Replica classes outside the 51-class taxonomy.
    No pixelwise GT labels are used by inference, only the set of class names.
    """
    ids = {
        gt_id_to_dense51(int(raw_id), id_space)
        for raw_id in np.unique(gt_ids)
    }
    ids.discard(None)
    return sorted(ids)



def gt_undefined_pixels(gt_ids: np.ndarray, id_space: str,
                        gt_rgb: np.ndarray | None = None) -> np.ndarray:
    """True for pixels without a defined Replica51 GT annotation.

    A raw Replica annotation may use ID 255, 'other' (raw ID 0), or any
    category excluded from Replica51. Dense Replica51 annotations use 255
    as ignore. An explicitly black GT visualization pixel is also treated
    as undefined, even if an inconsistent ID PNG assigns it a class.
    """
    if gt_ids.ndim != 2:
        raise ValueError(f'GT IDs must be HxW; got {gt_ids.shape}')
    if id_space == 'raw':
        valid = np.isin(gt_ids, list(RAW_ID_TO_DENSE51))
    elif id_space == 'replica51':
        valid = (gt_ids >= 0) & (gt_ids < len(REPLICA51_TAXONOMY.class_names))
    else:
        raise ValueError(f'Unsupported GT ID space: {id_space}')
    undefined = ~valid
    if gt_rgb is not None:
        if gt_rgb.shape != (*gt_ids.shape, 3):
            raise ValueError(f'GT RGB/ID shape mismatch: {gt_rgb.shape} vs {gt_ids.shape}')
        undefined |= np.all(gt_rgb == 0, axis=2)
    return undefined


def palette_from_json(
    path: Path, id_space: str = 'replica51',
) -> dict[int, tuple[int, int, int]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('GT palette JSON must be an object {"class_id": [R, G, B]}')
    palette: dict[int, tuple[int, int, int]] = {}
    for key, value in data.items():
        try:
            source_id = int(key)
        except ValueError as e:
            raise ValueError(f'GT palette key must be a numeric class ID: {key!r}') from e
        if not isinstance(value, list) or len(value) != 3 or any(
            not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 255
            for v in value
        ):
            raise ValueError(f'Palette value for class {key} must be [R,G,B] (integer 0..255)')
        if source_id != 255 and not (0 <= source_id < (
                len(REPLICA_RAW_CLASSES) if id_space == 'raw' else 51)):
            raise ValueError(f'GT palette ID {source_id} is invalid for {id_space} coding.')
        dense_id = gt_id_to_dense51(source_id, id_space)
        if dense_id is None:
            continue  # Other raw categories do not enter 51-class predictions.
        color = tuple(value)
        if dense_id in palette and palette[dense_id] != color:
            raise ValueError(f'Conflicting palette RGB for dense label {dense_id}')
        palette[dense_id] = color
    if not palette:
        raise ValueError('No Replica51 classes found in the GT palette JSON')
    print(f'[GT] Loaded exact colors for {len(palette)} Replica51 classes '
          f'from {id_space}-indexed palette JSON')
    return palette


def palette_from_gt_pairs(
    id_paths: dict[int, Path], color_paths: dict[int, Path],
    requested_frames: list[int], sample_count: int, id_space: str = 'raw',
) -> dict[int, tuple[int, int, int]]:
    common = sorted(id_paths.keys() & color_paths.keys())
    if not common:
        raise RuntimeError('GT ID and color directories have no matching numbered PNG files')
    if sample_count > 0:
        reference = [common[i] for i in np.unique(np.linspace(
            0, len(common) - 1, min(len(common), sample_count), dtype=int))]
        reference = sorted(set(reference) | (set(requested_frames) & set(common)))
    else:
        reference = common
    palette: dict[int, tuple[int, int, int]] = {}
    ignored_raw_ids: set[int] = set()
    for frame_id in tqdm(reference, desc='Recovering exact GT colors'):
        ids, rgb = read_ids(id_paths[frame_id], id_space), read_color(color_paths[frame_id])
        if ids.shape != rgb.shape[:2]:
            raise ValueError(f'GT ID/color shape mismatch at frame {frame_id}: {ids.shape}, {rgb.shape}')
        for gt_label in np.unique(ids):
            gt_label = int(gt_label)
            dense_id = gt_id_to_dense51(gt_label, id_space)
            if dense_id is None:
                ignored_raw_ids.add(gt_label)
                continue
            pixels = rgb[ids == gt_label]
            colors, counts = np.unique(pixels, axis=0, return_counts=True)
            winner = int(np.argmax(counts))
            dominant_fraction = float(counts[winner] / len(pixels))
            if dominant_fraction < 0.995:
                raise ValueError(
                    f'GT {id_space} ID {gt_label} ({REPLICA51_TAXONOMY.class_names[dense_id]}) '
                    f'maps to multiple colors at frame {frame_id} '
                    f'(most common {dominant_fraction:.2%}). '
                    'GT IDs and colors must be aligned, uncropped, non-antialiased PNGs.'
                )
            color = tuple(int(v) for v in colors[winner])
            if dense_id in palette and palette[dense_id] != color:
                name = REPLICA51_TAXONOMY.class_names[dense_id]
                raise ValueError(f'Inconsistent GT color for {name} (Replica51 ID {dense_id}): '
                                 f'{palette[dense_id]} vs {color} in frame {frame_id}')
            palette[dense_id] = color
    if not palette:
        raise RuntimeError('No GT classes belonging to Replica51 found in paired annotations')
    print(f'[GT] Recovered exact RGB colors for {len(palette)} Replica51 classes '
          f'from {len(reference)} views; skipped {len(ignored_raw_ids)} raw/non-evaluated IDs')
    return palette


def colorize_exact(
    labels: np.ndarray, palette: dict[int, tuple[int, int, int]],
    ignore_rgb: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Class colors match GT exactly; ignore pixels have a separate display color."""
    if labels.ndim != 2:
        raise ValueError(f'Labels must be HxW; got {labels.shape}')
    present = [int(x) for x in np.unique(labels)]
    if any(x not in range(51) and x != 255 for x in present):
        raise ValueError(f'Unexpected predicted Replica51 labels: {present}')
    missing = [idx for idx in present if idx != 255 and idx not in palette]
    if missing:
        names = REPLICA51_TAXONOMY.class_names
        details = ', '.join(f'{idx} ({names[idx]})' for idx in missing)
        raise ValueError('Predicted labels absent from exact GT palette: ' + details + '. '
                         'Provide more paired GT reference views (--palette-sample-frames 0) '
                         'or --gt-palette-json with the missing RGB entries. No class colors were fabricated.')
    rgb = np.empty((*labels.shape, 3), dtype=np.uint8)
    for label in present:
        rgb[labels == label] = ignore_rgb if label == 255 else palette[label]
    return rgb


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f'Expected uint8 RGB image, got {rgb.shape}/{rgb.dtype}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f'Could not write {path}')


def panel(rgb: np.ndarray, width: int, title: str) -> np.ndarray:
    native_h, native_w = rgb.shape[:2]
    height = round(width * native_h / native_w)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST if title in ('Ground truth', 'Prediction') else cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    label_strip = np.full((35, width, 3), 255, dtype=np.uint8)
    cv2.putText(label_strip, title, (9, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack((label_strip, bgr))


def main() -> None:
    args = parse_args()
    frames = list(dict.fromkeys(args.frames))
    if not frames or min(frames) < 0:
        raise ValueError('--frames must be nonempty and nonnegative (zero-indexed).')
    if args.pca_sample_views <= 0 or args.pca_pixels_per_view <= 0 or args.figure_panel_width < 64:
        raise ValueError('PCA sample counts and figure-panel-width must be positive.')
    if args.palette_sample_frames < 0:
        raise ValueError('--palette-sample-frames must be >= 0.')
    validate_taxonomy_mapping()
    if any(c < 0 or c > 255 for c in args.ignore_rgb):
        raise ValueError('--ignore-rgb requires three integers from 0 to 255')
    if not args.gt_id_dir:
        raise ValueError('--gt-id-dir is required: per-frame GT annotations determine '
                         'which prediction pixels must be marked undefined (black).')
    if not args.gt_palette_json and not args.gt_color_dir:
        raise ValueError('Provide --gt-color-dir for exact GT class colors, '
                         'or --gt-palette-json along with --gt-id-dir.')
    if not torch.cuda.is_available():
        raise RuntimeError('The LEG-SLAM Gaussian rasterizer and RADIO predictor require CUDA.')

    output = args.output_dir.resolve()
    result = args.result.resolve()
    sequence = args.sequence.resolve()
    output.mkdir(parents=True, exist_ok=True)

    id_paths = image_index(args.gt_id_dir) if args.gt_id_dir else {}
    color_paths = image_index(args.gt_color_dir) if args.gt_color_dir else {}
    if args.gt_palette_json:
        palette = palette_from_json(args.gt_palette_json, args.gt_palette_id_space)
    else:
        palette = palette_from_gt_pairs(id_paths, color_paths, frames,
                                        args.palette_sample_frames, args.gt_id_space)

    # Never guess missing GT images when user asked to compare against GT.
    if args.gt_color_dir:
        missing_gt = set(frames) - color_paths.keys()
        if missing_gt:
            raise FileNotFoundError(f'Missing colored GT images for frames: {sorted(missing_gt)}')
    missing_ids = set(frames) - id_paths.keys()
    if missing_ids:
        raise FileNotFoundError(f'Missing GT class-ID images required for '
                                f'per-pixel ignore masking: {sorted(missing_ids)}')

    scene = load_gaussian_scene(result.parent, result.name)
    all_views = build_full_views(scene, sequence, args.dataset, result, args.trajectory)
    view_by_frame = {int(cam['id']): (cam, rgb_path) for cam, rgb_path in all_views}
    missing_poses = [i for i in frames if i not in view_by_frame]
    if missing_poses:
        raise RuntimeError(f'No saved ESTIMATED tracked poses for frames {missing_poses}. '
                           'Check CameraTrajectory_TUM.txt and input RGB ordering; no GT-pose substitution.')

    predictor = build_replica51_dual_predictor(args)
    if args.candidate_classes_source == 'scene_gt':
        fixed_candidate_ids = sorted(int(i) for i in palette.keys())
        if not fixed_candidate_ids:
            raise RuntimeError('Scene-GT restriction requested, but the GT palette is empty.')
        names = [REPLICA51_TAXONOMY.class_names[i] for i in fixed_candidate_ids]
        print(f'[Semantic] Scene-GT candidates ({len(fixed_candidate_ids)}): {names}')
        if args.palette_sample_frames > 0 and not args.gt_palette_json:
            print('[Semantic] Scene-GT candidates come from sampled GT frames; '
                  'use --palette-sample-frames 0 for complete scene coverage.')
    elif args.candidate_classes_source == 'all51':
        fixed_candidate_ids = list(range(len(REPLICA51_TAXONOMY.class_names)))
        print('[Semantic] Using all 51 Replica51 classes')
    else:
        fixed_candidate_ids = None
        print('[Semantic] Frame-GT candidates: each frame uses ONLY its own GT-present '
              'Replica51 classes. This is GT-constrained visualization.')
    background = torch.zeros(3, dtype=torch.float32, device='cuda')
    mean, basis, lower, upper, explained = fit_pca_from_rendered_views(
        [cam for cam, _ in all_views], scene, background,
        args.pca_sample_views, args.pca_pixels_per_view,
    )
    np.savez_compressed(output / 'rendered_features_pca_transform.npz', mean=mean,
                        basis=basis, lower=lower, upper=upper,
                        explained_variance_ratio=explained)
    (output / 'gt_palette_rgb.json').write_text(
        json.dumps({str(idx): list(c) for idx, c in sorted(palette.items())}, indent=2), encoding='utf-8')
    (output / 'classes.json').write_text(
        json.dumps({str(i): name for i, name in enumerate(REPLICA51_TAXONOMY.class_names)}, indent=2),
        encoding='utf-8')

    grid_rows = []
    per_frame_candidate_ids: dict[str, list[int]] = {}
    per_frame_undefined_pixels: dict[str, int] = {}
    for frame_id in tqdm(frames, desc='Exporting qualitative frames'):
        cam, input_path = view_by_frame[frame_id]
        width, height = int(cam['width']), int(cam['height'])
        filename = f'frame_{frame_id:04d}.png'

        # GT validity is checked for EVERY prediction mode, not only frame_gt.
        frame_ids = read_ids(id_paths[frame_id], args.gt_id_space)
        if frame_ids.shape != (height, width):
            raise ValueError(f'Frame {frame_id}: GT IDs shape {frame_ids.shape} '
                             f'differs from camera image {(height, width)}')
        gt_rgb = read_color(color_paths[frame_id]) if args.gt_color_dir else None
        if gt_rgb is not None and gt_rgb.shape[:2] != (height, width):
            raise ValueError(f'Frame {frame_id}: GT colors shape {gt_rgb.shape[:2]} '
                             f'differs from camera image {(height, width)}')
        undefined_gt = gt_undefined_pixels(frame_ids, args.gt_id_space, gt_rgb)
        per_frame_undefined_pixels[str(frame_id)] = int(undefined_gt.sum())

        if args.candidate_classes_source == 'frame_gt':
            candidate_ids = frame_gt_candidates(frame_ids, args.gt_id_space)
            if not candidate_ids:
                raise RuntimeError(f'Frame {frame_id} contains no GT classes belonging '
                                   'to Replica51; cannot make a constrained prediction.')
        else:
            candidate_ids = fixed_candidate_ids
        per_frame_candidate_ids[str(frame_id)] = candidate_ids
        names = [REPLICA51_TAXONOMY.class_names[idx] for idx in candidate_ids]
        print(f'[Frame {frame_id}] {len(candidate_ids)} candidate classes: {names}')

        # GT-constrained candidate sets must all have an exact GT color.
        # In all51 mode, only classes actually PREDICTED need colors, as before.
        if args.candidate_classes_source != 'all51':
            absent_colors = [idx for idx in candidate_ids if idx not in palette]
            if absent_colors:
                raise RuntimeError(f'Frame {frame_id}: no GT RGB colors for candidate '
                                   f'classes {absent_colors}. Provide full GT palette.')

        original_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if original_bgr is None:
            raise FileNotFoundError(f'Could not load RGB image: {input_path}')
        if original_bgr.shape[:2] != (height, width):
            raise ValueError(f'Input dimensions {original_bgr.shape[:2]} do not match camera {(height, width)}')
        input_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        rendered = render_one(cam, scene, background)
        rendered_rgb = cv2.cvtColor(
            rgb_from_render(rendered['rendered_image'], (width, height)), cv2.COLOR_BGR2RGB)
        pca_rgb = cv2.cvtColor(
            feature_visualization(rendered['rendered_lf'], (width, height),
                                  mean, basis, lower, upper), cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            prediction = predictor.predict(rendered['rendered_lf'], candidate_ids=candidate_ids)
        labels = prediction.labels.detach().cpu().numpy().astype(np.uint8)
        bad_labels = set(np.unique(labels).astype(int)) - set(candidate_ids) - {REPLICA51_TAXONOMY.ignore_index}
        if bad_labels:
            raise RuntimeError(
                f'Frame {frame_id}: predictor returned labels outside the allowed '
                f'Replica51 candidate set: {sorted(bad_labels)}. Verify the '
                'predictor returns GLOBAL Replica51 IDs for subset candidate_ids.'
            )
        if labels.shape != (height, width):
            raise ValueError(f'Predicted label dimensions {labels.shape} differ from rendered view {(height, width)}; '
                             'do not resize predictions or change evaluation semantics implicitly.')
        if args.semantic_refinement == 'components':
            labels = refine_label_image(
                labels, min_area=600, max_fill_area=5000,
                ignore_index=REPLICA51_TAXONOMY.ignore_index)
        # Fixed visualization policy: undefined GT pixels may not display any
        # predicted class, regardless of the predictor's candidate mode.
        labels[undefined_gt] = REPLICA51_TAXONOMY.ignore_index
        pred_rgb = colorize_exact(labels, palette, tuple(args.ignore_rgb))
        pred_rgb[undefined_gt] = 0  # black, even if --ignore-rgb was customized

        label_path = output / 'pred_ids' / filename
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(label_path), labels):
            raise IOError(f'Could not write class IDs: {label_path}')
        save_rgb(output / 'input_rgb' / filename, input_rgb)
        save_rgb(output / 'rendered_rgb' / filename, rendered_rgb)
        save_rgb(output / 'pca_rgb' / filename, pca_rgb)
        save_rgb(output / 'pred_gt_colors' / filename, pred_rgb)

        if gt_rgb is not None:
            save_rgb(output / 'gt_rgb' / filename, gt_rgb)

        row = [panel(input_rgb, args.figure_panel_width, 'Input RGB'),
               panel(rendered_rgb, args.figure_panel_width, 'Rendered RGB')]
        if gt_rgb is not None:
            row.append(panel(gt_rgb, args.figure_panel_width, 'Ground truth'))
        row += [panel(pred_rgb, args.figure_panel_width, 'Prediction'),
                panel(pca_rgb, args.figure_panel_width, 'Rendered feature PCA')]
        grid_rows.append(np.hstack(row))
        del rendered, prediction
        print(f'[Frame {frame_id}] GT undefined pixels blacked out: '
              f'{per_frame_undefined_pixels[str(frame_id)]} '
              f'({100.0 * np.mean(undefined_gt):.2f}%); saved native-resolution outputs')

    figure = np.vstack(grid_rows)
    figure_path = output / 'qualitative_grid.png'
    if not cv2.imwrite(str(figure_path), figure):
        raise IOError(f'Could not write contact sheet: {figure_path}')
    manifest = {
        'frames': frames,
        'dataset': args.dataset,
        'pose_source': str(args.trajectory or (result / 'CameraTrajectory_TUM.txt')),
        'pose_type': 'estimated tracked camera pose, aligned to Gaussian map',
        'prediction_input': 'rendered Gaussian compressed language features',
        'semantic_protocol': 'Replica51 RADIOv4 dual, configurable candidate class set, optional evaluation component refinement',
        'semantic_refinement': args.semantic_refinement,
        'candidate_classes_source': args.candidate_classes_source,
        'candidate_ids_by_frame': per_frame_candidate_ids,
        'gt_undefined_pixels_by_frame': per_frame_undefined_pixels,
        'gt_undefined_prediction_policy': ('After semantic prediction/refinement, mask '
            'GT ignore 255, raw GT IDs outside Replica51 (including other), and '
            'black GT color pixels; save predicted IDs as 255 and colors as black.'),
        'candidate_class_names_by_frame': {
            frame: [REPLICA51_TAXONOMY.class_names[i] for i in ids]
            for frame, ids in per_frame_candidate_ids.items()
        },
        'gt_conditioned_class_candidates': args.candidate_classes_source != 'all51',
        'evaluation_note': ('Per-frame GT-defined-pixel masking is always applied; '
                            'frame/scene GT candidate restriction is also optionally applied. '
                            'These are GT-conditioned qualitative visualizations, not '
                            'standard open-vocabulary evaluation.'),
        'feature_model': args.feature_model,
        'probe_checkpoint': str(args.probe_checkpoint),
        'prototype_file': str(args.prototype_file),
        'fusion_beta': args.fusion_beta,
        'fusion_tau_ov': args.fusion_tau_ov,
        'fusion_tau_cs': args.fusion_tau_cs,
        'semantic_z_threshold': args.semantic_z_threshold,
        'pca_method': 'top-3 PCA on scene-sampled rendered compressed features, shared across all frames',
        'pca_sample_views': args.pca_sample_views,
        'pca_pixels_per_view': args.pca_pixels_per_view,
        'color_source': str(args.gt_palette_json or args.gt_color_dir),
        'gt_id_space': args.gt_id_space if args.gt_id_dir else None,
        'gt_palette_id_space': args.gt_palette_id_space if args.gt_palette_json else None,
        'gt_color_mapping': 'RAW_ID_TO_DENSE51 by identical class name, with non-Replica51 raw classes omitted',
        'prediction_ignore_rgb': args.ignore_rgb,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nComplete: {figure_path}\nRaw PNGs: {output / "pred_gt_colors"}, {output / "pca_rgb"}')


if __name__ == '__main__':
    main()

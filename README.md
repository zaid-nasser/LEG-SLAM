# LEG-SLAM

LEG-SLAM is a real-time RGB-D SLAM system that reconstructs a Gaussian scene while storing semantic visual features for later semantic querying and evaluation.

This README describes installation and usage.

---

## 1. Installation

LEG-SLAM is intended to be run inside the provided Docker environment.  
The current build targets Blackwell (CUDA 12.8, SM 120). Other GPU targets require separate validation.

### 1.1 Clone the repository

```bash
git clone https://github.com/Titrom025/legs-slam.git
cd legs-slam
```

### 1.2 Build the Docker image

From the repository root:

```bash
docker compose -f docker-compose.blackwell.yml build
```

The first build may take some time because the image contains the dependencies required by LEG-SLAM, ORB-SLAM3, Gaussian Splatting, and the semantic evaluation pipeline.

### 1.3 Start the container

Start the Docker services:

```bash
docker compose -f docker-compose.blackwell.yml up -d
```

Open a shell inside the LEG-SLAM service:

```bash
docker compose -f docker-compose.blackwell.yml exec legs-slam-api bash
```

All commands in the rest of this README are intended to be run **inside the Docker container**.

```bash
cd /workspace
```

### 1.4 Verify the installation

```bash
python -m tools.slam.run --help
python -m eval --help
python -m eval semantic --help
python -m eval reconstruction --help
```

---

## 2. Dataset layout

All dataset paths shown below are paths **inside the Docker container**.

### 2.1 Replica

Expected root:

```text
/workspace/data/Replica/
```

Typical scene layout:

```text
/workspace/data/Replica/office0/
    results/
        frame000000.jpg
        depth000000.png
        ...
    semantic_ids/
        semantic_id000000.png
        ...
```

Standard Replica scenes:

```text
office0 office1 office2 office3 office4 room0 room1 room2
```

### 2.2 ScanNet

Expected root:

```text
/workspace/data/scannet/
```

Example:

```text
/workspace/data/scannet/scene0378_01/
```

Semantic evaluation also uses:

```text
eval/scannetv2-labels.modified.tsv
```

Current ScanNet evaluation scenes:

```text
scene0011_00
scene0011_01
scene0050_00
scene0050_01
scene0050_02
scene0231_00
scene0231_01
scene0231_02
scene0378_00
scene0378_01
scene0378_02
scene0518_00
```

---

## 3. Running LEG-SLAM

The unified reconstruction entry point is:

```bash
python -m tools.slam.run
```

A run requires:

- dataset type
- sequence path
- output path
- ORB-SLAM3 configuration
- feature model registry (defaults to `cfg/feature_models.json`)
- Gaussian mapper configuration
- feature model ID

### 3.1 ScanNet example

```bash
python -m tools.slam.run \
    --dataset scannet \
    --sequence /workspace/data/scannet/scene0378_01 \
    --output /workspace/results/results_scannet_radiov4_128/scene0378_01 \
    --orb-config cfg/ORB_SLAM3/RGB-D/ScanNet/scene0378_01.yaml \
    --feature-registry cfg/feature_models.json \
    --gaussian-config cfg/gaussian_mapper/RGB-D/ScanNet/scannet.yaml \
    --feature-model radio_v4_so400m
```

### 3.2 Replica example

```bash
python -m tools.slam.run \
    --dataset replica \
    --sequence /workspace/data/Replica/office0 \
    --output /workspace/results/results_replica_radiov4_128/office0 \
    --orb-config cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    --feature-registry cfg/feature_models.json \
    --gaussian-config cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    --feature-model radio_v4_so400m
```

---

## 4. Reconstruction output

A completed reconstruction is expected to contain:

```text
<recon-root>/<scene>/
    experiment/
        ply/
            cameras.json
            point_cloud/
                point_cloud.ply
```

These files are consumed by the evaluation tools.

---

## 5. Semantic evaluation

Run semantic evaluation with:

```bash
python -m eval semantic
```

Add:

```bash
--save-masks
```

to save qualitative prediction and ground-truth masks.

Typical output:

```text
<out-root>/<scene>/
    pred_ids/
    pred_vis/
    gt_ids/
    gt_vis/
    classes.json
    confusion.npy
    metrics.json
```

### 5.1 ScanNet semantic evaluation

```bash
python -m eval semantic \
    --protocol scannet12_20 \
    --scenes scene0011_00 scene0011_01 scene0050_00 scene0050_01 scene0050_02 scene0231_00 scene0231_01 scene0231_02 scene0378_00 scene0378_01 scene0378_02 scene0518_00 \
    --data-root /workspace/data/scannet \
    --recon-root /workspace/results/results_scannet_radiov4 \
    --feature-model radio_v4_so400m \
    --feature-registry cfg/feature_models.json \
    --scannet-label-file eval/scannetv2-labels.modified.tsv \
    --out-root /workspace/results/results_scannet_radiov4/semantic_eval_results/all_scenes \
    --save-masks
```

### 5.2 Replica-51 with RADIO

```bash
python -m eval semantic \
    --protocol replica51 \
    --scenes office0 office1 office2 office3 office4 room0 room1 room2 \
    --data-root /workspace/data/Replica \
    --recon-root /workspace/results/results_replica_radiov4 \
    --feature-model radio_v4_so400m \
    --feature-registry cfg/feature_models.json \
    --out-root /workspace/results/results_replica_radiov4/semantic_eval_results/all_scenes \
    --save-masks
```

### 5.3 Replica-51 with depth-affinity refinement

```bash
python -m eval semantic \
    --protocol replica51 \
    --scenes office0 office1 office2 office3 office4 room0 room1 room2 \
    --data-root /workspace/data/Replica \
    --recon-root /workspace/results/results_replica_radiov4 \
    --feature-model radio_v4_so400m \
    --feature-registry cfg/feature_models.json \
    --out-root /workspace/results/results_replica_radiov4/semantic_eval_results/all_scenes_depth_refine \
    --semantic-refinement iterative \
    --affinity-source depth \
    --affinity-dilations 1 2 4 \
    --refine-iterations 50 \
    --refine-lambda 3.0 \
    --depth-sigma 0.05 \
    --save-masks
```

### 5.4 Indoor37 closed-set RADIO probe

```bash
python -m eval semantic \
    --protocol indoor37_replica \
    --scenes office0 \
    --data-root /workspace/data/Replica \
    --recon-root /workspace/results/results_replica_radiov4 \
    --feature-model radio_v4_so400m \
    --feature-registry cfg/feature_models.json \
    --probe-checkpoint /workspace/encoder_models/radio_indoor37_linear_probe.pt \
    --out-root /workspace/results/results_replica_radiov4/semantic_eval_results/indoor37_probe_smoke \
    --frame-stride 100 \
    --semantic-refinement none \
    --save-masks
```

---

## 6. Reconstruction evaluation

Run reconstruction evaluation with:

```bash
python -m eval reconstruction
```

### 6.1 ScanNet

```bash
python -m eval reconstruction \
    --protocol scannet12 \
    --scenes scene0011_00 scene0011_01 scene0050_00 scene0050_01 scene0050_02 scene0231_00 scene0231_01 scene0231_02 scene0378_00 scene0378_01 scene0378_02 scene0518_00 \
    --data-root /workspace/data/scannet \
    --recon-root /workspace/results/results_scannet_radiov4 \
    --out-root /workspace/results/results_scannet_radiov4/reconstruction_eval_results/all_scenes
```

### 6.2 Replica smoke test

```bash
python -m eval reconstruction \
    --protocol replica8 \
    --scenes office1 \
    --data-root /workspace/data/Replica \
    --recon-root /workspace/results/results_replica_radiov4_128 \
    --out-root /workspace/results/results_replica_radiov4_128/reconstruction_eval_results/office1
```

---

## 7. Useful commands

```bash
python -m tools.slam.run --help
python -m eval --help
python -m eval semantic --help
python -m eval reconstruction --help
```

For a new configuration, test one scene with a large `--frame-stride` and `--save-masks` before launching the full benchmark.

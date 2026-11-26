from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(
    path_value: str | Path,
) -> Path:
    """
    Resolve a registry path relative to the repository root.
    """
    path = Path(path_value)

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path.resolve()


class RadioPCAReconstructor:
    """
    Reconstruct full-dimensional RADIO features from the
    compressed feature vectors stored/rendered by LEG-SLAM.

    Compression is assumed to have the PCA form:

        compressed = (full - mean) @ components.T

    where:

        full        : [..., D]
        compressed  : [..., K]
        mean        : [D]
        components  : [K, D]

    Reconstruction is therefore:

        reconstructed = compressed @ components + mean

    This class deliberately knows nothing about semantic
    classes, text embeddings, SigLIP, or linear probes.
    """

    def __init__(
        self,
        *,
        compressor_path: Path,
        compressed_dim: int,
        device: torch.device,
    ) -> None:

        self.device = torch.device(device)
        self.compressor_path = Path(
            compressor_path
        ).resolve()

        self.compressed_dim = int(
            compressed_dim
        )

        if self.compressed_dim <= 0:
            raise ValueError(
                "compressed_dim must be > 0"
            )

        if not self.compressor_path.is_file():
            raise FileNotFoundError(
                "RADIO PCA compressor not found: "
                f"{self.compressor_path}"
            )

        # ------------------------------------------------------
        # Load exactly the PCA buffers used by the LEG-SLAM
        # RADIO feature pipeline.
        # ------------------------------------------------------

        print(
            "Loading RADIO PCA compressor:",
            self.compressor_path,
        )

        pca_module = torch.jit.load(
            str(self.compressor_path),
            map_location="cpu",
        ).eval()

        state = pca_module.state_dict()

        if "mean" not in state:
            raise RuntimeError(
                "TorchPCA does not expose a 'mean' buffer."
            )

        if "components" not in state:
            raise RuntimeError(
                "TorchPCA does not expose a 'components' buffer."
            )

        mean = (
            state["mean"]
            .detach()
            .float()
            .cpu()
        )

        components = (
            state["components"]
            .detach()
            .float()
            .cpu()
        )

        del pca_module

        # Some PCA implementations store mean as [1, D].
        if (
            mean.ndim == 2
            and mean.shape[0] == 1
        ):
            mean = mean.squeeze(0)

        if mean.ndim != 1:
            raise RuntimeError(
                "Expected PCA mean [D] or [1,D], "
                f"got {tuple(mean.shape)}"
            )

        if components.ndim != 2:
            raise RuntimeError(
                "Expected PCA components [N,D], "
                f"got {tuple(components.shape)}"
            )

        if (
            components.shape[0]
            < self.compressed_dim
        ):
            raise RuntimeError(
                "PCA has fewer components than requested: "
                f"components={tuple(components.shape)}, "
                f"compressed_dim={self.compressed_dim}"
            )

        if (
            components.shape[1]
            != mean.shape[0]
        ):
            raise RuntimeError(
                "PCA mean/components dimension mismatch: "
                f"mean={tuple(mean.shape)}, "
                f"components={tuple(components.shape)}"
            )

        self.full_dim = int(
            mean.shape[0]
        )

        # Only the first K PCA components were used when
        # creating the compressed semantic representation.
        self.mean = mean.to(
            self.device,
            non_blocking=True,
        )

        self.components = (
            components[
                : self.compressed_dim
            ]
            .contiguous()
            .to(
                self.device,
                non_blocking=True,
            )
        )

        print(
            "RADIO PCA loaded: "
            f"{self.compressed_dim} -> "
            f"{self.full_dim}"
        )

    @classmethod
    def from_model_config(
        cls,
        *,
        model_cfg: Mapping[str, Any],
        device: torch.device,
    ) -> "RadioPCAReconstructor":
        """
        Construct from one entry in cfg/feature_models.json.

        Expected compressor configuration:

            "compressor": {
                "type": "TorchPCA",
                "path": "...",
                "output_dim": 64
            }
        """

        compressor_cfg = model_cfg.get(
            "compressor",
            {}
        )

        compressor_type = (
            compressor_cfg.get("type")
        )

        if compressor_type != "TorchPCA":
            raise ValueError(
                "RADIO PCA reconstruction expects "
                "compressor.type='TorchPCA', got "
                f"{compressor_type!r}"
            )

        if "path" not in compressor_cfg:
            raise ValueError(
                "RADIO compressor config has no 'path'."
            )

        if "output_dim" not in compressor_cfg:
            raise ValueError(
                "RADIO compressor config has no "
                "'output_dim'."
            )

        compressor_path = resolve_repo_path(
            compressor_cfg["path"]
        )

        compressed_dim = int(
            compressor_cfg["output_dim"]
        )

        reconstructor = cls(
            compressor_path=compressor_path,
            compressed_dim=compressed_dim,
            device=device,
        )

        # Optional consistency check against the encoder
        # metadata already present in the registry.
        encoder_cfg = model_cfg.get(
            "encoder",
            {}
        )

        declared_full_dim = (
            encoder_cfg.get(
                "embedding_size"
            )
        )

        if declared_full_dim is not None:
            declared_full_dim = int(
                declared_full_dim
            )

            if (
                declared_full_dim
                != reconstructor.full_dim
            ):
                raise RuntimeError(
                    "RADIO encoder/PCA full-dimension "
                    "mismatch: "
                    f"encoder={declared_full_dim}, "
                    f"PCA={reconstructor.full_dim}"
                )

        return reconstructor

    def prepare_rendered_features(
        self,
        rendered_lf: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        int,
        int,
    ]:
        """
        Convert rendered [K,H,W] features into [H*W,K].

        No normalization or sign change is applied here.
        """

        if rendered_lf.ndim != 3:
            raise ValueError(
                "Expected rendered RADIO features "
                "[K,H,W], got "
                f"{tuple(rendered_lf.shape)}"
            )

        k, h, w = rendered_lf.shape

        if k != self.compressed_dim:
            raise RuntimeError(
                "Rendered RADIO feature dimension "
                "does not match PCA compressor: "
                f"rendered={k}, "
                f"expected={self.compressed_dim}"
            )

        compressed = (
            rendered_lf
            .permute(1, 2, 0)
            .contiguous()
            .reshape(-1, k)
        )

        return compressed, h, w

    @torch.inference_mode()
    def reconstruct_chunk(
        self,
        compressed_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct one [N,K] chunk into [N,D].

        Returned features are float32 because the PCA inverse
        is deliberately performed in float32.
        """

        if compressed_chunk.ndim != 2:
            raise ValueError(
                "Expected compressed chunk [N,K], "
                f"got {tuple(compressed_chunk.shape)}"
            )

        if (
            compressed_chunk.shape[1]
            != self.compressed_dim
        ):
            raise RuntimeError(
                "Compressed chunk has wrong dimension: "
                f"got {compressed_chunk.shape[1]}, "
                f"expected {self.compressed_dim}"
            )

        compressed_chunk = (
            compressed_chunk
            .to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

        # Inverse of:
        #
        #   compressed
        #       = (full - mean) @ components.T
        #
        # using the truncated PCA basis.
        reconstructed = (
            torch.matmul(
                compressed_chunk,
                self.components,
            )
            + self.mean
        )

        return reconstructed

    @torch.inference_mode()
    def iter_reconstructed_chunks(
        self,
        rendered_lf: torch.Tensor,
        *,
        chunk_size: int,
    ) -> Iterator[
        tuple[
            int,
            int,
            torch.Tensor,
        ]
    ]:
        """
        Iterate over reconstructed RADIO pixel features
        without ever allocating [H*W,D] for the whole image.

        Yields:
            start
            end
            reconstructed_chunk [N,D]
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be > 0"
            )

        compressed, _, _ = (
            self.prepare_rendered_features(
                rendered_lf
            )
        )

        num_pixels = int(
            compressed.shape[0]
        )

        for start in range(
            0,
            num_pixels,
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                num_pixels,
            )

            reconstructed = (
                self.reconstruct_chunk(
                    compressed[
                        start:end
                    ]
                )
            )

            yield (
                start,
                end,
                reconstructed,
            )

    @torch.inference_mode()
    def reconstruct_map(
        self,
        rendered_lf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience/debug method.

        Returns the entire reconstructed RADIO feature map:

            [D,H,W]

        Avoid this during normal evaluation because the full
        1152-D map can consume a large amount of memory.
        """

        compressed, h, w = (
            self.prepare_rendered_features(
                rendered_lf
            )
        )

        reconstructed = (
            self.reconstruct_chunk(
                compressed
            )
        )

        return (
            reconstructed
            .reshape(
                h,
                w,
                self.full_dim,
            )
            .permute(2, 0, 1)
            .contiguous()
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "compressor_path": str(
                self.compressor_path
            ),
            "compressed_dim": (
                self.compressed_dim
            ),
            "full_dim": self.full_dim,
            "reconstruction_rule": (
                "compressed @ components + mean"
            ),
        }
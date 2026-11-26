from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from .base import SemanticPrediction, SemanticPredictor
from .radio_closedset import (
    INDOOR37_CLASS_NAMES,
    RadioIndoor37KnownnessBranch,
)
from .radio_features import RadioPCAReconstructor

from eval.common.torch_compat import (
    patch_torch_compiler_compat,
    patch_torch_hub_weights_only_compat,
    patch_torch_load_state_dict_compat,
)

from ..prompts import DEFAULT_PROMPT_SET, build_prompt_groups


@torch.inference_mode()
def encode_radio_class_text_features(
    adaptor: torch.nn.Module,
    prompt_groups: Sequence[Sequence[str]],
    device: torch.device,
    expected_num_classes: int,
) -> torch.Tensor:
    """Encode one normalized SigLIP2-G text prototype per class/query."""
    class_features: List[torch.Tensor] = []

    for prompts in prompt_groups:
        tokens = adaptor.tokenizer(list(prompts)).to(device)
        features = adaptor.encode_text(
            tokens,
            normalize=True,
        ).float()

        class_feature = F.normalize(
            features.mean(dim=0, keepdim=True),
            dim=-1,
        )
        class_features.append(class_feature)

    text_features = torch.cat(class_features, dim=0)

    if text_features.shape[0] != expected_num_classes:
        raise RuntimeError(
            f"Expected {expected_num_classes} RADIO text embeddings, "
            f"got {tuple(text_features.shape)}"
        )

    return text_features


def _cosine_to_gate(
    cosine: torch.Tensor,
    threshold: float = 0.6,
) -> torch.Tensor:
    """
    Convert text cosine similarity into a hard semantic-family gate.

    Returns 1.0 when cosine >= threshold and 0.0 otherwise.

    This deliberately does NOT use (cosine + 1) / 2.  The closed branch is
    allowed to support only queries that pass a real compatibility gate.
    """
    return (cosine >= threshold).to(cosine.dtype)


def _canonicalize_semantic_name(name: str) -> str:
    """
    Canonicalize a class/query name for exact semantic-name matching.

    This makes spelling variants such as:
        floor_mat / floor-mat / "floor mat"
        night_stand / night-stand / nightstand
    compare equal when they differ only by separators/case.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


class RadioSiglipSemanticPredictor(SemanticPredictor):
    """
    Standard RADIO open-vocabulary semantic predictor.

    compressed Gaussian features
        -> PCA inverse
        -> full RADIO features
        -> SigLIP2-G alignment head
        -> cosine similarity with arbitrary query text prototypes
    """

    def __init__(
        self,
        class_names: Sequence[str],
        feature_model_id: str,
        model_cfg: Dict[str, Any],
        chunk_size: int,
        device: str | torch.device,
        repo_root: Path,
    ) -> None:
        super().__init__(
            class_names=class_names,
            device=device,
        )

        self.feature_model_id = feature_model_id
        self.repo_root = Path(repo_root)
        self.alignment_type = "radio_siglip"
        self.chunk_size = int(chunk_size)

        if self.chunk_size <= 0:
            raise ValueError("RADIO alignment chunk size must be > 0")

        self.reconstructor = RadioPCAReconstructor.from_model_config(
            model_cfg=model_cfg,
            device=self.device,
        )
        self.compressed_dim = self.reconstructor.compressed_dim

        alignment_cfg = model_cfg.get("text_alignment", {})
        radio_version = alignment_cfg.get(
            "version",
            "c-radio_v4-so400m",
        )
        adaptor_name = alignment_cfg.get(
            "adaptor",
            "siglip2-g",
        )

        print(
            "Loading RADIO text alignment: "
            f"version={radio_version}, adaptor={adaptor_name}"
        )

        patch_torch_hub_weights_only_compat()
        patch_torch_compiler_compat()
        patch_torch_load_state_dict_compat()

        radio_model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=radio_version,
            progress=True,
            skip_validation=True,
            adaptor_names=[adaptor_name],
            trust_repo=True,
        ).eval()

        if adaptor_name not in radio_model.adaptors:
            raise RuntimeError(
                f"RADIO adaptor {adaptor_name!r} was not loaded. "
                f"Available: {list(radio_model.adaptors.keys())}"
            )

        self.adaptor = (
            radio_model.adaptors[adaptor_name]
            .to(self.device)
            .eval()
        )
        del radio_model

        prompt_groups = build_prompt_groups(
            self.class_names,
            DEFAULT_PROMPT_SET,
        )
        self.text_emb = encode_radio_class_text_features(
            adaptor=self.adaptor,
            prompt_groups=prompt_groups,
            device=self.device,
            expected_num_classes=self.num_classes,
        )
        self.text_emb = F.normalize(
            self.text_emb.float(),
            dim=-1,
        )

        print(
            "RADIO text embedding shape: "
            f"{tuple(self.text_emb.shape)}"
        )
        print(
            "RADIO PCA: "
            f"compressed_dim={self.reconstructor.compressed_dim}, "
            f"full_dim={self.reconstructor.full_dim}"
        )

        self.prediction_rule = (
            "rendered compressed RADIO features -> PCA inverse -> "
            "SigLIP2-G head -> argmax cosine similarity"
        )

    def _validate_candidate_ids(
        self,
        candidate_ids: Sequence[int],
    ) -> list[int]:
        ids = [int(class_id) for class_id in candidate_ids]
        if not ids:
            raise ValueError("candidate_ids must contain at least one class")
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_ids must be unique")

        invalid = [
            class_id
            for class_id in ids
            if class_id < 0 or class_id >= self.num_classes
        ]
        if invalid:
            raise ValueError(
                f"candidate_ids outside [0,{self.num_classes - 1}]: {invalid}"
            )
        return ids

    def _candidate_text(
        self,
        candidate_ids: Sequence[int],
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        ids = self._validate_candidate_ids(candidate_ids)
        ids_t = torch.as_tensor(
            ids,
            dtype=torch.long,
            device=self.device,
        )
        candidate_text = self.text_emb.index_select(0, ids_t)
        return ids, ids_t, candidate_text

    @torch.inference_mode()
    def score_open_vocab_features(
        self,
        full_features: torch.Tensor,
        candidate_text: torch.Tensor,
    ) -> torch.Tensor:
        """Score full reconstructed RADIO features [N,D] against query text [Q,T]."""
        if full_features.ndim != 2:
            raise ValueError(
                f"Expected full RADIO features [N,D], got {tuple(full_features.shape)}"
            )
        if full_features.shape[1] != self.reconstructor.full_dim:
            raise RuntimeError(
                "Full RADIO feature dimension mismatch: "
                f"features={full_features.shape[1]}, "
                f"expected={self.reconstructor.full_dim}"
            )

        head_dtype = next(self.adaptor.head_mlp.parameters()).dtype
        aligned = self.adaptor.head_mlp(
            full_features.to(head_dtype)
        ).float()
        aligned = F.normalize(aligned, dim=-1)

        return torch.matmul(
            aligned,
            candidate_text.transpose(0, 1),
        )

    @torch.inference_mode()
    def predict(
        self,
        rendered_lf: torch.Tensor,
        candidate_ids: Sequence[int],
    ) -> SemanticPrediction:
        if rendered_lf.ndim != 3:
            raise ValueError(
                "Expected rendered_lf [K,H,W], "
                f"got {tuple(rendered_lf.shape)}"
            )

        k, h, w = rendered_lf.shape
        if k != self.reconstructor.compressed_dim:
            raise RuntimeError(
                f"RADIO map has {k} semantic channels, but registry "
                f"compressor output_dim is {self.reconstructor.compressed_dim}."
            )

        ids, candidate_ids_t, candidate_text = self._candidate_text(
            candidate_ids
        )

        pred_chunks: List[torch.Tensor] = []
        score_chunks: List[torch.Tensor] = []
        class_score_chunks: List[torch.Tensor] = []

        for _, _, full_chunk in self.reconstructor.iter_reconstructed_chunks(
            rendered_lf,
            chunk_size=self.chunk_size,
        ):
            scores = self.score_open_vocab_features(
                full_chunk,
                candidate_text,
            )
            class_score_chunks.append(scores)

            selected_score, local_pred = torch.max(scores, dim=-1)
            pred_chunks.append(candidate_ids_t[local_pred])
            score_chunks.append(selected_score)

        pred = torch.cat(pred_chunks, dim=0).reshape(h, w)
        selected_score = torch.cat(score_chunks, dim=0).reshape(h, w)

        class_scores = (
            torch.cat(class_score_chunks, dim=0)
            .reshape(h, w, len(ids))
            .permute(2, 0, 1)
            .contiguous()
        )

        decoded_labels = self.labels_from_scores(
            class_scores,
            ids,
        )
        if not torch.equal(decoded_labels, pred):
            raise RuntimeError(
                "labels_from_scores() does not reproduce the predictor's labels"
            )

        return SemanticPrediction(
            labels=pred,
            scores=selected_score,
            class_scores=class_scores,
        )

    @torch.inference_mode()
    def labels_from_scores(
        self,
        class_scores: torch.Tensor,
        candidate_ids: Sequence[int],
    ) -> torch.Tensor:
        if class_scores.ndim != 3:
            raise ValueError(
                "Expected class_scores [C,H,W], "
                f"got {tuple(class_scores.shape)}"
            )

        ids = self._validate_candidate_ids(candidate_ids)
        if class_scores.shape[0] != len(ids):
            raise ValueError(
                "Score-channel/candidate mismatch: "
                f"{class_scores.shape[0]} score channels for {len(ids)} candidate IDs"
            )

        candidate_ids_t = torch.as_tensor(
            ids,
            dtype=torch.long,
            device=class_scores.device,
        )
        local_pred = torch.argmax(class_scores, dim=0)
        return candidate_ids_t[local_pred]

    def metadata(self) -> Dict[str, Any]:
        return {
            "feature_model": self.feature_model_id,
            "text_alignment": self.alignment_type,
            "prediction_rule": self.prediction_rule,
            "alignment_chunk_size": self.chunk_size,
            "prompt_set": DEFAULT_PROMPT_SET.name,
            **{
                f"pca_{key}": value
                for key, value in self.reconstructor.metadata().items()
            },
        }



class RadioDualSemanticPredictor(RadioSiglipSemanticPredictor):
    """
    Simple closed-set / open-vocabulary probability fusion.

    1. P_open: RADIO + SigLIP open-vocabulary probabilities.
    2. P_closed: Indoor37 probabilities transferred to arbitrary queries.
    3. A(c,q): binary statistical compatibility gate.
    4. R(q): whether any closed class supports query q.
    5. alpha(x,q) = beta * K(x) * M(x) * R(q).

    Final:
        P_final = alpha * P_closed + (1 - alpha) * P_open

    No semantic families, top-k restrictions, name matching,
    OOV protection J, or additive logit priors.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        feature_model_id: str,
        model_cfg: Dict[str, Any],
        chunk_size: int,
        device: str | torch.device,
        repo_root: Path,
        probe_checkpoint: Path,
        prototype_path: Path,
        fusion_beta: float = 1.0,
        tau_ov: float = 1.0,
        tau_cs: float = 1.0,
        use_query_coverage: bool = True,
        semantic_z_threshold: float = 2.2,
    ) -> None:
        super().__init__(
            class_names=class_names,
            feature_model_id=feature_model_id,
            model_cfg=model_cfg,
            chunk_size=chunk_size,
            device=device,
            repo_root=repo_root,
        )

        self.fusion_beta = float(fusion_beta)
        self.tau_ov = float(tau_ov)
        self.tau_cs = float(tau_cs)
        self.semantic_z_threshold = float(
            semantic_z_threshold
        )

        if self.fusion_beta < 0:
            raise ValueError("fusion_beta must be >= 0")
        if self.tau_ov <= 0 or self.tau_cs <= 0:
            raise ValueError("Temperatures must be > 0")
        if self.semantic_z_threshold < 0:
            raise ValueError(
                "semantic_z_threshold must be >= 0"
            )

        # R(q) is essential to this version of the algorithm.
        # Keep this argument for compatibility with the old CLI.
        if not use_query_coverage:
            raise ValueError(
                "This fusion requires use_query_coverage=True"
            )

        # --------------------------------------------------
        # 1. Load the existing Indoor37 classifier/prototypes.
        #    This branch produces p_c(x), K(x), and M(x).
        # --------------------------------------------------
        self.closed_branch = RadioIndoor37KnownnessBranch(
            checkpoint_path=probe_checkpoint,
            prototype_path=prototype_path,
            device=self.device,
            feature_dim=self.reconstructor.full_dim,
        )

        # --------------------------------------------------
        # 2. Encode the 37 closed-class names.
        #    Use exactly the same SigLIP prompt policy as OV.
        # --------------------------------------------------
        closed_prompts = build_prompt_groups(
            INDOOR37_CLASS_NAMES,
            DEFAULT_PROMPT_SET,
        )

        self.closed_text_emb = (
            encode_radio_class_text_features(
                adaptor=self.adaptor,
                prompt_groups=closed_prompts,
                device=self.device,
                expected_num_classes=len(
                    INDOOR37_CLASS_NAMES
                ),
            )
        )

        # --------------------------------------------------
        # 3. Text similarity between closed classes and
        #    arbitrary registered queries.
        #
        # S[c,q] = cosine(t_c, t_q)
        # Shape: [37, num_queries]
        # --------------------------------------------------
        similarity = (
            self.closed_text_emb @ self.text_emb.T
        )

        # --------------------------------------------------
        # 4. Query-relative statistical normalization.
        #
        # For each query q:
        #   mean_q = mean over 37 classes
        #   std_q  = std over 37 classes
        #
        # Z[c,q] = (S[c,q] - mean_q) / std_q
        #
        # A high Z means this closed class is unusually
        # similar to the query relative to the other classes.
        # --------------------------------------------------
        mean_q = similarity.mean(
            dim=0,
            keepdim=True,
        )

        std_q = similarity.std(
            dim=0,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)

        self.text_z_scores = (
            similarity - mean_q
        ) / std_q

        # --------------------------------------------------
        # 5. Binary compatibility gate A(c,q).
        #
        # No top-1/top-2 restriction.
        # No manual name matching.
        # No absolute cosine threshold.
        # --------------------------------------------------
        self.closed_query_affinity = (
            self.text_z_scores
            >= self.semantic_z_threshold
        ).float()

        self.prediction_rule = (
            "probability mixture using statistical binary "
            "text affinity and alpha=beta*K*M*R"
        )

        print(
            "RADIO dual fusion: "
            f"beta={self.fusion_beta}, "
            f"tau_ov={self.tau_ov}, "
            f"tau_cs={self.tau_cs}, "
            f"z_threshold={self.semantic_z_threshold}"
        )

        # Print the compatible classes for inspection.
        print("Statistical closed-class compatibility:")

        for q_idx, query in enumerate(self.class_names):
            compatible = torch.nonzero(
                self.closed_query_affinity[:, q_idx],
                as_tuple=False,
            ).flatten().tolist()

            matches = [
                (
                    INDOOR37_CLASS_NAMES[c],
                    round(
                        float(
                            self.text_z_scores[c, q_idx]
                        ),
                        2,
                    ),
                )
                for c in compatible
            ]

            print(f"  {query}: {matches}")

    @torch.inference_mode()
    def predict(
        self,
        rendered_lf: torch.Tensor,
        candidate_ids: Sequence[int],
    ) -> SemanticPrediction:

        if rendered_lf.ndim != 3:
            raise ValueError(
                "Expected rendered_lf [K,H,W]"
            )

        k, h, w = rendered_lf.shape

        if k != self.reconstructor.compressed_dim:
            raise RuntimeError(
                "RADIO semantic channel mismatch"
            )

        ids, candidate_ids_t, candidate_text = (
            self._candidate_text(candidate_ids)
        )

        # --------------------------------------------------
        # A(c,q): [37,Q]
        # R(q) = max_c A(c,q): [Q]
        # --------------------------------------------------
        affinity = self.closed_query_affinity.index_select(
            1,
            candidate_ids_t,
        )

        query_coverage = affinity.max(dim=0).values

        covered = query_coverage > 0.5

        # Optional probability diagnostics (no effect on prediction).
        # Use the same evaluation frames and candidate-query set as fusion.
        prob_dir = os.environ.get("RADIO_FUSION_PROB_DIAG_DIR", "")
        prob_frame = getattr(self, "_prob_diag_frame_index", 0)
        prob_limit = int(os.environ.get("RADIO_FUSION_PROB_DIAG_LIMIT", "2"))
        save_prob = bool(prob_dir) and prob_frame < prob_limit
        if prob_dir:
            self._prob_diag_frame_index = prob_frame + 1

        debug_series = {}
        debug_query_indices = {}
        if save_prob:
            requested = os.environ.get(
                "RADIO_FUSION_PROB_CLASSES",
                "floor,rug,vase,cushion,pillow,blanket",
            )
            for query_name in (s.strip().lower() for s in requested.split(",")):
                if not query_name:
                    continue
                for local_idx, class_id in enumerate(ids):
                    if self.class_names[class_id].lower() == query_name:
                        debug_query_indices[query_name] = local_idx
                        for prefix in ("ov", "cs", "final", "support"):
                            debug_series[f"{prefix}__{query_name}"] = []
                        break
            for name in ("ov_winner", "final_winner", "ov_max", "final_max",
                         "closed_conditional_max", "beta_km"):
                debug_series[name] = []
            print(
                "[radio probabilities] collecting:",
                list(debug_query_indices),
                f"frame={prob_frame}"
            )

        # Output buffers.
        probability_chunks = []
        score_chunks = []
        label_chunks = []

        knownness_chunks = []
        confidence_chunks = []
        alpha_chunks = []

        head_label_chunks = []
        proto_label_chunks = []
        head_proto_agreement_chunks = []

        for _, _, full_chunk in (
            self.reconstructor.iter_reconstructed_chunks(
                rendered_lf,
                chunk_size=self.chunk_size,
            )
        ):

            # ==============================================
            # A. OPEN-VOCABULARY BRANCH
            # ==============================================

            # L_open(x,q)
            ov_logits = self.score_open_vocab_features(
                full_chunk,
                candidate_text,
            )

            # P_open(q|x)
            p_open = torch.softmax(
                ov_logits / self.tau_ov,
                dim=-1,
            )

            # ==============================================
            # B. CLOSED-SET BRANCH
            # ==============================================

            closed = self.closed_branch.infer_features(
                full_chunk
            )

            # Closed posterior p_c(x): [N,37]
            p_c = closed.probabilities

            # Pixel knownness K(x): [N]
            K = closed.knownness.clamp(0.0, 1.0)

            # Closed confidence M(x): [N]
            M = closed.confidence.clamp(0.0, 1.0)

            # ----------------------------------------------
            # Closed support for every query:
            #
            # B(x,q) = sum_c p_c(x) * A(c,q)
            #
            # Shape: [N,Q]
            # ----------------------------------------------
            B = p_c @ affinity

            # Number of compatible closed classes for each query.
            # Shape: [Q]
            # n_matches = affinity.sum(dim=0).clamp_min(1.0)

            # # Average the probabilities over compatible closed classes.
            # # Shape: [N, Q]
            # B = (p_c @ affinity) / n_matches.unsqueeze(0)

            # ----------------------------------------------
            # Construct P_closed over arbitrary queries.
            #
            # Only queries with R(q)=1 receive closed-set
            # classification evidence.
            #
            # We preserve the total P_open probability mass
            # of covered queries. This is necessary because
            # alpha is query-dependent.
            #
            # For R(q)=0, P_closed is set equal to P_open.
            # ----------------------------------------------
            p_closed = p_open.clone()

            if bool(covered.any()):
                # Total OV mass belonging to covered queries.
                covered_mass = p_open[:, covered].sum(
                    dim=-1,
                    keepdim=True,
                )

                # Closed distribution conditioned on coverage.
                closed_conditional = torch.softmax(
                    B[:, covered] / self.tau_cs,
                    dim=-1,
                )

                # Redistribute only the covered mass.
                p_closed[:, covered] = (
                    covered_mass * closed_conditional
                )

            # ==============================================
            # C. FUSION COEFFICIENT
            # ==============================================

            # alpha(x,q) = beta * K(x) * M(x) * R(q)
            #
            # K and M depend on pixels.
            # R depends on queries.
            #
            # Shape: [N,Q]
            alpha = (
                self.fusion_beta
                * K[:, None]
                * M[:, None]
                * query_coverage[None, :]
            ).clamp(0.0, 1.0)

            # ==============================================
            # D. FINAL PROBABILITY MIXTURE
            # ==============================================

            # P_final = alpha * P_closed
            #           + (1-alpha) * P_open
            #
            # When R(q)=0:
            #   alpha(x,q)=0
            #   P_final(q|x)=P_open(q|x)
            #
            # No additional softmax or normalization is
            # required: both branches preserve the covered
            # probability mass and the uncovered mass.
            p_final = (
                alpha * p_closed
                + (1.0 - alpha) * p_open
            )

            # Final prediction over arbitrary queries.
            selected_score, local_pred = p_final.max(
                dim=-1
            )

            final_labels = candidate_ids_t[local_pred]

            if save_prob:
                # Save only a handful of query-specific score maps, rather
                # than allocating the entire [H,W,Q] tensor in the NPZ.
                for query_name, j in debug_query_indices.items():
                    debug_series[f"ov__{query_name}"].append(
                        p_open[:, j].detach().float().cpu()
                    )
                    debug_series[f"cs__{query_name}"].append(
                        p_closed[:, j].detach().float().cpu()
                    )
                    debug_series[f"final__{query_name}"].append(
                        p_final[:, j].detach().float().cpu()
                    )
                    debug_series[f"support__{query_name}"].append(
                        B[:, j].detach().float().cpu()
                    )

                # IDs correspond to the registered query vocabulary.
                debug_series["ov_winner"].append(
                    candidate_ids_t[p_open.argmax(dim=-1)].detach().cpu()
                )
                debug_series["final_winner"].append(
                    final_labels.detach().cpu()
                )
                debug_series["ov_max"].append(
                    p_open.max(dim=-1).values.detach().float().cpu()
                )
                debug_series["final_max"].append(
                    p_final.max(dim=-1).values.detach().float().cpu()
                )
                debug_series["beta_km"].append(
                    (self.fusion_beta * K * M).clamp(0, 1).detach().float().cpu()
                )
                # A nearly-uniform closed conditional has max approx. 1/n.
                if bool(covered.any()):
                    debug_series["closed_conditional_max"].append(
                        closed_conditional.max(dim=-1).values.detach().float().cpu()
                    )
                else:
                    debug_series["closed_conditional_max"].append(
                        torch.zeros_like(K, device="cpu")
                    )

            probability_chunks.append(p_final)
            score_chunks.append(selected_score)
            label_chunks.append(final_labels)

            # ==============================================
            # E. DIAGNOSTICS
            # ==============================================

            knownness_chunks.append(K)
            confidence_chunks.append(M)

            # Mean alpha across the current query set.
            alpha_chunks.append(alpha.mean(dim=-1))

            head_label_chunks.append(closed.head_labels)
            proto_label_chunks.append(
                closed.prototype_labels
            )

            head_proto_agreement_chunks.append(
                closed.head_labels.eq(
                    closed.prototype_labels
                )
            )

        # ==============================================
        # F. RECONSTRUCT FULL IMAGE OUTPUTS
        # ==============================================

        labels = torch.cat(
            label_chunks,
            dim=0,
        ).reshape(h, w)

        selected_score = torch.cat(
            score_chunks,
            dim=0,
        ).reshape(h, w)

        class_scores = (
            torch.cat(probability_chunks, dim=0)
            .reshape(h, w, len(ids))
            .permute(2, 0, 1)
            .contiguous()
        )

        decoded = self.labels_from_scores(
            class_scores,
            ids,
        )

        if not torch.equal(decoded, labels):
            raise RuntimeError(
                "Class scores do not reproduce labels"
            )

        def make_map(chunks):
            return torch.cat(
                chunks,
                dim=0,
            ).reshape(h, w)

        aux = {
            "closed_head_labels": make_map(
                head_label_chunks
            ),
            "closed_proto_labels": make_map(
                proto_label_chunks
            ),
            "prototype_knownness": make_map(
                knownness_chunks
            ),
            "closed_confidence": make_map(
                confidence_chunks
            ),
            "fusion_alpha_mean": make_map(
                alpha_chunks
            ),
            "head_proto_agreement": make_map(
                head_proto_agreement_chunks
            ),

            # Query-level diagnostics.
            "query_coverage": query_coverage.detach(),
            "text_z_scores": (
                self.text_z_scores.index_select(
                    1,
                    candidate_ids_t,
                ).detach()
            ),
            "closed_query_affinity": affinity.detach(),
        }

        if save_prob:
            # Dump raw floating-point predictions for GT-conditioned analysis.
            # The saved arrays align pixel-for-pixel with rendered_lf and GT.
            import numpy as np

            dest = Path(prob_dir)
            dest.mkdir(parents=True, exist_ok=True)
            arrays = {
                key: torch.cat(chunks).reshape(h, w).numpy()
                for key, chunks in debug_series.items()
            }
            arrays.update({
                "candidate_ids": np.asarray(ids, dtype=np.int16),
                "candidate_names": np.asarray(
                    [self.class_names[k] for k in ids], dtype="U80"
                ),
                "query_coverage": query_coverage.detach().cpu().numpy(),
                "n_covered": np.int32(covered.sum().item()),
                "beta": np.float32(self.fusion_beta),
                "tau_ov": np.float32(self.tau_ov),
                "tau_cs": np.float32(self.tau_cs),
            })
            output_path = dest / f"frame_{prob_frame:04d}_probabilities.npz"
            np.savez_compressed(output_path, **arrays)
            changed = (arrays["ov_winner"] != arrays["final_winner"]).mean()
            print(
                f"[radio probabilities] saved {output_path}; "
                f"winner_changed={changed:.2%}; "
                f"n_covered={arrays['n_covered']}/{len(ids)}; "
                f"mean_max_PCS_cond={arrays['closed_conditional_max'].mean():.4f}"
            )

        # Debug output is optional and does not change the prediction.
        self._save_reliability_diagnostics(aux)

        return SemanticPrediction(
            labels=labels,
            scores=selected_score,
            class_scores=class_scores,
            aux=aux,
        )

    def _save_reliability_diagnostics(self, aux: Dict[str, Any]) -> None:
        """Optional K/M visualization. Does not modify inference or scores.

        Enable with the environment variable RADIO_FUSION_DIAG_DIR.
        Save at most RADIO_FUSION_DIAG_LIMIT prediction frames (default 8).

        Each frame produces:
          - a PNG with K, M, K*M, and beta*K*M (fixed grayscale 0..1)
          - an NPZ with the exact floating-point maps for quantitative analysis
        """
        output_dir = os.environ.get("RADIO_FUSION_DIAG_DIR")
        if not output_dir:
            return

        limit = int(os.environ.get("RADIO_FUSION_DIAG_LIMIT", "8"))
        frame_idx = getattr(self, "_diag_frame_index", 0)
        self._diag_frame_index = frame_idx + 1
        if frame_idx >= limit:
            return

        import numpy as np
        from PIL import Image, ImageDraw

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        K = aux["prototype_knownness"].detach().float().cpu().numpy()
        M = aux["closed_confidence"].detach().float().cpu().numpy()
        KM = K * M
        beta_KM = np.clip(self.fusion_beta * KM, 0.0, 1.0)

        arrays = {
            "K_knownness": K,
            "M_confidence": M,
            "K_times_M": KM,
            "beta_K_times_M": beta_KM,
        }
        np.savez_compressed(out / f"frame_{frame_idx:04d}_maps.npz", **arrays)

        # All panels share a FIXED 0..1 scale. Do NOT normalize each map to
        # its own maximum: that would hide whether the values are actually low.
        h, w = K.shape
        title_h = 24
        canvas = Image.new("RGB", (4 * w, h + title_h), "white")
        drawer = ImageDraw.Draw(canvas)
        for i, (title, values) in enumerate(arrays.items()):
            image_8bit = (np.clip(values, 0, 1) * 255).round().astype(np.uint8)
            panel = Image.fromarray(image_8bit, mode="L").convert("RGB")
            canvas.paste(panel, (i * w, title_h))
            drawer.text((i * w + 8, 6), title, fill="black")
        dest = out / f"frame_{frame_idx:04d}_reliability.png"
        canvas.save(dest)

        values_info = []
        for name, values in arrays.items():
            values_info.append(
                f"{name}: mean={values.mean():.3f}, "
                f"p10={np.percentile(values, 10):.3f}, "
                f"p50={np.median(values):.3f}, "
                f"p90={np.percentile(values, 90):.3f}"
            )
        print(f"[radio reliability] saved {dest}")
        for line in values_info:
            print("  " + line)

    def metadata(self) -> Dict[str, Any]:
        result = super().metadata()

        result.update(
            {
                "predictor": "radio_dual_fusion",
                "fusion_mode": "probability_mixture",
                "prediction_rule": self.prediction_rule,
                "fusion_beta": self.fusion_beta,
                "tau_ov": self.tau_ov,
                "tau_cs": self.tau_cs,
                "semantic_z_threshold": (
                    self.semantic_z_threshold
                ),
                "text_affinity": (
                    "binary query-relative z-score gate"
                ),
                "query_coverage": "max_c A(c,q)",
                "fusion_alpha": "beta*K(x)*M(x)*R(q)",
                "closed_branch": self.closed_branch.metadata(),
            }
        )

        return result

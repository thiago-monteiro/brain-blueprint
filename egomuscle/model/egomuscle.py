from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.utils.checkpoint as cp

from .adapters import AdapterLinear
from .fusion import CrossAttentionFusion, LateFusionBlock
from .memory import FastStateEncoder
from .muscle_encoder import MuscleEncoder
from .pbit import PBitBottleneck
from .video_encoder import FrozenVideoEncoder


@dataclass
class EgoMuscleOutput:
    pred: torch.Tensor | None
    target: torch.Tensor | None
    attention: torch.Tensor | None
    fused: torch.Tensor
    pooled: torch.Tensor
    video_repr: torch.Tensor | None
    conditioning_repr: torch.Tensor | None
    layerwise_video_repr: dict[str, torch.Tensor] | None = None
    pred_mu: torch.Tensor | None = None
    pred_log_var: torch.Tensor | None = None
    pred_video_latent: torch.Tensor | None = None
    video_latent_target: torch.Tensor | None = None
    fast_state: torch.Tensor | None = None
    fast_mu: torch.Tensor | None = None
    fast_log_var: torch.Tensor | None = None
    fast_kl: torch.Tensor | None = None
    pbit_probabilities: torch.Tensor | None = None
    pbit_entropy: torch.Tensor | None = None
    pbit_entropy_loss: torch.Tensor | None = None
    homeostasis_loss: torch.Tensor | None = None
    capacity_loss: torch.Tensor | None = None
    plastic_loss: torch.Tensor | None = None


class EgoMuscleModel(nn.Module):
    def __init__(
        self,
        video_model_name: str = "MCG-NJU/videomae-base",
        muscle_dim: int = 20,
        prediction_dim: int = 20,
        muscle_hidden_dim: int = 128,
        fusion_mode: str = "cross_attn",
        use_video: bool = True,
        use_muscle: bool = True,
        label_conditioning: bool = False,
        label_vocab_size: int = 512,
        scramble_video: bool = False,
        video_trainable_strategy: str = "frozen",
        video_trainable_layers: int = 0,
        video_unfreeze_embeddings: bool = False,
        fusion_dropout: float = 0.1,
        pred_dropout: float = 0.1,
        predictive_distribution: str = "point",
        video_latent_prediction: bool = False,
        fast_memory: dict[str, Any] | None = None,
        pbit: dict[str, Any] | None = None,
        slow_adapter: dict[str, Any] | None = None,
        need_weights: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not use_video and not use_muscle and not label_conditioning:
            raise ValueError("At least one conditioning stream must be enabled.")

        self.use_video = use_video
        self.use_muscle = use_muscle
        self.label_conditioning = label_conditioning
        self.fusion_mode = fusion_mode
        self.scramble_video = scramble_video
        self.prediction_dim = prediction_dim
        self.predictive_distribution = predictive_distribution
        self.video_latent_prediction = bool(video_latent_prediction)
        self._need_weights = need_weights

        self.video_encoder = (
            FrozenVideoEncoder(
                video_model_name,
                trainable_strategy=video_trainable_strategy,
                trainable_layers=video_trainable_layers,
                unfreeze_embeddings=video_unfreeze_embeddings,
            )
            if use_video
            else None
        )
        vision_dim = self.video_encoder.hidden_dim if self.video_encoder is not None else muscle_hidden_dim

        self.muscle_encoder = MuscleEncoder(input_dim=muscle_dim, hidden_dim=muscle_hidden_dim, gradient_checkpointing=gradient_checkpointing) if use_muscle else None
        self.label_embedding = nn.Embedding(label_vocab_size, muscle_hidden_dim) if label_conditioning else None

        self.fusion: CrossAttentionFusion | None = None
        self.late_fusion: LateFusionBlock | None = None
        self.sequence_proj: nn.Module | None = None
        self._gradient_checkpointing = gradient_checkpointing

        if use_video and (use_muscle or label_conditioning):
            if fusion_mode == "cross_attn":
                self.fusion = CrossAttentionFusion(vision_dim=vision_dim, muscle_dim=muscle_hidden_dim, dropout=fusion_dropout, need_weights=need_weights)
                predictor_dim = vision_dim
            elif fusion_mode == "late":
                self.late_fusion = LateFusionBlock(vision_dim=vision_dim, muscle_dim=muscle_hidden_dim)
                predictor_dim = vision_dim
            else:
                raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        elif use_video:
            predictor_dim = vision_dim
        else:
            predictor_dim = muscle_hidden_dim
            self.sequence_proj = nn.Sequential(
                nn.LayerNorm(muscle_hidden_dim),
                nn.Linear(muscle_hidden_dim, muscle_hidden_dim),
                nn.GELU(),
                nn.Linear(muscle_hidden_dim, muscle_hidden_dim),
            )

        self.pred_drop = nn.Dropout(pred_dropout)
        self.representation_dim = predictor_dim

        fast_memory_cfg = fast_memory or {}
        self.fast_memory_enabled = bool(fast_memory_cfg.get("enabled", False))
        self.fast_memory: FastStateEncoder | None = None
        fast_state_dim = int(fast_memory_cfg.get("num_slots", 8)) * int(fast_memory_cfg.get("slot_dim", 128))
        if self.fast_memory_enabled:
            fast_type = fast_memory_cfg.get("type", "gru_slots")
            if fast_type != "gru_slots":
                raise ValueError(f"Unsupported fast_memory.type: {fast_type}")
            self.fast_memory = FastStateEncoder(
                input_dim=predictor_dim,
                state_dim=fast_state_dim,
                dropout=float(fast_memory_cfg.get("dropout", 0.1)),
                learned_decay=bool(fast_memory_cfg.get("learned_decay", True)),
                decay_init=float(fast_memory_cfg.get("decay_init", 0.90)),
            )
            self.fast_state_proj = nn.Linear(fast_state_dim, predictor_dim)
        else:
            self.fast_state_proj = None

        pbit_cfg = pbit or {}
        self.pbit_enabled = bool(pbit_cfg.get("enabled", False))
        self.pbit_target_entropy = float(pbit_cfg.get("target_entropy", 0.4))
        self.pbit: PBitBottleneck | None = None
        if self.pbit_enabled:
            pbit_input_dim = fast_state_dim if self.fast_memory_enabled else predictor_dim
            self.pbit = PBitBottleneck(
                input_dim=pbit_input_dim,
                output_dim=predictor_dim,
                num_bits=int(pbit_cfg.get("num_bits", 256)),
                temperature=float(pbit_cfg.get("temperature", 0.67)),
                mode=str(pbit_cfg.get("mode", "stochastic_straight_through")),
            )

        slow_adapter_cfg = slow_adapter or {}
        adapter_enabled = bool(slow_adapter_cfg.get("enabled", False))
        adapter_rank = int(slow_adapter_cfg.get("rank", 0)) if adapter_enabled else 0
        adapter_levels = slow_adapter_cfg.get("quantization_levels")
        adapter_mode = str(slow_adapter_cfg.get("quantization_mode", "none"))
        target_modules = set(slow_adapter_cfg.get("target_modules", ["predictor", "video_latent_head"]))

        predictor_out_dim = prediction_dim * 2 if predictive_distribution == "gaussian" else prediction_dim
        predictor = nn.Linear(predictor_dim, predictor_out_dim)
        if adapter_enabled and "predictor" in target_modules:
            predictor = AdapterLinear.from_linear(
                predictor,
                rank=adapter_rank,
                quantization_levels=adapter_levels,
                quantization_mode=adapter_mode,
            )
        self.predictor = predictor

        self.video_latent_head: nn.Module | None = None
        if self.video_latent_prediction:
            latent_head: nn.Module = nn.Linear(predictor_dim, vision_dim)
            if adapter_enabled and "video_latent_head" in target_modules:
                latent_head = AdapterLinear.from_linear(
                    latent_head,
                    rank=adapter_rank,
                    quantization_levels=adapter_levels,
                    quantization_mode=adapter_mode,
                )
            self.video_latent_head = latent_head

        if adapter_enabled and "fusion" in target_modules and self.fusion is not None:
            self.fusion.muscle_proj = AdapterLinear.from_linear(
                self.fusion.muscle_proj,
                rank=adapter_rank,
                quantization_levels=adapter_levels,
                quantization_mode=adapter_mode,
            )

    def _mask_split(self, seq_len: int, mask_ratio: float) -> int:
        split = int(seq_len * (1.0 - mask_ratio))
        return min(max(split, 1), seq_len - 1)

    def _encode_conditioning(
        self,
        muscle: torch.Tensor | None,
        activity_ids: torch.Tensor | None,
        seq_len: int,
        t_split: int,
    ) -> torch.Tensor | None:
        if self.label_conditioning:
            if activity_ids is None or torch.any(activity_ids < 0):
                raise ValueError("label_conditioning=True requires valid non-negative activity_ids.")
            return self.label_embedding(activity_ids).unsqueeze(1).expand(-1, seq_len, -1)

        if self.muscle_encoder is None or muscle is None:
            return None

        visible = muscle.clone()
        visible[:, t_split:] = 0
        return self.muscle_encoder(visible)

    def forward(
        self,
        frames: torch.Tensor | None = None,
        muscle: torch.Tensor | None = None,
        activity_ids: torch.Tensor | None = None,
        mask_ratio: float = 0.5,
        return_layerwise_video: bool = False,
        video_features: torch.Tensor | None = None,
    ) -> EgoMuscleOutput:
        seq_len = 0
        if frames is not None:
            seq_len = frames.shape[1]
        elif muscle is not None:
            seq_len = muscle.shape[1]
        elif video_features is not None:
            seq_len = video_features.shape[1]
        else:
            raise ValueError("Either frames, video_features, or muscle inputs are required.")

        if self.scramble_video and frames is not None:
            permutation = torch.randperm(frames.shape[1], device=frames.device)
            frames = frames[:, permutation]

        t_split = self._mask_split(seq_len, mask_ratio)
        target = muscle[:, t_split:] if muscle is not None else None

        layerwise_video_repr: dict[str, torch.Tensor] | None = None
        if video_features is not None:
            video_repr = self.video_encoder.adapter(video_features) if self.video_encoder is not None else None
        elif self.video_encoder is not None and frames is not None:
            if return_layerwise_video:
                video_repr, layerwise_video_repr = self.video_encoder(frames, return_layerwise=True)
            else:
                video_repr = self.video_encoder(frames)
        else:
            video_repr = None
        conditioning = self._encode_conditioning(muscle, activity_ids, seq_len, t_split)

        attention = None
        if video_repr is not None and conditioning is not None:
            if self.fusion is not None:
                fused, attention = self.fusion(video_repr, conditioning)
            else:
                fused = self.late_fusion(video_repr, conditioning)
        elif video_repr is not None:
            fused = video_repr
        elif conditioning is not None:
            fused = self.sequence_proj(conditioning) if self.sequence_proj is not None else conditioning
        else:
            raise ValueError("No representation stream is available for fusion.")

        model_repr = fused
        fast_state = None
        fast_mu = None
        fast_log_var = None
        fast_kl = None
        pbit_probabilities = None
        pbit_entropy = None
        pbit_entropy_loss = None
        homeostasis_loss = None

        if self.fast_memory is not None:
            fast_output = self.fast_memory(fused)
            fast_state = fast_output.state
            fast_mu = fast_output.mu
            fast_log_var = fast_output.log_var
            fast_kl = fast_output.kl
            assert self.fast_state_proj is not None
            model_repr = model_repr + self.fast_state_proj(fast_state)
            homeostasis_loss = (fast_state.abs().mean() - 0.5).abs()

        if self.pbit is not None:
            pbit_input = fast_state if fast_state is not None else fused
            pbit_output = self.pbit(pbit_input)
            model_repr = model_repr + pbit_output.out
            pbit_probabilities = pbit_output.probabilities
            pbit_entropy = pbit_output.entropy_bits
            pbit_entropy_loss = (pbit_entropy.mean() - self.pbit_target_entropy).abs()
            homeostasis_pbit = (pbit_probabilities.mean() - 0.5).abs()
            homeostasis_loss = homeostasis_pbit if homeostasis_loss is None else 0.5 * (homeostasis_loss + homeostasis_pbit)

        raw_pred = self.predictor(self.pred_drop(model_repr[:, t_split:])) if target is not None else None
        pred_mu = None
        pred_log_var = None
        pred = raw_pred
        if raw_pred is not None and self.predictive_distribution == "gaussian":
            pred_mu, pred_log_var = raw_pred.chunk(2, dim=-1)
            pred_log_var = pred_log_var.clamp(-8.0, 4.0)
            pred = pred_mu
        elif raw_pred is not None:
            pred_mu = raw_pred

        pred_video_latent = None
        video_latent_target = None
        if self.video_latent_head is not None and video_repr is not None:
            pred_video_latent = self.video_latent_head(self.pred_drop(model_repr[:, t_split:]))
            video_latent_target = video_repr[:, t_split:].detach()

        capacity_loss, plastic_loss = self.adapter_regularization()
        return EgoMuscleOutput(
            pred=pred,
            target=target,
            attention=attention,
            fused=model_repr,
            pooled=model_repr.mean(dim=1),
            video_repr=video_repr,
            conditioning_repr=conditioning,
            layerwise_video_repr=layerwise_video_repr,
            pred_mu=pred_mu,
            pred_log_var=pred_log_var,
            pred_video_latent=pred_video_latent,
            video_latent_target=video_latent_target,
            fast_state=fast_state,
            fast_mu=fast_mu,
            fast_log_var=fast_log_var,
            fast_kl=fast_kl,
            pbit_probabilities=pbit_probabilities,
            pbit_entropy=pbit_entropy,
            pbit_entropy_loss=pbit_entropy_loss,
            homeostasis_loss=homeostasis_loss,
            capacity_loss=capacity_loss,
            plastic_loss=plastic_loss,
        )

    def adapter_regularization(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        capacity_terms = []
        plastic_terms = []
        for module in self.modules():
            if isinstance(module, AdapterLinear):
                capacity_terms.append(module.capacity_penalty())
                plastic_terms.append(module.plastic_penalty())
        if not capacity_terms:
            return None, None
        return torch.stack(capacity_terms).mean(), torch.stack(plastic_terms).mean()

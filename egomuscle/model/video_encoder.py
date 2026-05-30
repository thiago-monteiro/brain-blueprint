from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
import torch.utils.checkpoint as cp


class VideoAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _get_hidden_dim_from_config(cfg) -> int:
    for key in ("hidden_size", "embed_dim", "hidden_dim", "d_model"):
        val = getattr(cfg, key, None)
        if val is not None:
            return int(val)
        model_conf = getattr(cfg, "model_config", None)
        if isinstance(model_conf, dict) and key in model_conf:
            return int(model_conf[key])
    raise AttributeError("VideoMAE config does not expose a known hidden-dim attribute")


def _resolve_image_size(image_size: int | tuple[int, int]) -> int:
    return image_size if isinstance(image_size, int) else image_size[0]


def _resolve_transformer_blocks(encoder: nn.Module) -> nn.ModuleList | None:
    inner = getattr(encoder, "encoder", None)
    if inner is not None:
        blocks = getattr(inner, "layer", None)
        if blocks is not None:
            return blocks
    vision = getattr(encoder, "model", None)
    if vision is not None:
        blocks = getattr(vision, "blocks", None)
        if blocks is not None:
            return blocks
    return None


def _is_videomaev2(encoder: nn.Module) -> bool:
    vision = getattr(encoder, "model", None)
    return vision is not None and getattr(vision, "blocks", None) is not None


def _frames_to_v2_pixel_values(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 5:
        raise ValueError(f"Expected 5D frame tensor, got shape {tuple(frames.shape)}")
    return frames.permute(0, 2, 1, 3, 4).contiguous()


class _VideoTokenOutput(NamedTuple):
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None


def _encode_v2_tokens(
    vision: nn.Module,
    pixel_values: torch.Tensor,
    *,
    return_layerwise: bool,
) -> _VideoTokenOutput:
    batch_size = pixel_values.size(0)
    hidden = vision.patch_embed(pixel_values)
    if vision.pos_embed is not None:
        pos_embed = vision.pos_embed.expand(batch_size, -1, -1).type_as(hidden).to(hidden.device).clone().detach()
        hidden = hidden + pos_embed
    hidden = vision.pos_drop(hidden)

    layerwise: list[torch.Tensor] | None = [hidden] if return_layerwise else None
    for block in vision.blocks:
        if vision.with_cp:
            hidden = cp.checkpoint(block, hidden)
        else:
            hidden = block(hidden)
        if layerwise is not None:
            layerwise.append(hidden)

    states = tuple(layerwise) if layerwise is not None else None
    return _VideoTokenOutput(last_hidden_state=hidden, hidden_states=states)


def _encode_video_tokens(
    encoder: nn.Module,
    frames: torch.Tensor,
    *,
    return_layerwise: bool,
) -> _VideoTokenOutput:
    if _is_videomaev2(encoder):
        pixel_values = _frames_to_v2_pixel_values(frames)
        return _encode_v2_tokens(encoder.model, pixel_values, return_layerwise=return_layerwise)

    outputs = encoder(pixel_values=frames, output_hidden_states=return_layerwise)
    hidden_states = tuple(outputs.hidden_states) if return_layerwise and outputs.hidden_states is not None else None
    return _VideoTokenOutput(last_hidden_state=outputs.last_hidden_state, hidden_states=hidden_states)


def _unfreeze_video_embeddings(encoder: nn.Module) -> None:
    embeddings = getattr(encoder, "embeddings", None)
    if embeddings is not None:
        for param in embeddings.parameters():
            param.requires_grad = True
        return

    vision = getattr(encoder, "model", None)
    if vision is None:
        return
    for name in ("patch_embed", "pos_embed"):
        module = getattr(vision, name, None)
        if module is None:
            continue
        if isinstance(module, (nn.Parameter, torch.Tensor)):
            module.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = True


def pool_video_tokens(
    hidden_states: torch.Tensor,
    target_frames: int,
    image_size: int,
    patch_size: int,
    tubelet_size: int,
) -> torch.Tensor:
    sequence = hidden_states
    spatial_tokens = (_resolve_image_size(image_size) // patch_size) ** 2

    if sequence.shape[1] % spatial_tokens == 1:
        sequence = sequence[:, 1:]

    if sequence.shape[1] % spatial_tokens != 0:
        temporal_tokens = max(1, min(target_frames, sequence.shape[1]))
        trimmed = (sequence.shape[1] // temporal_tokens) * temporal_tokens
        sequence = sequence[:, :trimmed]
        spatial_tokens = max(1, sequence.shape[1] // temporal_tokens)
    else:
        temporal_tokens = max(1, sequence.shape[1] // spatial_tokens)

    pooled = sequence.reshape(sequence.shape[0], temporal_tokens, spatial_tokens, sequence.shape[-1]).mean(dim=2)

    if temporal_tokens == target_frames:
        return pooled

    pooled = pooled.transpose(1, 2)
    pooled = F.interpolate(pooled, size=target_frames, mode="linear", align_corners=False)
    return pooled.transpose(1, 2)


class FrozenVideoEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        freeze: bool = True,
        trainable_strategy: str = "frozen",
        trainable_layers: int = 0,
        unfreeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_dim = _get_hidden_dim_from_config(self.encoder.config)
        self.adapter = VideoAdapter(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.tubelet_size = getattr(self.encoder.config, "tubelet_size", 2)
        self.patch_size = getattr(self.encoder.config, "patch_size", 16)
        self.image_size = getattr(self.encoder.config, "image_size", 224)
        self.trainable_strategy = trainable_strategy
        self.trainable_layers = int(trainable_layers)
        self.unfreeze_embeddings = bool(unfreeze_embeddings)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self._apply_trainable_strategy()

    def _freeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_all(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def _apply_trainable_strategy(self) -> None:
        strategy = str(self.trainable_strategy).lower()
        if strategy in {"frozen", "adapter_only"}:
            self._freeze_backbone()
            return
        if strategy in {"full", "all"}:
            self._unfreeze_all()
            return
        if strategy != "last_n":
            raise ValueError(f"Unsupported video trainable strategy: {self.trainable_strategy}")

        self._freeze_backbone()
        blocks = _resolve_transformer_blocks(self.encoder)
        if blocks is None:
            raise AttributeError(
                "Video encoder does not expose transformer blocks "
                "(expected encoder.encoder.layer or model.blocks)."
            )
        n_blocks = len(blocks)
        n_trainable = max(0, min(int(self.trainable_layers), n_blocks))
        if n_trainable == 0:
            return
        for block in blocks[n_blocks - n_trainable :]:
            for param in block.parameters():
                param.requires_grad = True
        if self.unfreeze_embeddings:
            _unfreeze_video_embeddings(self.encoder)

    def forward(
        self,
        frames: torch.Tensor,
        *,
        return_layerwise: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs = _encode_video_tokens(self.encoder, frames, return_layerwise=return_layerwise)
        pooled = pool_video_tokens(
            outputs.last_hidden_state,
            target_frames=frames.shape[1],
            image_size=self.image_size,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
        )
        adapted = self.adapter(pooled)
        if not return_layerwise:
            return adapted

        layerwise: dict[str, torch.Tensor] = {}
        hidden_states = outputs.hidden_states or ()
        for idx, hidden in enumerate(hidden_states):
            layerwise[f"layer_{idx:02d}"] = pool_video_tokens(
                hidden,
                target_frames=frames.shape[1],
                image_size=self.image_size,
                patch_size=self.patch_size,
                tubelet_size=self.tubelet_size,
            )
        layerwise["adapted_final"] = adapted
        return adapted, layerwise

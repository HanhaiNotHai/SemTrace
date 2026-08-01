from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from transformers import AutoModel

from semtrace.backbones.base import BackboneOutput, split_backbone_tokens

# Token layout and config fields:
# https://huggingface.co/docs/transformers/model_doc/dinov3#notes


class DINOv3Backbone(nn.Module):
    """Frozen Transformers DINOv3 ViT wrapper that captures only requested blocks."""

    model: Any
    config: Any

    def __init__(
        self,
        model: nn.Module,
        selected_layers: Iterable[int],
        *,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_id = model_id
        self.revision = revision
        self.config: Any = model.config
        if getattr(self.config, "model_type", None) != "dinov3_vit":
            raise ValueError("DINOv3Backbone requires a Transformers DINOv3 ViT model")
        self.hidden_size = int(self.config.hidden_size)
        self.patch_size = int(self.config.patch_size)
        self.num_layers = int(self.config.num_hidden_layers)
        self.num_register_tokens = int(self.config.num_register_tokens)
        self.selected_layers = tuple(sorted(set(selected_layers)))
        if any(index < 0 or index >= self.num_layers for index in self.selected_layers):
            raise ValueError(f"selected layers must be in [0, {self.num_layers - 1}]")

        self._captured: dict[int, torch.Tensor] = {}
        self._hook_handles: list[RemovableHandle] = []
        encoder_layers = self.model.model.layer
        for index in self.selected_layers:
            self._hook_handles.append(encoder_layers[index].register_forward_hook(self._hook(index)))
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise AssertionError("DINOv3 backbone parameters must be frozen")

    @classmethod
    def from_pretrained(
        cls,
        model_source: str | Path,
        selected_layers: Iterable[int],
        *,
        model_id: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> DINOv3Backbone:
        source = str(model_source)
        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None and not Path(source).exists():
            kwargs["revision"] = revision
        model = AutoModel.from_pretrained(source, **kwargs)
        return cls(model, selected_layers, model_id=model_id or source, revision=revision)

    def train(self, mode: bool = True) -> DINOv3Backbone:
        super().train(False)
        self.model.train(False)
        return self

    def forward(self, pixel_values: torch.Tensor) -> BackboneOutput:
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("input height and width must be divisible by DINOv3 patch_size")
        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        self._captured.clear()
        with torch.no_grad():
            model_output = self.model(pixel_values=pixel_values, return_dict=True)
            if set(self._captured) != set(self.selected_layers):
                raise RuntimeError("not all requested DINOv3 layers were captured")
            semantic_cls, final_patches = split_backbone_tokens(
                model_output.last_hidden_state,
                self.num_register_tokens,
                patch_grid_size,
            )
            intermediate_patches: dict[int, torch.Tensor] = {}
            for index, hidden_state in self._captured.items():
                # Transformers hidden states are post-block but pre-final-LayerNorm.
                normalized = self.model.norm(hidden_state)
                _, patches = split_backbone_tokens(
                    normalized,
                    self.num_register_tokens,
                    patch_grid_size,
                )
                intermediate_patches[index] = patches
        return BackboneOutput(
            semantic_cls=semantic_cls,
            final_patch_tokens=final_patches,
            intermediate_patch_tokens=intermediate_patches,
            patch_grid_size=patch_grid_size,
        )

    def _hook(self, index: int) -> Any:
        def capture(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            hidden_state = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden_state, torch.Tensor):
                raise TypeError("DINOv3 block hook did not receive a tensor")
            self._captured[index] = hidden_state

        return capture

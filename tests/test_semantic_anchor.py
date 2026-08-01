from __future__ import annotations

import torch

from semtrace.models.semantic_anchor import FrozenSemanticAnchor


def test_semantic_anchor_concatenates_cls_and_patch_mean_and_stays_frozen() -> None:
    anchor = FrozenSemanticAnchor(backbone_dim=6, semantic_dim=4, seed=7)
    semantic_cls = torch.randn(2, 6, requires_grad=True)
    final_patches = torch.randn(2, 5, 6, requires_grad=True)

    output = anchor(semantic_cls, final_patches)
    output.square().mean().backward()

    assert output.shape == (2, 4)
    assert not any(parameter.requires_grad for parameter in anchor.parameters())
    assert all(parameter.grad is None for parameter in anchor.parameters())

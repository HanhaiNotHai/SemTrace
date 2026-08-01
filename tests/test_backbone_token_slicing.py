import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from semtrace.backbones.base import BackboneOutput, split_backbone_tokens
from semtrace.backbones.dinov3 import DINOv3Backbone


def test_token_slicing_excludes_cls_and_register_tokens() -> None:
    sequence = torch.arange(11, dtype=torch.float32).view(1, 11, 1)

    semantic_cls, patches = split_backbone_tokens(
        sequence,
        num_register_tokens=4,
        patch_grid_size=(2, 3),
    )

    assert semantic_cls.item() == 0
    assert patches.flatten().tolist() == [5, 6, 7, 8, 9, 10]


def test_token_slicing_rejects_inconsistent_grid() -> None:
    with pytest.raises(ValueError, match="patch token count"):
        split_backbone_tokens(torch.zeros(1, 11, 4), 4, (2, 4))


def test_dinov3_wrapper_returns_requested_normalized_layers_on_rectangular_grid() -> None:
    model = _tiny_dinov3(num_register_tokens=2)
    backbone = DINOv3Backbone(model, selected_layers=[0, 1])

    output = backbone(torch.randn(2, 3, 32, 48))

    assert isinstance(output, BackboneOutput)
    assert output.semantic_cls.shape == (2, 32)
    assert output.final_patch_tokens.shape == (2, 6, 32)
    assert output.patch_grid_size == (2, 3)
    assert set(output.intermediate_patch_tokens) == {0, 1}
    for tokens in output.intermediate_patch_tokens.values():
        assert tokens.shape == (2, 6, 32)
        assert torch.allclose(tokens.mean(dim=-1), torch.zeros(2, 6), atol=1e-5)


def _tiny_dinov3(num_register_tokens: int) -> DINOv3ViTModel:
    config = DINOv3ViTConfig(
        image_size=32,
        patch_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_register_tokens=num_register_tokens,
    )
    return DINOv3ViTModel(config)

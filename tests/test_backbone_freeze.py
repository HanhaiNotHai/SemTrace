import torch
from torch import nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from semtrace.backbones.dinov3 import DINOv3Backbone


def test_frozen_backbone_stays_eval_and_receives_no_gradients() -> None:
    config = DINOv3ViTConfig(
        image_size=32,
        patch_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_register_tokens=4,
    )
    backbone = DINOv3Backbone(DINOv3ViTModel(config), selected_layers=[0])
    head = nn.Linear(16, 1)

    backbone.train()
    logits = head(backbone(torch.randn(2, 3, 32, 32)).final_patch_tokens.mean(dim=1))
    logits.sum().backward()

    assert backbone.training is False
    assert backbone.model.training is False
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert all(parameter.grad is not None for parameter in head.parameters())

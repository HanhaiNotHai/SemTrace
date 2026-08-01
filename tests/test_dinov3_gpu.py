from __future__ import annotations

from pathlib import Path

import pytest
import torch

from semtrace.backbones.dinov3 import DINOv3Backbone
from semtrace.config import compose_config

MODEL_PATH = Path(str(compose_config("stage3_detector").model.model_path))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason="local DINOv3 snapshot is unavailable")
def test_local_dinov3_real_forward() -> None:
    backbone = DINOv3Backbone.from_pretrained(
        MODEL_PATH,
        selected_layers=(2, 6, 10),
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        revision="master",
        local_files_only=True,
    ).to("cuda")

    output = backbone(torch.randn(1, 3, 128, 128, device="cuda"))

    assert output.semantic_cls.shape == (1, 768)
    assert output.final_patch_tokens.shape == (1, 64, 768)
    assert output.patch_grid_size == (8, 8)
    assert set(output.intermediate_patch_tokens) == {2, 6, 10}
    assert backbone.num_register_tokens == 4
    assert not any(parameter.requires_grad for parameter in backbone.parameters())

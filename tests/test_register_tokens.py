import pytest
import torch

from semtrace.backbones.base import split_backbone_tokens


@pytest.mark.parametrize("num_register_tokens", [0, 4])
def test_register_count_is_read_before_patch_slicing(num_register_tokens: int) -> None:
    patches = torch.full((1, 4, 2), 7.0)
    prefix = torch.zeros(1, 1 + num_register_tokens, 2)
    sequence = torch.cat([prefix, patches], dim=1)

    _, actual_patches = split_backbone_tokens(sequence, num_register_tokens, (2, 2))

    assert torch.equal(actual_patches, patches)

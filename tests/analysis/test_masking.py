import torch

from semtrace.analysis.scale_masking import mask_patches, mask_scales


def test_masking_replaces_only_requested_scale() -> None:
    scales = {2: torch.ones(1, 4, 2), 6: torch.ones(1, 4, 2) * 2}

    masked = mask_scales(scales, masked_layers={6})

    torch.testing.assert_close(masked[2], scales[2])
    assert torch.count_nonzero(masked[6]) == 0


def test_patch_masking_replaces_exact_requested_ratio() -> None:
    tokens = torch.ones(2, 20, 3)

    masked, mask = mask_patches(tokens, ratio=0.2, strategy="random", seed=3)

    assert mask.sum(dim=1).tolist() == [4, 4]
    assert torch.count_nonzero(masked[mask]) == 0
    torch.testing.assert_close(masked[~mask], tokens[~mask])

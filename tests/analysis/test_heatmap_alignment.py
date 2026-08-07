import torch

from semtrace.analysis.visualize_trace_maps import patch_map, resize_patch_map


def test_heatmap_uses_patch_grid_row_major_alignment() -> None:
    values = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    grid = patch_map(values, (2, 3))
    resized = resize_patch_map(grid, (8, 12))

    torch.testing.assert_close(grid, torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]))
    assert resized.shape == (1, 8, 12)

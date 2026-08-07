import torch

from semtrace.analysis.semantic_counterfactual import MechanismState, swap_trace


def test_trace_swap_does_not_swap_semantic_state() -> None:
    state = MechanismState(
        semantic_anchor=torch.tensor([[1.0], [2.0]]),
        trace_tokens=torch.tensor([[[10.0]], [[20.0]]]),
    )

    swapped = swap_trace(state, torch.tensor([1, 0]))

    torch.testing.assert_close(swapped.semantic_anchor, state.semantic_anchor)
    assert swapped.trace_tokens[:, 0, 0].tolist() == [20.0, 10.0]

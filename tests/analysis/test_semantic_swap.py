import torch

from semtrace.analysis.semantic_counterfactual import MechanismState, swap_semantic


def test_semantic_swap_does_not_swap_trace_state() -> None:
    state = MechanismState(
        semantic_anchor=torch.tensor([[1.0], [2.0]]),
        trace_tokens=torch.tensor([[[10.0]], [[20.0]]]),
    )

    swapped = swap_semantic(state, torch.tensor([1, 0]))

    assert swapped.semantic_anchor[:, 0].tolist() == [2.0, 1.0]
    torch.testing.assert_close(swapped.trace_tokens, state.trace_tokens)

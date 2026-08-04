from semtrace.cli import evaluate


def test_evaluate_progress_labels_domains_and_is_rank_safe() -> None:
    assert evaluate._evaluation_progress(
        "self_synthesis",
        position=1,
        total=1,
        is_main_process=True,
    ) == ("Evaluating self_synthesis (1/1)", True)
    assert evaluate._evaluation_progress(
        "biggan",
        position=8,
        total=8,
        is_main_process=False,
    ) == ("Evaluating biggan (8/8)", False)

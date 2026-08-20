from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from semtrace.analysis.proposal_package import build_proposal_package
from semtrace.config import parse_config_args


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_config_args(
        argv,
        description=(
            "Build the reproducible SemTrace proposal mechanism package from existing "
            "mechanism cache and evaluation outputs."
        ),
        default_config_name="analysis/proposal_mechanisms",
    )
    if float(config.confidence_level) != 0.95:
        raise ValueError("the current statistical implementation supports confidence_level=0.95")
    seeds = OmegaConf.to_container(config.random_seeds, resolve=True)
    formats = OmegaConf.to_container(config.image_formats, resolve=True)
    if not isinstance(seeds, list) or not isinstance(formats, list):
        raise TypeError("random_seeds and image_formats must be lists")
    package = build_proposal_package(
        mechanism_root=str(config.mechanism_root),
        eval_root=str(config.eval_root),
        checkpoint=str(config.checkpoint),
        selected_layers_path=str(config.selected_layers_path),
        dataset=str(config.dataset),
        protocol=str(config.protocol),
        output_root=str(config.output_root),
        output_dir=(Path(str(config.output_dir)) if config.output_dir is not None else None),
        random_seeds=tuple(int(value) for value in seeds),
        bootstrap_iterations=int(config.bootstrap_iterations),
        dpi=int(config.dpi),
        image_formats=tuple(str(value) for value in formats),
        head_batch_size=int(config.head_batch_size),
        device=str(config.device),
    )
    print(f"proposal mechanism package: {package}")


if __name__ == "__main__":
    main()

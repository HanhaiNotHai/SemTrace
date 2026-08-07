from __future__ import annotations

import sys
from collections.abc import Sequence

from semtrace.analysis.runner import run_analysis_cli


def main(argv: Sequence[str] | None = None) -> int:
    return run_analysis_cli("mechanism_suite", argv)


if __name__ == "__main__":
    sys.exit(main())

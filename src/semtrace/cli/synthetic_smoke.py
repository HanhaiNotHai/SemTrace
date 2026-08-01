from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict

from semtrace.smoke import run_synthetic_smoke


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline SemTrace smoke workflow.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    arguments = parser.parse_args(argv)
    result = run_synthetic_smoke(arguments.output_dir, seed=arguments.seed)
    payload = asdict(result)
    payload["checkpoint_path"] = str(result.checkpoint_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

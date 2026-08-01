from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from semtrace.config import parse_config_args
from semtrace.data.manifest import protocol_scan_rules, scan_manifest, write_manifest


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config_args(
        argv,
        description="Build a SemTrace JSONL manifest from an official dataset layout.",
        default_config_name="manifest",
    )
    if config.data.root is None:
        raise ValueError("data.root must point to an installed dataset")
    output = Path(str(config.manifest.output))
    records, audit = scan_manifest(
        root=str(config.data.root),
        rules=protocol_scan_rules(str(config.protocol.name)),
        minimum_size=int(config.preprocessing.crop_size),
        small_image_policy=str(config.preprocessing.small_image_policy),
    )
    write_manifest(records, output)
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(asdict(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(output), **asdict(audit)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

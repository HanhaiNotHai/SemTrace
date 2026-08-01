#!/usr/bin/env bash
set -euo pipefail

uv run torchrun --standalone --nproc_per_node=4 \
  -m semtrace.cli.evaluate --config-name eval \
  protocol=forensynths_progan4 "$@"


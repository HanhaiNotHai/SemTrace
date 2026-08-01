#!/usr/bin/env bash
set -euo pipefail

uv run torchrun --standalone --nproc_per_node=4 \
  -m semtrace.cli.train_normal --config-name stage2_normal "$@"


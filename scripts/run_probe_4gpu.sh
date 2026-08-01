#!/usr/bin/env bash
set -euo pipefail

uv run torchrun --standalone --nproc_per_node=4 \
  -m semtrace.cli.probe_layers --config-name stage1_probe "$@"


#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m semtrace.cli.probe_layers \
    --config-name stage1_probe \
    protocol=genimage_sdv14 \
    data.train_manifest=artifacts/manifests/genimage_sdv14.jsonl \
    data.validation_manifest=artifacts/manifests/genimage_sdv14.jsonl \
    probe.batch_size=256 \
    training.num_workers=4 \
    |& tee scripts/run_probe_4gpu.log

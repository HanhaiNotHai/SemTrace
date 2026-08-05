#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m semtrace.cli.probe_layers \
    --config-name stage1_probe \
    protocol=forensynths_progan4 \
    data.train_manifest=artifacts/manifests/forensynths_progan4.jsonl \
    data.validation_manifest=artifacts/manifests/forensynths_progan4.jsonl \
    probe.batch_size=256 \
    probe.selected_layers_path=artifacts/probes_forensynths_progan4/selected_layers.json \
    probe.semantic_anchor_path=artifacts/probes_forensynths_progan4/semantic_anchor.pt \
    probe.output_dir=artifacts/probes_forensynths_progan4 \
    training.num_workers=4 \
    |& tee scripts/run_probe_4gpu_forensynths_progan4.log

#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    uv run torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m semtrace.cli.train_detector \
    --config-name stage3_detector \
    protocol=forensynths_progan4 \
    data.root=/data/zhy/CNNDetection/dataset \
    data.train_manifest=artifacts/manifests/forensynths_progan4.jsonl \
    data.validation_manifest=artifacts/manifests/forensynths_progan4.jsonl \
    probe.selected_layers_path=artifacts/probes_forensynths_progan4/selected_layers.json \
    probe.semantic_anchor_path=artifacts/probes_forensynths_progan4/semantic_anchor.pt \
    normal.checkpoint=outputs/stage2_normal/20260803T164716Z/checkpoints/normal_best.pt \
    training.per_device_batch_size=256 \
    training.gradient_accumulation_steps=1 \
    training.target_global_batch_size=1024 \
    training.num_workers=4 \
    training.amp=bf16 \
    2>&1 | tee scripts/run_detector_4gpu_forensynths_progan4.log

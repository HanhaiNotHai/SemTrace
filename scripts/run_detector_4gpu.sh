#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    uv run torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m semtrace.cli.train_detector \
    --config-name stage3_detector \
    protocol=genimage_sdv14 \
    data.root=/data/zhy/GenImage \
    data.train_manifest=artifacts/manifests/genimage_sdv14.jsonl \
    data.validation_manifest=artifacts/manifests/genimage_sdv14.jsonl \
    probe.selected_layers_path=artifacts/probes/selected_layers.json \
    probe.semantic_anchor_path=artifacts/probes/semantic_anchor.pt \
    normal.checkpoint=outputs/stage2_normal/20260803T113850Z/checkpoints/normal_best.pt \
    training.per_device_batch_size=32 \
    training.gradient_accumulation_steps=1 \
    training.target_global_batch_size=128 \
    training.num_workers=4 \
    training.amp=bf16 \
    2>&1 | tee scripts/run_detector_4gpu.log

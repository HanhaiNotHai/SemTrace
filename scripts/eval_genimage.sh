#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    uv run torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m semtrace.cli.evaluate \
    --config-name eval \
    protocol=genimage_sdv14 \
    experiment=genimage_all_generators_eval \
    checkpoint=outputs/stage3_detector/20260803T144441Z/checkpoints/semtrace_best.pt \
    normal.checkpoint=outputs/stage2_normal/20260803T113850Z/checkpoints/normal_best.pt \
    probe.selected_layers_path=artifacts/probes/selected_layers.json \
    probe.semantic_anchor_path=artifacts/probes/semantic_anchor.pt \
    data.validation_manifest=artifacts/manifests/genimage_all_validation.jsonl \
    evaluation.per_device_batch_size=64 \
    evaluation.num_workers=4 \
    evaluation.amp=bf16 \
    evaluation.threshold=0.5 \
    2>&1 | tee scripts/eval_genimage.log
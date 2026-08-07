#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=4 \
    uv run python -m semtrace.cli.run_mechanism_suite \
    --config-name analysis/mechanism_base \
    protocol=genimage_sdv14 \
    checkpoint=outputs/stage3_detector/20260803T144441Z/checkpoints/semtrace_best.pt \
    normal.checkpoint=outputs/stage2_normal/20260803T113850Z/checkpoints/normal_best.pt \
    probe.selected_layers_path=artifacts/probes/selected_layers.json \
    probe.semantic_anchor_path=artifacts/probes/semantic_anchor.pt \
    '+data.test_manifests={genimage:artifacts/manifests/genimage_sdv14.jsonl}' \
    analysis.feature_batch_size=512 \
    analysis.max_samples_per_group=2000 \
    |& tee scripts/mechanism_genimage.log

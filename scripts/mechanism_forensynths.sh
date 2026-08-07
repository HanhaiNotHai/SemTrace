#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=4 \
    uv run python -m semtrace.cli.run_mechanism_suite \
    --config-name analysis/mechanism_base \
    protocol=self_synthesis \
    analysis.experiment_name=forensynths_self_synthesis_coverage \
    checkpoint=outputs/stage3_detector/20260803T174134Z/checkpoints/semtrace_best.pt \
    normal.checkpoint=outputs/stage2_normal/20260803T164716Z/checkpoints/normal_best.pt \
    probe.selected_layers_path=artifacts/probes_forensynths_progan4/selected_layers.json \
    probe.semantic_anchor_path=artifacts/probes_forensynths_progan4/semantic_anchor.pt \
    '+data.test_manifests={forensynths:artifacts/manifests/forensynths_progan4.jsonl,self_synthesis:artifacts/manifests/self_synthesis.jsonl}' \
    analysis.feature_batch_size=512 \
    analysis.max_samples_per_group=2000 \
    |& tee scripts/mechanism_forensynths.log

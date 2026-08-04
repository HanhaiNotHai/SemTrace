#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 \
    uv run torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m semtrace.cli.evaluate \
    --config-name eval \
    protocol=self_synthesis \
    experiment=self_synthesis_eval \
    data.root=/data/zhy/GANGen-Detection \
    checkpoint=outputs/stage3_detector/20260803T174134Z/checkpoints/semtrace_best.pt \
    normal.checkpoint=outputs/stage2_normal/20260803T164716Z/checkpoints/normal_best.pt \
    probe.selected_layers_path=artifacts/probes_forensynths_progan4/selected_layers.json \
    probe.semantic_anchor_path=artifacts/probes_forensynths_progan4/semantic_anchor.pt \
    evaluation.per_device_batch_size=64 \
    evaluation.num_workers=4 \
    evaluation.amp=bf16 \
    evaluation.threshold=0.5 \
    2>&1 | tee scripts/eval_forensynths.log

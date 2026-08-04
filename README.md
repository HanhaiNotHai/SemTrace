# SemTrace

Semantic-Conditioned Multi-Scale Trace Residual Learning for generalizable
AI-generated image detection. SemTrace freezes DINOv3 ViT-B/16, selects three
blocks with offline probes, predicts real-image normal features at each scale,
and classifies only fused candidate trace residual evidence.

## Install

Python 3.11 and the CUDA 13.0 PyTorch wheels are locked by uv:

```bash
uv sync --extra dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The default model path is the local snapshot in
`configs/model/dinov3_vitb16.yaml`; override `model.model_path=...` for another
local directory. Default tests never access the network.

## Data manifests

Supported official layouts are documented in `docs/DATA_LAYOUT.md`. Build a
manifest without resizing images:

```bash
uv run python -m semtrace.cli.build_manifest \
  --config-name manifest \
  protocol=genimage_sdv14 \
  data.root=/data/GenImage
```

The result is `artifacts/manifests/genimage_sdv14.jsonl`. Point both
`data.train_manifest` and `data.validation_manifest` to it; split filtering is
driven by each JSONL record. Small images are counted and skipped by default.

Build the combined nine-generator Self-Synthesis test manifest with:

```bash
uv run python -m semtrace.cli.build_manifest \
  --config-name manifest \
  protocol=self_synthesis \
  data.root=/data/zhy/GANGen-Detection
```

This writes `artifacts/manifests/self_synthesis.jsonl` and its audit file. It
keeps every accepted image under each generator's `0_real` and `1_fake`
directories; no sampling or cross-generator content deduplication is applied.

## Three-stage training

Stage 1 writes `probe_results.csv`, `selected_layers.json`,
`layer_score_plot.png`, and the frozen semantic projection:

```bash
uv run torchrun --standalone --nproc_per_node=4 \
  -m semtrace.cli.probe_layers --config-name stage1_probe \
  protocol=genimage_sdv14 \
  data.train_manifest=artifacts/manifests/genimage_sdv14.jsonl \
  data.validation_manifest=artifacts/manifests/genimage_sdv14.jsonl
```

Rank 0 displays separate train/validation extraction progress bars with batch
percentage, elapsed time, ETA, processed samples, and samples/second. Feature
gathering, per-layer probe fitting, layer selection, and artifact writing also
report stage completion and wall time. Redirect both terminal streams through
`2>&1 | tee stage1_probe.log` to retain these progress messages.

Stage 2 trains three independent predictors on real images only:

```bash
uv run torchrun --standalone --nproc_per_node=4 \
  -m semtrace.cli.train_normal --config-name stage2_normal \
  protocol=genimage_sdv14 \
  data.train_manifest=artifacts/manifests/genimage_sdv14.jsonl \
  data.validation_manifest=artifacts/manifests/genimage_sdv14.jsonl \
  probe.selected_layers_path=artifacts/probes/selected_layers.json
```

Rank 0 displays separate training and validation progress bars with batch ETA,
sample count, running loss, and learning rate. Each epoch summary also estimates
the remaining Stage 2 time; append `2>&1 | tee stage2_normal.log` to retain it.

Stage 3 loads and freezes those predictors:

```bash
uv run torchrun --standalone --nproc_per_node=4 \
  -m semtrace.cli.train_detector --config-name stage3_detector \
  protocol=genimage_sdv14 \
  data.train_manifest=artifacts/manifests/genimage_sdv14.jsonl \
  data.validation_manifest=artifacts/manifests/genimage_sdv14.jsonl \
  probe.selected_layers_path=artifacts/probes/selected_layers.json \
  normal.checkpoint=/path/to/normal_best.pt
```

Rank 0 displays training and validation progress with batch ETA and sample
count. The training bar also shows running loss and learning rate; epoch
summaries report validation Accuracy/AP and a resume-aware Stage 3 ETA. Append
`2>&1 | tee stage3_detector.log` to retain the output.

These configurations default to 50 and 200 epochs; do not launch them merely as
smoke tests. The strict detector protocol is 4 GPUs × batch 16 × accumulation 2
= 128. Six-GPU runs are logged as non-strict and linearly scale the learning
rate.

## Evaluation and verification

```bash
uv run torchrun --standalone --nproc_per_node=4 \
  -m semtrace.cli.evaluate --config-name eval \
  protocol=genimage_sdv14 \
  checkpoint=/path/to/semtrace_best.pt \
  normal.checkpoint=/path/to/normal_best.pt \
  probe.selected_layers_path=artifacts/probes/selected_layers.json \
  'data.test_manifests={sdv1.4:/data/manifests/sdv14_val.jsonl}'

uv run python -m semtrace.cli.synthetic_smoke --output-dir /tmp/semtrace-smoke
uv run pytest
CUDA_VISIBLE_DEVICES=4 uv run pytest -m gpu
uv run ruff check .
uv run mypy src/semtrace
```

Evaluation emits continuous-score AP, Accuracy at fake probability 0.5,
per-generator metrics, mAcc/mAP, residual distributions, predictions, and
Cross-Attention maps. See `docs/TRAINING.md` and
`docs/REPRODUCIBILITY.md` for checkpoints and ablations.
Rank 0 shows one tqdm progress bar per configured manifest with its domain,
position, batch ETA, and processed sample count. Append `2>&1 | tee eval.log`
to the evaluation command to retain the progress output.

Evaluate a ForenSynths/ProGAN-trained SemTrace checkpoint on Self-Synthesis:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run torchrun --standalone --nproc_per_node=4 \
  -m semtrace.cli.evaluate --config-name eval \
  protocol=self_synthesis \
  experiment=self_synthesis_eval \
  checkpoint=/path/to/semtrace_best.pt \
  normal.checkpoint=/path/to/normal_best.pt \
  probe.selected_layers_path=artifacts/probes/selected_layers.json \
  probe.semantic_anchor_path=artifacts/probes/semantic_anchor.pt
```

The detector, normal predictor, selected layers, and semantic anchor must come
from the same ForenSynths/ProGAN training run.

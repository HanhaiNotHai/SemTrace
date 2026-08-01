# Training and Ablations

The stages have a strict artifact dependency:

```text
probe_layers -> selected_layers.json + semantic_anchor.pt
             -> train_normal -> normal_best.pt
             -> train_detector -> semtrace_best.pt
             -> evaluate
```

Every run creates `outputs/<experiment>/<UTC timestamp>/` with resolved config,
environment, Git revision, selected layers, JSONL metrics, checkpoints,
TensorBoard events, and predictions.

The main detector uses Adam (`lr=2e-4`, betas `0.9/0.999`, weight decay
`2e-4`), 200 epochs, no scheduler, and global batch 128. Stage 2 defaults to 50
epochs and reports its setting separately. Runtime chooses bf16 only when
supported, otherwise fp16 with GradScaler.

Resume Stage 3 with `training.resume=/path/to/semtrace_last.pt`. Checkpoints
contain model, optimizer, scaler, epoch, step, RNG state, resolved config,
selected layers, manifest identity, and best validation AP.

Main ablations need only overrides:

```bash
model_options.use_normal_predictor=false
model_options.num_trace_scales=1
model_options.use_cross_attention=false
model_options.use_separation_loss=false
```

Diagnostic frozen-feature baselines are deliberately separate:

```bash
model_options.baseline=final_cls model_options.semantic_direct_classifier=true \
model_options.use_normal_predictor=false model_options.use_separation_loss=false

model_options.baseline=intermediate_patch_mean \
model_options.semantic_direct_classifier=true \
model_options.use_normal_predictor=false model_options.use_separation_loss=false
```

`semantic_direct_classifier` remains false and prohibited for SemTrace.


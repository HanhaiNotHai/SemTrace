# SemTrace Mechanism Analysis

## Scope and data flow

The optional `SemTrace.analyze()` path returns the frozen semantic anchor `s`, selected-layer
patch features `h`, predicted normal features `h_hat`, pre-normalization prediction errors
`h-h_hat`, post-LayerNorm `candidate_trace_residual`, adapted scale tokens `r`, fused tokens
`R`, per-head Cross-Attention weights, trace evidence `z_t`, logits, and patch-grid size. The
ordinary `forward()` return type, parameter names, and checkpoint state are unchanged.

Candidate residuals are mixed diagnostic representations: they may contain generation traces,
remaining semantics, post-processing interference, and normal-prediction error. They are never
treated as pure generation traces.

## Extraction and cache

Feature extraction writes one CPU `.pt` shard per batch. `index_rankNNN.json` supports restart;
rank 0 creates `index.json` after DDP extraction. Every cache index records SHA-256 hashes for
the detector checkpoint, resolved configuration, and manifests. A mismatched cache is rejected.
Patch tensors are read shard by shard; pooled `float32` statistics are accumulated separately.
Missing metadata is excluded rather than encoded as a category.

The default cap is 2,000 samples per `(label, generator)` group. The cache defaults to
`float16`; statistical calculations convert to `float32`. Use distinct manifest records with
`split=train` and `split=test` or `validation` for linear probes and prototype coverage.

## Definitions

- Normal errors: image means of Smooth-L1, `1-cos(h,h_hat)`, and patch L2 error.
- Candidate residual strength: patch L1/L2, channel mean-square energy, image mean/max,
  top-k mean, entropy of normalized spatial L2 mass, and near-zero sparsity.
- Scale usage: `u_l = mean_i ||r_i^l||_2`, normalized by the sum across scales.
- Effective scales: `1 / sum_l usage_l^2`, bounded by 1 and the configured scale count.
- Scale entropy: `-sum_l usage_l log(usage_l + eps)`.
- Approximate fusion contribution: `mean_i ||W_l r_i^l||_2` for the corresponding block of
  the concat-linear fusion. This and scale norms are activation diagnostics, not causal effects.
- MI: PCA followed by per-dimension kNN `mutual_info_classif`, averaged across dimensions and
  compared with label permutations over configured seeds. It is a finite-sample relative
  estimate, not exact mutual information. HSIC, linear CKA, and train-only-preprocessed linear
  probes provide complementary dependence checks.
- Attention effective patches: `1 / sum_i a_i^2`; entropy, maximum, top-k mass, Gini,
  head cosine/JS, and residual-strength correlations are descriptive diagnostics.
- Prototype coverage: intersection of a target generator's top-r assigned post-hoc prototypes
  with training top-r prototypes, divided by the target top-r count. Nearest distance, OOD ratio,
  Jaccard, co-occurrence, and combination novelty must be interpreted together. Prototypes are
  post-hoc clusters, not learned discrete forensic primitives.

## Interventions

`analyze_masking` masks individual/retained scales, random/residual/attention/center/edge Patch
sets at configured ratios, and individual attention heads. `analyze_semantic_counterfactual`
tests zero, Gaussian, batch, same/different authenticity, semantic-class, and content-environment
semantic replacements at the normal predictor, Cross-Attention query, or both, plus a real/fake
fused-trace swap. Normal-predictor controls remove semantic conditioning or neighborhoods.

All conditions reuse the single accuracy-optimal threshold selected from the complete baseline
evaluation samples. The semantic sensitivity rate is the fraction of labels flipped by semantic
replacement. The trace following rate is the fraction matching the swapped donor trace label.
Logit change, prediction flips, Acc, AP, AUROC, real/fake accuracy, mAcc/mAP, and bootstrap
intervals are reported. These are inference-time intervention diagnostics, not strict causal
proofs.

## Commands

Complete extraction, diagnostics, statistics, plots, and report:

```bash
uv run torchrun --standalone --nproc_per_node=4 \
  -m semtrace.cli.run_mechanism_suite \
  checkpoint=/path/to/semtrace_best.pt \
  normal.checkpoint=/path/to/normal_best.pt \
  probe.selected_layers_path=/path/to/selected_layers.json \
  probe.semantic_anchor_path=/path/to/semantic_anchor.pt \
  protocol=genimage_sdv14 \
  'data.test_manifests={genimage:/path/to/test.jsonl}' \
  analysis.max_samples_per_group=2000
```

Use one GPU for a quick run by replacing `torchrun ...` with `python` and setting
`analysis.max_samples_per_group=32 analysis.feature_batch_size=8
analysis.bootstrap_iterations=100 analysis.prototype_counts=[16]`.

Independent entry points use the same checkpoint/cache contract:

```bash
uv run python -m semtrace.cli.analyze_normal_predictor --config-name analysis/normal_predictor ...
uv run python -m semtrace.cli.analyze_residuals --config-name analysis/residuals ...
uv run python -m semtrace.cli.analyze_scale_usage --config-name analysis/scale_usage ...
uv run python -m semtrace.cli.analyze_representation_mi --config-name analysis/representation_mi ...
uv run python -m semtrace.cli.analyze_masking --config-name analysis/masking ...
uv run python -m semtrace.cli.analyze_semantic_counterfactual --config-name analysis/semantic_counterfactual ...
uv run python -m semtrace.cli.analyze_cross_attention --config-name analysis/cross_attention ...
uv run python -m semtrace.cli.analyze_trace_coverage --config-name analysis/trace_coverage ...
uv run python -m semtrace.cli.visualize_trace_maps --config-name analysis/visualization ...
uv run python -m semtrace.cli.run_linear_probes --config-name analysis/linear_probe ...
```

Set `analysis.cache=/path/to/feature_cache` to rerun CPU statistics without inference. The
checkpoint, manifests, and resolved configuration must still match the cache fingerprint. Live
masking/counterfactual diagnostics require extraction mode and are explicitly marked skipped
when only a cache is supplied.

## Outputs and interpretation

Runs are stored under
`outputs/mechanism/<experiment>/<checkpoint>/<dataset>/<timestamp>/` with resolved config,
environment, feature cache, CSV tables, plots, heatmaps, predictions, summary JSON, and
`mechanism_report.md`. Tables retain generator and `content_env`; additional metadata fields are
available for grouped analysis.

Fit PCA, standardization, clustering, and probes only on training samples. Do not use the test
split to select probe preprocessing, prototype count, mask ratio, or a deployment threshold.
Low probe accuracy does not prove absence of information; zero estimated MI does not prove
independence; attention does not establish causality. Heatmaps only show internal candidate
residual intensity or attention and are not automatically human-interpretable artifacts.

## Proposal mechanism package

Build the proposal-defense handoff package from the existing mechanism cache and formal
GenImage evaluation without rerunning DINOv3 or training:

```bash
MPLCONFIGDIR=/tmp/semtrace-mpl uv run python \
  -m semtrace.cli.build_proposal_mechanism_package \
  mechanism_root=outputs/mechanism/mechanism_suite/semtrace_best/genimage_sdv14/20260806T075824Z \
  eval_root=outputs/genimage_all_generators_eval/20260804T105924Z \
  checkpoint=outputs/stage3_detector/20260803T144441Z/checkpoints/semtrace_best.pt
```

The command exports three traceable results: pre-LayerNorm real/fake prediction-error
distributions, per-generator after-adapter leave-one-scale-out AP changes, and globally matched
semantic-versus-trace swaps recomputed with only the trained Cross-Attention/classification
head. It writes CSV/JSON, 300-DPI PNG/PDF figures, a triptych, Chinese descriptions, a ChatGPT
handoff, provenance hashes, a file manifest, and a validated ZIP. Missing semantic labels are
never synthesized; the semantic intervention is described only by its actual generator and
authenticity matching rule.

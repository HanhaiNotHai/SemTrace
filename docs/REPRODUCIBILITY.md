# Reproducibility

Dependencies are resolved in `uv.lock`; use `uv sync --extra dev` and do not
substitute an unpinned requirements file. The local DINOv3 snapshot is recorded
as model ID `facebook/dinov3-vitb16-pretrain-lvd1689m`, revision label `master`,
with environment and lock hashes.

Each run records Python, PyTorch, CUDA runtime, GPU names/count, Transformers,
model ID/revision, lock hash, Git commit, seed, actual global batch, and AMP
mode. Dataset manifests are deterministic JSONL files and their scanner audit
records matched, accepted, invalid, duplicate, and skipped-small counts.

Seeds cover Python, NumPy, CPU Torch, and all CUDA devices. Distributed samplers
receive the epoch. Frozen DINOv3 always remains in evaluation mode. Default
tests use `TinyBackbone` and synthetic tensors, never network access or real
weights.

Required checks:

```bash
uv lock --check
uv sync --extra dev
uv run ruff check .
uv run mypy src/semtrace
uv run pytest
uv run python -m semtrace.cli.synthetic_smoke --output-dir /tmp/semtrace-smoke
```

Full dataset training/evaluation and multi-GPU resource reservation are
intentionally not part of verification.

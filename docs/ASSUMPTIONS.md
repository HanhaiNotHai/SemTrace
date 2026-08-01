# Implementation Assumptions

This file records choices that are not established by the SemTrace method or the
referenced evaluation paper.

1. Layer identifiers are zero-based Transformer block indices. For a 12-block
   ViT-B/16, automatic selection considers blocks 0 through 10 and excludes
   block 11, which supplies the final semantic output.
2. The frozen semantic projection is initialized with orthonormal rows using
   seed 3407, saved by stage 1, and never updated by authenticity labels.
3. The local DINOv3 snapshot label is `master`; it is not treated as an upstream
   commit hash. Environment records include file hashes for traceability.
4. Dataset adapters consume a common JSONL manifest. Directory rules not proven
   by a dataset's official documentation remain explicit scanner configuration,
   not hidden filename heuristics.
5. Images smaller than 128 pixels on either side are skipped by the main
   protocol. Reflect padding is available only as an explicit non-default option.
6. The host has six RTX 5090 GPUs rather than the anticipated RTX 3090 GPUs.
   Four-GPU global-batch semantics remain the strict protocol; six-GPU runs are
   labeled non-strict and use their actual global batch.

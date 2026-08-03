# Protocol Audit

SemTrace follows the evaluation setup of *Beyond Semantic Features: Pixel-level
Mapping for Generalized AI-Generated Image Detection* without adding its pixel
mapping module to the model.

| Item | Authority | Implemented rule | Status |
|---|---|---|---|
| ForenSynths training | Paper, experiments | ProGAN; car, cat, chair, horse; matching LSUN real classes | Confirmed |
| Self-Synthesis evaluation | Paper appendix | AttGAN, BEGAN, CramerGAN, InfoMaxGAN, MMDGAN, RelGAN, S3GAN, SNGAN, STGAN | Confirmed |
| GenImage training | Paper, experiments | SDv1.4 fake images and corresponding ImageNet real images | Confirmed |
| GenImage evaluation | Paper appendix | Midjourney, SDv1.4, SDv1.5, ADM, GLIDE, Wukong, VQDM, BigGAN | Confirmed |
| UniversalFakeDetect | Paper appendix | Guided, LDM variants, GLIDE variants, DALL-E; retain variants in manifests | Confirmed |
| Geometry | Paper, implementation details | Random 128x128 training crop; center 128x128 evaluation crop; no preceding resize | Confirmed |
| Optimization | Paper, implementation details | Adam, lr 2e-4, betas 0.9/0.999, weight decay 2e-4, 200 detector epochs, batch 128 | Confirmed |
| Metrics | Paper, implementation details | Accuracy at threshold 0.5 and AP from continuous fake scores | Confirmed |
| Small images | Current implementation | Skip and count by default; optional explicit reflect padding | Assumption |
| Self-Synthesis directory mapping | Dataset installation | Nine generator directories, each with `0_real` and `1_fake` | Confirmed locally |
| Self-Synthesis sampling | Current implementation | Use every accepted file; no random sampling or content deduplication | User-confirmed |
| Directory sample counts | Generated manifest audit | 2,000 real and 2,000 fake files for each of nine Self-Synthesis generators; 36,000 accepted, zero skipped/invalid/duplicate | Verified locally |

Primary sources:

- Paper and appendix: https://arxiv.org/html/2512.17350
- DINOv3 Transformers API: https://huggingface.co/docs/transformers/model_doc/dinov3
- Official DINOv3 repository: https://github.com/facebookresearch/dinov3
- GenImage paper: https://arxiv.org/abs/2306.08571

Every manifest build records accepted, skipped, and invalid sample counts. No
unverified folder layout or sampling count is presented as part of the paper's
original protocol.

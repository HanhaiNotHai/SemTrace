# SemTrace Method

SemTrace uses one frozen DINOv3 backbone. It is not a renamed two-encoder
classifier. The final normalized CLS token and final patch mean are concatenated
and mapped by a fixed orthogonal projection to a 512-dimensional semantic
anchor.

Stage 1 fits independent linear authenticity, semantic, and nuisance probes to
the mean patch representation of blocks 0–10. For each layer,
`J = z(AP_auth) - 0.5*z(Acc_sem) - 0.5*z(Acc_nuis)`. One maximum is selected
from each depth third; block 11 remains the final semantic output. A generator
probe is used only when at least two training generators exist.

Stage 2 trains three independent neighborhood Cross-Attention predictors on
real images. A query combines the stopped semantic anchor and 2-D target
position. Keys/values contain a 3×3 neighborhood with the center removed and an
explicit boundary mask. Smooth-L1 plus cosine loss is computed after
affine-free LayerNorm. Predictor targets are detached.

Stage 3 freezes the backbone, anchor, and predictors. Per-scale
`candidate_trace_residual = LN(observed - predicted)` is processed by a local
depthwise-convolution Adapter. Three aligned grids are concatenated and
projected to 256 dimensions. Cross-Attention uses semantic `Q` and trace-only
`K/V`. Its output is mixed with mean trace evidence through a gate capped at
0.5; the query has no residual path to the classifier.

The classifier receives only trace evidence. A margin-based cross-covariance
loss discourages excessive semantic/trace linear dependence using a
differentiable global DDP batch. Candidate residuals are explicitly understood
to mix possible generator traces, residual semantics, post-processing effects,
and normal-prediction error.


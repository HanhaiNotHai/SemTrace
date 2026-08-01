# Data Layout

All loaders consume JSONL manifests; absolute dataset paths never appear in
Python source. Each record contains `path`, `label`, `semantic_class`,
`generator`, `source`, `split`, optional `degradation`, and `file_format`.
Labels are always real `0`, fake `1`.

## ForenSynths

The scanner expects the official CNNDetection layout:

```text
<root>/
  train/{car,cat,chair,horse}/{0_real,1_fake}/...
  val/{car,cat,chair,horse}/{0_real,1_fake}/...
```

Training fake images are ProGAN. Self-Synthesis and UniversalFakeDetect data use
their own manifests so generator/variant names remain explicit.

## GenImage

The scanner expects:

```text
<root>/<Generator>/{train,val}/{ai,nature}/...
```

Only `Stable Diffusion V1.4/train` enters training. Validation rules cover
Midjourney, SDv1.4, SDv1.5, ADM, GLIDE, Wukong, VQDM, and BigGAN. If the
installed archive uses different directory spelling, create an explicit
manifest rather than adding filename heuristics.

Transforms perform only random 128×128 training crop or center 128×128
evaluation crop, followed by DINO normalization. They never resize. Images
smaller than 128 are audited and skipped by default; `reflect` is an explicit
non-main-protocol alternative.


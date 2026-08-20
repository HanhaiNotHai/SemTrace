# SemTrace项目实现与实验技术报告

> 报告性质：基于当前仓库代码、配置、checkpoint 和指定输出目录形成的技术事实报告。它不是开题报告正文。审计日期为 2026-08-11，当前仓库提交为 `12cafd9f7dfce7e4f6eff29525f705e77cd8ad58`。

## 1. 项目概述

SemTrace（Semantic-Conditioned Multi-Scale Trace Residual Learning）面向 AI 生成图像检测的跨生成器泛化问题。冻结视觉基础模型同时保留内容语义、局部纹理与结构信息；若直接用高层语义分类，模型容易利用内容或数据来源捷径，但完全删除语义又忽略了“不同内容具有不同正常局部模式”这一事实。

当前实现将语义从直接真假证据重新定位为条件变量：语义用于预测真实图像在给定内容与邻域下应有的局部特征，并作为 Cross-Attention Query 选择候选痕迹；最终分类器只读取痕迹证据。核心信息流为：

```text
128×128 图像
→ 冻结 DINOv3 ViT-B/16
→ 自动选择 block 2/6/8 的 Patch 特征
→ 三个冻结的语义条件正常预测器
→ 候选痕迹残差
→ 三个独立 Trace Adapter
→ Patch 级三尺度融合
→ 语义 Query、痕迹 Key/Value 的 Cross-Attention
→ trace_evidence
→ 真假 logit
```

这里的候选痕迹残差可能同时含有生成过程异常、剩余语义、后处理干扰和正常预测误差，不能称为“纯生成痕迹”。

## 2. 当前代码与实验版本

- 当前代码版本：`12cafd9f7dfce7e4f6eff29525f705e77cd8ad58`，工作区审计前为 clean。
- 阶段一、二运行代码版本：`5db9e559b2483ba9877d4876ea8a8afac6697b1b`。
- 阶段三运行代码版本：`27375b7016e108c2fc0515049e41b0e04e726cd4`。
- 指定主评测运行代码版本：`fedcc5ab2ed08766c8e3fe8bd4a3df5a2050cff8`。
- 当前环境：Python 3.11.15、PyTorch 2.11.0+cu130、Transformers 5.14.0；项目由 `pyproject.toml` 和 `uv.lock` 锁定。
- DINOv3：`facebook/dinov3-vitb16-pretrain-lvd1689m`，revision `master`，从本地 ModelScope snapshot 由 Transformers `AutoModel.from_pretrained` 离线加载。
- 正常预测 checkpoint：`outputs/stage2_normal/20260803T113850Z/checkpoints/normal_best.pt`。
- 检测 checkpoint：`outputs/stage3_detector/20260803T144441Z/checkpoints/semtrace_best.pt`，SHA-256 为 `ae38fe62b9e7a140dffdd212182fa290a993d8a603bf5e7ce1354eddeed7129a`。
- 主结果唯一来源：`outputs/genimage_all_generators_eval/20260804T105924Z`。
- 机制结果唯一来源：`outputs/mechanism/mechanism_suite/semtrace_best/genimage_sdv14/20260806T075824Z`。

### 2.1 实现映射表

| 研究概念 | 实际代码类/函数 | 文件路径 | 真实输入 | 真实输出 |
| --- | --- | --- | --- | --- |
| DINOv3 冻结主干 | `DINOv3Backbone` | `src/semtrace/backbones/dinov3.py` | `[B,3,128,128]` | `BackboneOutput` |
| Token 切分 | `split_backbone_tokens` | `src/semtrace/backbones/base.py` | `[B,69,768]`、4 个 register token | CLS `[B,768]`、Patch `[B,64,768]` |
| 语义锚点 | `FrozenSemanticAnchor` | `src/semtrace/models/semantic_anchor.py` | 最终 CLS 与 Patch 均值 | `s:[B,512]` |
| 三层探针 | `fit_layer_probes`、`select_probe_layers` | `src/semtrace/models/probes.py` | 各 block Patch 均值及标签 | layer 2、6、8 |
| 中间 Patch 特征 | `BackboneOutput.intermediate_patch_tokens` | `src/semtrace/backbones/base.py` | 选定层 hook 输出 | `h^(l):[B,64,768]` |
| 正常特征预测器 | `NormalFeaturePredictor`、`MultiScaleNormalPredictors` | `src/semtrace/models/normal_predictor.py` | `sg(s)`、8 邻域、二维位置 | `hat_h^(l):[B,64,768]` |
| 候选痕迹残差 | `candidate_trace_residual` | `src/semtrace/models/trace_adapter.py` | `h^(l)`、`hat_h^(l)` | `e^(l):[B,64,768]` |
| 三尺度 Adapter | `TraceAdapter` | `src/semtrace/models/trace_adapter.py` | `e^(l)` 与 8×8 网格 | `r^(l):[B,64,256]` |
| 多尺度融合 | `MultiScaleTraceFusion` | `src/semtrace/models/multiscale_fusion.py` | 三个 `r^(l)` | `R:[B,64,256]` |
| Cross-Attention | `SemanticTraceCrossAttention` | `src/semtrace/models/cross_attention.py` | Q=`s`，K/V=`R` | `z_t:[B,256]`、权重 `[B,8,1,64]` |
| 完整模型及分析接口 | `SemTrace`、`SemTraceAnalysisOutput` | `src/semtrace/models/semtrace.py` | 图像及 `return_analysis` | 正常输出或完整机制张量 |
| 最终分类器 | `TraceClassifier` | `src/semtrace/models/classifier.py` | 仅 `trace_evidence:[B,256]` | `logits:[B]` |
| 正常损失 | `normal_prediction_loss` | `src/semtrace/losses/normal_prediction.py` | 预测与停止梯度目标 | SmoothL1 + cosine 标量 |
| 检测损失 | `detection_loss` | `src/semtrace/losses/detection.py` | logit、0/1 标签 | BCEWithLogits 标量 |
| 分离损失 | `semantic_trace_separation_loss` | `src/semtrace/losses/separation.py` | `sg(s)`、`z_t` | margin 协方差能量 |

## 3. 方法整体设计

### 3.1 研究问题

已知生成器上的高检测准确率不能自动转化为未知生成器上的可靠性。训练集合可能把内容类别、文件格式、真实图像来源或后处理方式与真假标签绑定，使检测器学习到不稳定相关性。SemTrace 的目标是保留冻结基础模型的稳定表示，同时把直接判别路径限制在局部候选残差及其提炼结果上。

### 3.2 核心思想

对图像内容 (c)、稳定真实性线索 (t)、生成器特定因素 (g) 和数据/后处理偏置 (d)，当前方法不假设其天然正交。语义锚点 (s) 只条件化正常预测与痕迹选择；真实图像正常预测器给出局部条件期望；实际特征与期望之差形成混合的候选残差；Adapter、真假监督、多尺度融合和分离约束共同筛选稳定真实性信息。

### 3.3 整体信息流

设三个选定层为 \(\mathcal L=\{2,6,8\}\)：

$$
x\rightarrow\{h^{(l)}\}_{l\in\mathcal L}
\rightarrow\{\hat h^{(l)}\}
\rightarrow\{e^{(l)}\}
\rightarrow\{r^{(l)}\}
\rightarrow R\rightarrow z_t\rightarrow \hat y.
$$

表1给出真实训练状态。

| 模块 | 输入 | 输出 | 可训练/冻结 | 作用 |
| --- | --- | --- | --- | --- |
| DINOv3 ViT-B/16 | 128×128 RGB | CLS、最终及选层 Patch token | 全阶段冻结 | 稳定语义与局部表示 |
| FrozenSemanticAnchor | CLS、Patch 均值 | 512 维语义锚点 | 固定正交权重、冻结 | 条件变量 |
| 三个 Normal Predictor | 语义、邻域、位置 | 3×正常 Patch 特征 | 阶段二训练，阶段三冻结 | 建模真实条件正常模式 |
| 三个 Trace Adapter | 3×候选残差 | 3×256 维 Patch token | 阶段三训练 | 局部真实性信息提炼 |
| MultiScaleTraceFusion | 三尺度 token | 256 维融合 token | 阶段三训练 | Patch 对齐的多尺度联合表示 |
| SemanticTraceCrossAttention | 语义 Q、痕迹 K/V | 256 维 trace evidence | 阶段三训练 | 语义条件化痕迹选择 |
| TraceClassifier | trace evidence | 单一 logit | 阶段三训练 | 直接真假判别 |

## 4. 方法模块

### 4.1 冻结 DINOv3

当前主干是 DINOv3 ViT-B/16：12 个 Transformer block、hidden dimension 768、12 个注意力头、MLP dimension 3072、Patch size 16、1 个 CLS token 和 4 个 register token。模型配置的预训练 image size 为 224，但当前协议实际输入为 128×128，形成 8×8=64 个 Patch；因此序列长度为 `1 + 4 + 64 = 69`。

包装器从模型配置读取 `num_register_tokens=4`，切分为：

$$
H=[h_{\mathrm{CLS}},H_{\mathrm{reg}},H_{\mathrm{patch}}],
\quad H_{\mathrm{patch}}\in\mathbb R^{B\times64\times768}.
$$

中间 block 通过 forward hook 取得，并在输出后显式应用模型最终 LayerNorm；最终 `last_hidden_state` 使用 Transformers 已归一化输出。包装器只返回选中的中间层和最终层，不返回全部 hidden states。构造时所有参数 `requires_grad_(False)`，强制 `eval()`，前向位于 `torch.no_grad()` 中。冻结可保留预训练先验，避免真假监督更新主干，并把阶段三可训练参数降至约 1.986M；checkpoint 中主干约 85.66M 参数。

### 4.2 语义锚点

真实实现先拼接最终 CLS 与最终 Patch 均值，再经过无 bias 的固定正交线性投影和无仿射 LayerNorm：

$$
s=\operatorname{LN}\!\left(W_s
[h_{\mathrm{CLS}}^{(L)};\operatorname{Mean}(H_{\mathrm{patch}}^{(L)})]\right),
\quad W_s:\mathbb R^{1536}\rightarrow\mathbb R^{512}.
$$

投影以 seed 3407 随机正交初始化，保存在 `artifacts/probes/semantic_anchor.pt` 并全程冻结。它用于三个位置：正常预测器 Query、Cross-Attention Query、分离损失中的停止梯度语义项。生产分类器没有语义输入接口。

### 4.3 阶段一：多层特征探针选择

候选层为 0-based block 0—10，最终 block 11 排除。每层使用 Patch 均值

$$u^{(l)}=\operatorname{Mean}_i h_i^{(l)}$$

训练 `StandardScaler + LogisticRegression(lbfgs)` 线性探针。真假探针以 AP 和 Balanced Accuracy 评估；语义探针在标签可用时计算准确率；干扰探针按 source、degradation、file format、generator 的优先级选择至少有两个类别的目标。实际 GenImage manifest 没有语义类别，语义覆盖率为 0，语义项因而为 0；实际干扰目标为 file format，generator probe 未启用。这一事实意味着阶段一结果不能被解释为已经抑制了语义可解码性。

层得分为：

$$J_l=z(\mathrm{AP}_{auth}^{(l)})-0.5z(\mathrm{Acc}_{sem}^{(l)})-0.5z(\mathrm{Acc}_{nuis}^{(l)}).$$

12 个 block 均分为 `[0,1,2,3]`、`[4,5,6,7]`、`[8,9,10,11]`，去除 11 后每段选最高分，平分时取较小层号。真实输出为：

| 深度段 | 选中层 | 真假 AP | 综合得分 |
| --- | ---: | ---: | ---: |
| 浅层 | 2 | 0.998470 | 0.227416 |
| 中层 | 6 | 0.999513 | 0.436479 |
| 深层 | 8 | 0.998301 | 0.388859 |

数据驱动选择使层号与当前冻结特征、数据分布和干扰标签显式绑定，同时用深度分段避免三层集中在相邻 block。原始证据位于 `artifacts/probes/probe_results.csv`、`selected_layers.json` 和 `layer_score_plot.png`。

### 4.4 阶段二：语义条件正常特征预测

对每个 (l\in\{2,6,8\})，`NormalFeaturePredictor` 接收 (h_i^{(l)}\in\mathbb R^{768})。代码以 3×3 `unfold` 提取邻域，显式删除中心，仅保留最多 8 个有效邻居并对边界做 mask。二维位置归一化到 ([-1,1]^2)。实际 Query、Key、Value 为：

$$
q_i=W_{sem}\operatorname{sg}(s)+W_{pos}p_i,\qquad
K_i=V_i=W_nH_{\mathcal N(i)}^{(l)}.
$$

每个位置作为独立的长度 1 Query，对最多 8 个邻居做两层轻量 MHA；hidden dimension 256、8 heads、dropout 0.1，每层含 Query residual 和 LayerNorm，最后线性映射回 768 维。三个尺度结构相同但参数完全独立，总计约 3.161M 可训练参数。中心 Patch 不进入预测输入，相关单元测试验证了改变中心 token 不改变预测结果。

该阶段的训练集和验证集都只保留 `label=0` 的真实图像，目标是学习“在当前语义和局部上下文下，真实图像该位置通常呈现何种冻结特征”。

### 4.5 候选痕迹残差

阶段三使用冻结预测器生成正常特征，实际残差为无仿射 LayerNorm 后的差：

$$
e_i^{(l)}=\operatorname{LN}_{\mathrm{no\ affine}}
\left(h_i^{(l)}-\hat h_i^{(l)}\right).
$$

代码名称为 `candidate_trace_residual`。机制缓存还单独保留 LayerNorm 前的 `prediction_error=h-hat_h`，以便残差幅值分析不被归一化固定范数掩盖。后续模块负责从混合残差中提炼真实性相关信息。

### 4.6 三尺度痕迹 Adapter

每层独立的 Adapter 为：

```text
LayerNorm(768) → Linear(768,256) → GELU
→ 8×8 网格上的 3×3 Depthwise Conv(groups=256)
→ Linear(256,512) → GELU → Linear(512,256)
→ 与投影分支相加 → LayerNorm(256)
```

$$r_i^{(l)}=A_l(e_i^{(l)})\in\mathbb R^{256}.$$

它不使用 CLS/register token，也不做全局自注意力。浅、中、深层承载不同抽象层级是设计动机；真实实验只支持“各尺度激活和屏蔽影响不同”，不能把某层预先命名为特定物理痕迹。

### 4.7 多尺度痕迹融合

三个实际网格均为 8×8。融合沿通道拼接，再投影：

$$
R=\operatorname{LN}\left(\operatorname{GELU}
(W_f[r^{(2)};r^{(6)};r^{(8)}])\right)
\in\mathbb R^{B\times64\times256}.
$$

若未来网格不同，代码会显式二维插值；本次实验没有发生。融合保留 Patch 空间对应关系。

### 4.8 阶段三：语义—痕迹 Cross-Attention

当前 `SemanticTraceCrossAttention` 使用 8 头 MHA：

$$Q=W_q\operatorname{sg}(s)\in\mathbb R^{B\times1\times256},\quad
K=W_kR,\quad V=W_vR.$$

$$z_{att}=\operatorname{MHA}(Q,K,V),\qquad
z_{mean}=\operatorname{Mean}_iR_i.$$

最终证据采用受限门控融合：

$$
\eta=0.5\sigma(a),\qquad
z_t=\operatorname{LN}\left((1-\eta)z_{mean}+\eta z_{att}\right).
$$

checkpoint 中 (a=0.204508)，所以 (eta\approx0.2755)。Query 不通过 residual 直接加入证据；Value 始终来自痕迹 token。这一结构实现“语义选择、痕迹供证”。注意力权重形状为 `[B,8,1,64]`，供机制分析而不等同于因果解释。

### 4.9 最终真假分类

`TraceClassifier` 是单个 `Linear(256,1)`：

$$\hat y_{logit}=W_cz_t+b_c.$$

没有额外 MLP 或 dropout。分类器只接收 `trace_evidence`；sigmoid 仅在推理、阈值和指标计算时使用。标签固定为 `real=0`、`fake=1`。

## 5. 损失函数

### 5.1 正常预测损失

预测和停止梯度目标先分别做无仿射 LayerNorm。每层损失为：

$$
\mathcal L_{normal}^{(l)}=
\operatorname{SmoothL1}(\operatorname{LN}(\hat H^{(l)}),
\operatorname{sg}(\operatorname{LN}(H^{(l)})))
+0.5\left[1-\operatorname{Mean}\cos(\operatorname{LN}(\hat H^{(l)}),
\operatorname{sg}(\operatorname{LN}(H^{(l)})))\right].
$$

三个尺度求和而非平均：

$$\mathcal L_{stage2}=\mathcal L_{normal}=\sum_{l\in\{2,6,8\}}\mathcal L_{normal}^{(l)}.$$

### 5.2 检测损失

$$
\mathcal L_{det}=\operatorname{BCEWithLogits}(\hat y_{logit},y),
\qquad y=0\text{（真实）},\ y=1\text{（生成）}.
$$

### 5.3 语义—痕迹分离约束

DDP 下先用支持自动求导的 `torch.distributed.nn.functional.all_gather` 形成全局批次。语义停止梯度；两个表示逐维中心化，并用中心化均方根标准化：

$$
\tilde S=\frac{\operatorname{sg}(S)-\mu_S}
{\sqrt{\operatorname{Mean}(S-\mu_S)^2}+\epsilon},\quad
\tilde Z=\frac{Z-\mu_Z}{\sqrt{\operatorname{Mean}(Z-\mu_Z)^2}+\epsilon}.
$$

$$
C_{SZ}=\frac{\tilde S^T\tilde Z}{B_{global}-1},\qquad
\mathcal L_{sep}=\max\left(0,
\frac{\lVert C_{SZ}\rVert_F^2}{512\times256}-0.01\right).
$$

该约束只向痕迹路径传梯度，目标是抑制过强线性语义相关性，不是证明或强制完全无语义。

### 5.4 总体训练目标

$$\mathcal L_{stage2}=\mathcal L_{normal},$$

$$\mathcal L_{stage3}=\mathcal L_{det}+0.05\mathcal L_{sep}.$$

当前主实现没有 GRL、频率分支、额外辅助头、pixel mapping 输入分支或排序损失。

## 6. 三阶段训练流程

### 6.1 阶段一：多层特征探针选择

冻结 DINOv3，提取 block 0—10 的 Patch 均值；在 CPU 上拟合 sklearn 线性探针，输出探针 CSV、选层 JSON、得分图和冻结语义投影。实际 4 GPU、每卡 batch 256、global batch 1024、workers 4。配置记录 `bf16`，但实际特征提取路径没有 autocast 包裹，不能把阶段一描述为已使用 bf16 前向。`probe.epochs=20` 实际转换成 logistic regression `max_iter=200`，`probe.learning_rate` 不参与 sklearn 优化。

### 6.2 阶段二：语义条件正常特征学习

冻结 DINOv3 与语义投影，只训练三个预测器；训练和验证 loader 均强制 real-only。实际运行 50 epochs，Adam，lr 2e-4，betas (0.9,0.999)，weight decay 2e-4，无 scheduler，bf16、梯度裁剪 1.0；4 GPU×每卡 32×累积 1=global batch 128。最佳 checkpoint 位于 epoch 47、global step 58,045，最佳验证 loss 0.372924。

### 6.3 阶段三：多尺度痕迹判别学习

加载并冻结主干、语义投影及阶段二三个预测器；训练 Adapter、融合、Cross-Attention 和分类器。实际运行 200 epochs，优化器和 batch 配置与阶段二相同，按 validation AP 保存最佳 checkpoint；最佳为 epoch 180、global step 450,180、validation AP 0.999980。训练支持 DDP、bf16/fp16 fallback、梯度累积、裁剪和 resume。

表2汇总阶段依赖。

| 阶段 | 训练模块 | 冻结模块 | 数据 | 损失 | 输出 |
| --- | --- | --- | --- | --- | --- |
| 阶段一：多层特征探针选择 | sklearn 线性探针；生成固定语义投影 | DINOv3 | GenImage train/validation 冻结特征 | 探针逻辑回归目标 | `selected_layers.json`、anchor、CSV、图 |
| 阶段二：语义条件正常特征学习 | 3 个 Normal Predictor | DINOv3、semantic anchor | real-only train/validation | `L_normal` | `normal_best.pt`、`normal_last.pt` |
| 阶段三：多尺度痕迹判别学习 | 3 Adapter、fusion、Cross-Attention、classifier | DINOv3、anchor、3 predictors | SDv1.4 train real/fake；八生成器 validation | `L_det+0.05L_sep` | `semtrace_best.pt`、`semtrace_last.pt` |

表3给出关键真实超参数。

| 参数 | 当前真实值 |
| --- | --- |
| 输入 / Patch 网格 | 128×128 / 8×8 |
| DINOv3 hidden / blocks / registers | 768 / 12 / 4 |
| selected layers | `[2,6,8]`（0-based） |
| semantic / trace dimension | 512 / 256 |
| Normal Predictor | hidden 256，8 heads，2 layers，3×3 邻域，dropout 0.1 |
| Cross-Attention | 8 heads，dropout 0.1，max gate 0.5，checkpoint gate ≈0.2755 |
| Stage 2 / Stage 3 epochs | 50 / 200 |
| optimizer / lr / weight decay | Adam / 2e-4 / 2e-4 |
| Stage 2/3 actual global batch | 4×32×1=128（严格协议） |
| AMP / gradient clip | bf16 / 1.0 |
| normal loss weights | SmoothL1 1.0，cosine 0.5 |
| separation weight / margin / eps | 0.05 / 0.01 / 1e-6 |
| seed | 3407 |

## 7. 数据与实验协议

当前 GenImage 协议按 Pixel-level Mapping 工作流实现：假图训练生成器仅 SDv1.4，真实图来自相应 ImageNet 数据；测试域为 Midjourney、SDv1.4、SDv1.5、ADM、GLIDE、Wukong、VQDM 和 BigGAN。训练随机裁剪 128×128，评测中心裁剪 128×128；裁剪前不 resize，随后仅做 RGB 转换、`/255` 和 ImageNet mean/std 归一化。短边不足 128 默认跳过，reflect padding 只能显式开启。

真实 manifest 规模如下：

- SDv1.4 train：158,142 real + 161,997 fake，共 320,139。
- 八生成器 validation：49,758 real + 50,000 fake，共 99,758。
- 阶段二训练只读取 158,142 个 train real；验证读取 validation real。
- 指标：Accuracy、AP、AUROC、FPR、FPR@95%TPR、real/fake accuracy、per-generator Acc/AP、mAcc/mAP。

必须记录的口径边界：阶段一和阶段三的 `validation_manifest` 均是包含八生成器的 `genimage_sdv14.jsonl`，阶段三还按该 validation AP 选择 epoch；指定主评测再次评估对应八生成器 validation 清单。因此当前数值是“既有 validation/model-selection 协议下的正式项目结果”，不是完全未参与模型选择的独立测试集估计。后续论文若要求严格未知生成器泛化，应另建只含训练生成器的选层/模型选择集和不可见测试集。

## 8. GenImage 主实验结果

本节只读取 `outputs/genimage_all_generators_eval/20260804T105924Z`。该运行使用上述 `semtrace_best.pt`、4 GPU、每卡 batch 64（评测 global batch 256）、bf16、中心裁剪且无 resize。

评测器在全部 99,758 个样本上选择使 Accuracy 最大的一个全局阈值；若候选准确率相同，先选最接近 0.5，再选较大阈值。实际阈值为 **0.0002611903**。因此 Acc、mAcc 和 real/fake accuracy 是测试集合校准后的结果；AP/AUROC 使用连续分数，不受该阈值影响。表4中的 per-generator AUROC 未在主结果文件保存，故不补写。

| Generator | Acc | AP |
| --- | ---: | ---: |
| ADM | 0.777991 | 0.902088 |
| BigGAN | 0.869293 | 0.955446 |
| GLIDE | 0.943289 | 0.986585 |
| Midjourney | 0.939634 | 0.984385 |
| SDv1.4 | 0.973023 | **0.999980** |
| SDv1.5 | 0.971498 | 0.999858 |
| VQDM | 0.911947 | 0.975802 |
| Wukong | **0.973675** | 0.999430 |

总体 Accuracy=0.922101、AP=0.978735、AUROC=0.975729；mAcc=0.920044、mAP=0.975447；real accuracy=0.945697、fake accuracy=0.898620、FPR=0.054303、FPR@95%TPR=0.155493。按 Acc，最佳为 Wukong、最差为 ADM；按 AP，最佳为训练生成器 SDv1.4、最差也为 ADM。结果显示当前模型在多数未知生成器上保持较高排序能力，但 ADM 和 BigGAN 仍是主要泛化薄弱点。

主结果中的 `residual_distributions` 是对 LayerNorm 后候选残差的运行时监控；其每 token 768 维 L2 范数接近 \(\sqrt{768}=27.713\)，主要验证数值稳定性，不适合当作真假幅值证据。第9节使用机制缓存中的 LayerNorm 前预测误差完成有效分布比较。

## 9. SemTrace 机制分析

本节只使用指定机制目录。分析从八生成器中按 `(label,generator)` 最多取 2,000 个样本，共 32,000（16,000 real、16,000 fake），缓存 float16，统计转 float32，bootstrap 1,000 次，随机种子 0/1/2。机制运行自身的全局阈值为 0.000168653，baseline Acc=0.904188、AP=0.969953、AUROC=0.965595。机制结果是有限样本诊断，不代替第8节主结果。

### 9.1 正常预测器

LayerNorm 前图像级 L2 预测误差在三个尺度均表现为 fake 大于 real：

| 层 | Real 均值（95% CI） | Fake 均值（95% CI） | Cohen's d | Mann–Whitney p |
| --- | ---: | ---: | ---: | ---: |
| 2 | 74.197 [74.101,74.293] | 76.385 [76.282,76.486] | 0.337 | 2.17e-244 |
| 6 | 50.487 [50.439,50.537] | 51.070 [51.016,51.120] | 0.174 | 3.09e-49 |
| 8 | 31.414 [31.388,31.439] | 32.213 [32.183,32.245] | 0.432 | < 数值下限 |

这支持真实图像条件正常模型对生成图像产生更大的 L2 不可解释偏差，且深层效应量最大。结论具有指标依赖性：cosine error 在三个尺度均为 fake 更低，SmoothL1 在 layer 2 也反向，故不能概括为“所有预测误差指标都更大”。语义置零只使最终预测翻转 0.409%，表明当前正常预测器的语义条件在最终判别上的独立影响较弱，仍需有标签反事实实验加强验证。

### 9.2 候选残差

LayerNorm 前 top-k Patch L2 误差同样在所有尺度 fake 更高：layer 2 为 90.410→93.506（d=0.267，p=2.74e-127），layer 6 为 59.802→60.626（d=0.160，p=3.00e-48），layer 8 为 35.889→36.853（d=0.359，p=3.30e-255）。这说明候选残差强度包含稳定的真假分布差异，但它仍是混合残差，不能等同于纯生成伪影。

### 9.3 表示演化与线性探针

表6汇总真假线性探针。机制 manifest 缺少 `semantic_class` 和 `content_env`，相应探针被明确跳过；训练假图只有一个 SDv1.4 类别，统一 generator 线性探针也因类别不足跳过。因此本次不能用线性探针证明语义泄漏下降或生成器身份消除。

| 表示 | 真假 Accuracy | 真假 AP | 真假 AUROC | 语义类别 | 内容环境 | 生成器 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| semantic anchor | 0.690548 | 0.801852 | 0.778920 | 缺标签，跳过 | 缺标签，跳过 | 类别不足，跳过 |
| raw layer 2 | 0.729429 | 0.891583 | 0.895490 | 跳过 | 跳过 | 跳过 |
| raw layer 6 | 0.679952 | 0.818161 | 0.806167 | 跳过 | 跳过 | 跳过 |
| raw layer 8 | 0.655940 | 0.793710 | 0.773225 | 跳过 | 跳过 | 跳过 |
| residual layer 2 | 0.723155 | 0.882231 | 0.889843 | 跳过 | 跳过 | 跳过 |
| residual layer 6 | 0.674226 | 0.811709 | 0.795141 | 跳过 | 跳过 | 跳过 |
| residual layer 8 | 0.655976 | 0.784455 | 0.758221 | 跳过 | 跳过 | 跳过 |
| adapted layer 2/6/8 AP | — | 0.807216 / 0.845311 / 0.836683 | — | 跳过 | 跳过 | 跳过 |
| fused trace mean | 0.786036 | **0.968457** | **0.966308** | 跳过 | 跳过 | 跳过 |
| trace evidence | 0.780964 | **0.953028** | **0.939185** | 跳过 | 跳过 | 跳过 |

支持的事实是：多尺度融合和最终证据的真假可分性显著高于语义锚点及任一单尺度表示。链路并非单调增强：单尺度 residual AP 略低于对应 raw，trace evidence AP 也低于 fused mean；不能表述成每一步都提高真实性可分性。

### 9.4 三尺度互补

激活强度归一化后，layer 2/6/8 使用率分别为 0.1663/0.7014/0.1323，有效尺度数

$$N_{eff}=1/\sum_l\tilde u_l^2=1.8613,$$

尺度熵为 0.8066。所有生成器均以 layer 6 为主，但 layer 2 和 8 保持非零使用。表7采用 Adapter 后屏蔽，避免预测器或 Adapter bias 干扰尺度定义。

| 干预 | Acc | Acc 下降 | AP | AP 下降 | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| 屏蔽 layer 2 | 0.861906 | 0.042281 | 0.956564 | 0.013389 | 浅层提供补充信息 |
| 屏蔽 layer 6 | 0.508500 | **0.395688** | 0.910847 | **0.059106** | 中层是主要尺度 |
| 屏蔽 layer 8 | 0.848750 | 0.055438 | 0.943090 | 0.026863 | 深层提供补充信息 |
| 仅 layer 2 | 0.500281 | 0.403906 | 0.819212 | 0.150741 | 单浅层不足 |
| 仅 layer 6 | 0.749938 | 0.154250 | 0.904111 | 0.065842 | 最强单尺度仍弱于完整模型 |
| 仅 layer 8 | 0.500875 | 0.403313 | 0.825268 | 0.144684 | 单深层不足 |

三个尺度的屏蔽均降低 AP，且任一单尺度均弱于完整模型，支持“中层主导、浅深层互补”的当前实验证据；这比简单依赖单一 block 更有利。

### 9.5 语义角色与反事实

只将 Cross-Attention 语义 Query 置零，Acc 下降 0.008563、AP 下降 0.003258、AUROC 下降 0.004051，预测翻转率 5.14%，平均绝对 logit 变化 1.036。这说明语义 Query 确实参与痕迹权重分配。高斯语义替换影响更大：Acc 下降 0.029094、AP 下降 0.015402、翻转 8.42%。

但 batch permutation 的 Acc/AP 略有上升，因此现有反事实不能证明逐样本“正确语义”始终优于替换语义；它支持的是语义路径对注意力和校准有可测影响。由于本批次的若干真实↔生成痕迹交换条件存在无合格 donor 时回退 identity 的实现语义，本报告不采用其 `trace_following_rate` 作为正向机制证据。

### 9.6 Cross-Attention

64 Patch 的最大熵为 \(\log64=4.1589\)。8 个头中，head 0、2—7 几乎完全均匀；head 1 的平均熵 4.0933、有效 Patch 数 57.28、最大权重 0.02632、Gini 0.1705、top-k mass 0.1487，是主要非均匀头。屏蔽 head 1 使 Acc/AP/AUROC 分别下降 0.01469/0.00444/0.00553，翻转率 6.23%；其他头的影响远小。

10% Patch 屏蔽中，随机屏蔽 Acc 下降 0.00324，而 top-attention 屏蔽下降 0.08844，说明注意力排序与阈值判别敏感区域有关。不过 low-attention 屏蔽下降 0.11241，且 top-attention 屏蔽的 AP 反而上升 0.00195；因此不能声称高注意力区域具有唯一因果作用。head 1 注意力与融合 token 范数的 Pearson 相关为 0.1668，只能视为弱相关诊断。

### 9.7 未知生成器连续痕迹覆盖

该分析对已学习的连续 token 做事后 PCA + MiniBatchKMeans 聚类，原型不是模型内部显式基元。以 64 原型、top-r=10 为例：

| 未知生成器 | Top-r coverage | 最近原型距离 | OOD token 比例 | 组合新颖度 |
| --- | ---: | ---: | ---: | ---: |
| ADM | 0.40 | 0.4787 | 0.0527 | 0.99975 |
| BigGAN | 0.30 | 0.4804 | 0.0533 | 1.00000 |
| GLIDE | 0.60 | 0.4689 | 0.0460 | 0.99975 |
| Midjourney | 0.60 | 0.4622 | 0.0425 | 0.99950 |
| SDv1.5 | 0.90 | 0.4785 | 0.0586 | 1.00000 |
| VQDM | 0.60 | 0.4703 | 0.0479 | 0.99975 |
| Wukong | 0.80 | 0.4755 | 0.0558 | 0.99950 |

OOD token 比例整体约 4.25%—5.86%，但 top-r coverage 从 0.30 到 0.90 且随原型数量定义明显变化。较低 OOD 比例、高组合新颖度与“未知生成器可能复用已学连续痕迹模式并形成不同组合”的假设一致；不能将此提升为已证明的内部离散基元机制。

### 9.8 单样本机制可视化

- 样本路径：`/data/zhy/GenImage/stable_diffusion_v_1_4/train/ai/658_sdv4_00138.png`
- 真实标签：生成图像（1）
- 来源：Stable Diffusion v1.4
- 模型输出：fake probability=0.000218，预测为真实图像（错误）
输出目录：`outputs/mechanism/project_report_single_sample/658_sdv4_00138/heatmaps`

可视化专用预处理不再裁剪：将完整 512×512 原图使用 Pillow LANCZOS resize
到 128×128，再应用与模型一致的归一化。完整画面包含猫的头部、躯干、四肢、尾部及棕色背景。
本次前向得到 logit=-8.428906、fake probability=0.000218413，在 0.5 阈值和主实验
全局阈值 0.000261190 下均判为真实。旧中心裁剪图标题中的概率为 1.000；因此该样本直接
暴露出视野和目标尺度变化引起的预测不稳定性，不能再作为“整图正确检测”的示例。

视觉检查显示：

- layer 2 候选残差较细碎，较高响应主要分布在上方及右侧背景、猫胸腹部局部和若干目标边界；头部中心和尾部附近存在低响应区域。
- layer 6 响应更平滑成片，上方、右上背景和下方地面响应较高，猫头、上躯干与尾部相对较低，表现出较明显的目标—背景尺度差异。
- layer 8 同样在上方和右侧背景形成较高响应，猫身体与尾部大部分区域较低；其空间分布比 layer 2 更粗粒度。
- Adapter 后三尺度仍不相同：layer 2 在面部若干点、目标边界及右下背景形成局部峰值；layer 6 更偏向下方地面和足部周围；layer 8 较强响应集中在左侧边界及少量尾端/图像边缘位置。
- 融合痕迹在上、下背景和边界处较强，穿过猫躯干与尾部的中部横带较低，说明融合结果并非任一单尺度图的直接复制。
- attention head 1 是唯一明显非均匀的头，整体偏重图像上半部的背景、耳部附近区域；其余头接近均匀，因此平均注意力也表现为平滑的上高下低分布。
- 完整画面把猫主体压缩到较少的 8×8 Patch，且高响应大量落在背景或边界。它与预测翻转共同提示：当前模型对 128×128 训练裁剪所形成的目标尺度较敏感，整图 resize 并非与主评测裁剪协议等价的输入。

这些热图仅表示模型内部候选痕迹强度或注意力分布，不自动等同于人类可解释的生成伪影，也不构成因果解释。

## 10. 已有消融和诊断实验

当前代码通过配置支持 frozen final feature、intermediate mean、direct multiscale、单尺度/三尺度、关闭 normal predictor、关闭 Cross-Attention 和关闭分离损失等基线；指定机制目录实际包含尺度、Patch、注意力头屏蔽，语义反事实，正常预测干预，注意力稳定性，MI/HSIC/CKA，线性探针和原型覆盖。表5汇总本报告采用的核心证据。

| 分析 | 指标 | 结果 | 支持的机制 |
| --- | --- | --- | --- |
| 正常预测 | fake-real L2 / Cohen's d | 三层均为正；d=0.174—0.432 | 真实条件正常模型产生真假幅值差异 |
| top-k 候选残差 | fake-real / p | 三层 fake 更高，p≤3.0e-48 | 候选残差含真实性相关统计差异 |
| 真假线性探针 | AP | semantic 0.802；fused 0.968；evidence 0.953 | 多尺度痕迹联合表示增强真假可分性 |
| 尺度使用 | 使用率 / Neff | 0.166/0.701/0.132；Neff=1.861 | 多尺度参与，中层主导 |
| 尺度屏蔽 | AP 下降 | 0.013/0.059/0.027 | 三尺度均提供有效信息 |
| 语义 Query 置零 | flip / AP 下降 | 5.14% / 0.00326 | 语义条件化影响注意力判别路径 |
| head 1 屏蔽 | flip / AP 下降 | 6.23% / 0.00444 | 非均匀注意力头有可测作用 |
| 64 原型覆盖 | OOD ratio | 4.25%—5.86% | 与连续模式复用假设一致 |

尚未纳入开题材料主体的补充诊断包括：MI/HSIC/CKA 全表、不同 JPEG/模糊/缩放的注意力稳定性、所有 Patch 比例与随机种子、128/256 原型敏感性、失败样本列表。

## 11. 当前已有科研基础

### 11.1 方法基础

已形成冻结 DINOv3、固定语义锚点、三层自动探针、三个真实图像正常预测器、候选痕迹残差、三个空间 Adapter、多尺度融合、受限语义—痕迹 Cross-Attention、trace-only classifier 和全局 DDP 分离约束的完整方法闭环。模型具备默认不改变训练路径的 `return_analysis=True` 中间表示接口。

### 11.2 工程基础

仓库采用 uv、Python 3.11、src layout、Hydra/OmegaConf 配置；支持 DDP、NCCL、bf16/fp16、梯度累积、checkpoint/resume、manifest 哈希、rank-safe 输出、tqdm、TensorBoard、统一 evaluator 和合成数据测试。机制工具支持分片缓存、断点复用、CPU 离线统计、批量 CLI 与自动 Markdown/JSON 报告。当前纳入 Git 的 Python/YAML/Markdown 文件合计约 12,969 行。

### 11.3 数据基础

代码适配器已实现 GenImage、ForenSynths、Self-Synthesis 和 UniversalFakeDetect。真实 manifest 已验证存在：GenImage（训练与八生成器验证）、ForenSynths ProGAN 四类、Self-Synthesis 九生成器；UniversalFakeDetect 本次只确认适配器代码，未发现可核验的实际 manifest 或正式输出，故不写为已完成实验。

### 11.4 实验与机制基础

已完成 SDv1.4 训练及 99,758 张八生成器评测，得到 mAcc 0.920044、mAP 0.975447；已形成 32,000 样本的机制缓存和正常预测、残差、尺度、表示探针、屏蔽、语义反事实、注意力与连续模式覆盖分析。本报告另外验证了指定 SDv1.4 样本的单图机制可视化链路。

## 12. 当前科研条件与后续实验可行性

硬件条件最多为 6×NVIDIA RTX 5090 24GB。冻结 DINOv3 后，阶段二只训练约 3.161M 参数，阶段三只训练约 1.986M 参数；4 GPU×32 可严格实现 global batch 128，6 卡可做非严格吞吐实验。DDP、NCCL 和 bf16 已有运行配置，机制分析可先多卡提取 float16 分片，再用 CPU 做统计，现有资源足以支撑后续消融、独立测试集复核和多随机种子实验。

软件条件包括 Linux、uv、Python 3.11、PyTorch 2.11.0/cu130、torchvision 0.26.0、Transformers 5.14.0、Hydra/OmegaConf、scikit-learn、SciPy、pandas、matplotlib、TensorBoard、Pillow、NumPy、safetensors、tqdm 和 NCCL。后续仍需确保 CUDA 驱动与 cu130 wheel 匹配。

数据方面已具备 GenImage、ForenSynths 和 Self-Synthesis 索引；进一步研究需要为机制样本补充可靠 `semantic_class`、`content_env`、`real_source` 和 `degradation`，并准备不参与选层、checkpoint 选择和阈值校准的严格独立测试划分。

## 13. 可用于开题报告的关键结论

1. 当前 SemTrace 已实现“语义辅助条件化、痕迹作为直接判别证据”的完整信息流；语义不直接进入生产分类器。
2. 三个真实图像正常预测器显式排除目标中心 Patch，在三层冻结特征上建立条件正常模式；生成图像的 LayerNorm 前 L2 预测误差在三层均显著更高。
3. 多尺度融合的真假线性探针 AP 为 0.9685，高于语义锚点 0.8019 和任一单尺度表示；其优势来自多尺度联合，而非每一步单调增强。
4. 中层 layer 6 是主要尺度，但屏蔽 layer 2、6、8 均使 AP 下降，完整模型优于所有单尺度条件，支持三尺度互补。
5. 指定项目主结果在八个 GenImage 生成器上达到 mAcc 0.9200、mAP 0.9754；ADM 是当前最明显的泛化薄弱点。
6. 语义 Query 置零、非均匀 attention head 屏蔽均引起可测性能下降和预测翻转，证明语义条件路径和注意力路径实际参与前向；这仍是干预诊断而非严格因果证明。
7. 当前机制数据缺少语义/内容环境标签，不能声称已经用探针证明语义泄漏下降；独立测试划分和标签补全是最重要的后续严谨性工作。

## 14. 可用于开题报告的表格与图片索引

| 图片/材料路径 | 内容说明 | 适合章节 | 推荐图题 |
| --- | --- | --- | --- |
| `artifacts/probes/layer_score_plot.png` | block 0—10 综合探针得分与选中层 | 技术路线/自动选层 | “DINOv3 多层探针得分与三尺度层选择” |
| `outputs/mechanism/mechanism_suite/semtrace_best/genimage_sdv14/20260806T075824Z/plots/normal_prediction_real_fake.png` | 三层 real/fake 正常预测误差分布 | 方法可行性/机制 | “三尺度语义条件正常预测误差分布” |
| `outputs/mechanism/mechanism_suite/semtrace_best/genimage_sdv14/20260806T075824Z/plots/residual_scale_correlation.png` | 三尺度候选残差相关矩阵 | 多尺度机制 | “候选痕迹残差的跨尺度相关性” |
| `.../tables/linear_probes_summary.csv` | 表示阶段真假可分性；仓库未生成独立折线图 | 实验方案/机制 | “SemTrace 表示演化的线性探针结果” |
| `.../tables/masking.csv` | 尺度、Patch、attention head 屏蔽；仓库未生成独立图 | 消融实验 | “多尺度与局部区域屏蔽诊断” |
| `.../tables/semantic_counterfactual.csv` | 语义置零、随机和交换反事实；仓库未生成独立图 | 语义角色 | “语义条件路径的推理期反事实诊断” |
| `.../tables/attention_statistics.csv` | 多头熵、有效 Patch、相关性 | Cross-Attention | “SemTrace Cross-Attention 头统计” |
| `outputs/mechanism/project_report_single_sample/658_sdv4_00138/heatmaps/sample_00000_residual_L2.png` | 完整猫图 resize 后的浅层候选残差 | 失败样本/稳健性诊断 | “整图输入的浅层候选痕迹响应（layer 2）” |
| `outputs/mechanism/project_report_single_sample/658_sdv4_00138/heatmaps/sample_00000_residual_L6.png` | 完整猫图 resize 后的中层候选残差 | 失败样本/稳健性诊断 | “整图输入的中层候选痕迹响应（layer 6）” |
| `outputs/mechanism/project_report_single_sample/658_sdv4_00138/heatmaps/sample_00000_residual_L8.png` | 完整猫图 resize 后的深层候选残差 | 失败样本/稳健性诊断 | “整图输入的深层候选痕迹响应（layer 8）” |
| `outputs/mechanism/project_report_single_sample/658_sdv4_00138/heatmaps/sample_00000_fused_trace.png` | 完整画面的融合痕迹强度；该样本被误判 | 失败样本/稳健性诊断 | “整图 resize 下的多尺度融合候选痕迹响应” |
| `outputs/mechanism/project_report_single_sample/658_sdv4_00138/heatmaps/sample_00000_attention_head1.png` | 完整画面上唯一明显非均匀的 attention head | 语义条件选择/失败分析 | “整图输入的语义—痕迹 Cross-Attention 头 1” |
| `outputs/mechanism/project_report_single_sample/658_sdv4_00138/heatmaps/sample_00000_attention_times_trace.png` | 完整画面的注意力与融合痕迹强度联合图 | 失败样本/稳健性诊断 | “整图输入的注意力加权候选痕迹响应” |

仓库当前没有独立的整体方法框架图、线性探针阶段变化图、scale masking 图或 semantic counterfactual 图；上述 CSV 可供后续开题材料按统一视觉规范重绘，但本报告不虚构不存在的图片。

## 15. 面向开题报告的材料映射建议

### A. 主要研究内容

使用第1、3、4节：问题是未知生成器与内容捷径；核心是语义作为正常建模和痕迹选择条件；方法由冻结主干、自动选层、正常预测、残差提炼、多尺度融合和 Cross-Attention 构成；损失为正常、检测和分离三类。

### B. 研究方法

使用第4、5节的准确公式和表1：突出中心 Patch 排除、参数独立的三尺度预测器、trace-only classifier、受限 gate 和全局批次分离约束。

### C. 技术路线

采用：`输入 → 冻结 DINOv3 → 探针选层 → 真实图像正常特征学习 → 候选痕迹残差 → 三尺度 Adapter → 融合 → 语义条件痕迹选择 → 真假分类 → 跨生成器评测`。阶段一产物是阶段二/三的强制依赖，而非在线动态选层。

### D. 实验方案

使用第7—10节：SDv1.4 train、八生成器 validation、Acc/AP/AUROC、线性探针、三尺度屏蔽、语义反事实、注意力头/Patch 屏蔽、连续原型覆盖和热图。论文版应新增严格独立测试划分、固定阈值或独立校准集结果及缺失元数据探针。

### E. 可行性分析

使用第11—12节：方法和三阶段 checkpoint 已完成，DDP/bf16/manifest/evaluator/机制缓存均可运行；冻结主干使可训练参数小于 3.2M，6×5090 足以继续多随机种子、消融和严格协议复核。

### F. 已有科研基础

引用第8节指定主结果、第9节 32,000 样本机制证据以及单样本热图。对外表述时同时保留阈值校准和 validation 复用边界，不把当前结果描述为完全独立测试集结论。

## 16. 尚未完成但不影响当前项目报告主体的工作

- 未重新训练模型，也未执行新的全量机制分析；本报告只消费既有 checkpoint 和输出。
- 当前主实验未保存 per-generator AUROC，故表4不提供该列。
- 机制样本缺少 semantic class、content environment、real source 和 degradation 标签，对应依赖/泄漏结论尚不能验证。
- 当前八生成器 validation 同时参与选层、checkpoint 选择和正式评测；需要严格独立测试集复核跨生成器泛化。
- 当前 Accuracy 使用评测集合全局最优阈值；需要补充固定 0.5 或独立校准集阈值结果。
- 现有语义反事实表明语义路径有影响，但 batch permutation 未造成性能下降，尚不足以证明逐样本正确语义条件最优。
- 痕迹交换部分存在无可用 donor 时 identity fallback，未纳入本报告的正向证据。
- UniversalFakeDetect 只确认代码适配器，未核验真实 manifest 或正式实验。
- 热图、注意力和事后聚类只提供内部响应与相关性诊断，不能推出纯痕迹或严格因果结论。

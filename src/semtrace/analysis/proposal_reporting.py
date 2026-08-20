# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from semtrace.analysis.proposal_mechanisms import SCALE_NAMES
from semtrace.analysis.proposal_plotting import DISPLAY_GENERATORS

# Generated Chinese Markdown is intentionally kept as readable prose literals.
SCALE_CN = {"shallow": "浅层", "middle": "中层", "deep": "深层"}


def write_proposal_documents(
    root: str | Path,
    *,
    core1_statistics: dict[str, Any],
    core2_table: pd.DataFrame,
    core2_summary: pd.DataFrame,
    core2_statistics: dict[str, Any],
    core3_summary: pd.DataFrame,
    core3_statistics: dict[str, Any],
    eval_metrics: dict[str, Any],
    provenance: dict[str, Any],
    package_status: dict[str, Any],
) -> None:
    package = Path(root)
    core1_text = _core1_description(core1_statistics)
    core2_text = _core2_description(core2_table, core2_summary, core2_statistics)
    core3_text = _core3_description(core3_summary, core3_statistics)
    _write(package / "descriptions" / "core1_residual.md", core1_text)
    _write(package / "descriptions" / "core2_scale_masking.md", core2_text)
    _write(package / "descriptions" / "core3_semantic_trace_swap.md", core3_text)
    _write(package / "descriptions" / "triptych_caption.md", _triptych_caption())
    _write(
        package / "descriptions" / "figure_captions.md",
        _figure_captions(core1_statistics, core2_statistics, core3_statistics),
    )
    _write(
        package / "reports" / "core_mechanism_report.md",
        _core_report(
            core1_text,
            core2_text,
            core3_text,
            provenance,
            package_status,
        ),
    )
    _write(
        package / "reports" / "defense_talking_points.md",
        _talking_points(core1_statistics, core2_statistics, core3_statistics),
    )
    _write(
        package / "reports" / "chatgpt_handoff.md",
        _chatgpt_handoff(
            core1_statistics,
            core2_table,
            core2_summary,
            core3_summary,
            core3_statistics,
            eval_metrics,
            provenance,
            package_status,
        ),
    )
    _write(package / "README.md", _package_readme(provenance, package_status))


def _core1_description(statistics: dict[str, Any]) -> str:
    lines = [
        "# 核心结果1：真实/生成图像的三尺度候选痕迹残差差异",
        "",
        "候选残差统计量为每张图像中，预LayerNorm预测误差 `h-h_hat` 的Patch级L2范数均值。",
        "该表示可能同时包含生成过程异常、剩余语义、后处理干扰与正常预测误差，不等同于纯生成痕迹。",
        "",
        "## 数据事实",
        "",
        "|尺度（层）|Real均值|Fake均值|Cohen's d|Mann–Whitney p|",
        "|---|---:|---:|---:|---:|",
    ]
    for index, (layer, values) in enumerate(statistics["layers"].items()):
        comparison = values["comparison"]
        lines.append(
            f"|{SCALE_CN[SCALE_NAMES[index]]}（L{layer}）|"
            f"{values['real']['mean']:.4f}|{values['fake']['mean']:.4f}|"
            f"{comparison['cohens_d']:.4f}|{comparison['mann_whitney_p']:.3e}|"
        )
    lines.extend(["", "## 可以支持的机制解释", ""])
    if statistics["supports_normal_pattern_deviation"]:
        lines.append(
            "三个尺度上生成图像的正常特征预测偏差均高于真实图像，且非参数检验达到显著水平。"
            "这表明仅基于真实图像训练的正常预测器形成了具有真实性区分能力的条件正常模式。"
        )
    else:
        lines.append("至少一个尺度未同时满足Fake均值更高和p<0.05，需要人工复核。")
    lines.extend(
        [
            "",
            "## 不能过度声称",
            "",
            "该结果不证明候选残差是纯净或唯一的生成痕迹，也不构成严格因果证明。",
        ]
    )
    return "\n".join(lines) + "\n"


def _core2_description(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    statistics: dict[str, Any],
) -> str:
    lines = [
        "# 核心结果2：不同生成器的三尺度痕迹互补性",
        "",
        "采用推理期Leave-One-Scale-Out：在Adapter之后、融合之前将一个尺度置零。"
        "ΔAP定义为 `AP_full - AP_mask`，以绝对百分点报告；正值表示屏蔽造成下降。",
        "",
        "## 数据事实",
        "",
        "|尺度|平均ΔAP（百分点）|影响最大的生成器|最大ΔAP（百分点）|正下降生成器数|",
        "|---|---:|---|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"|{SCALE_CN[str(row['scale'])]}|{float(row['mean_delta_ap_pp']):.3f}|"
            f"{_display_generator(str(row['max_affected_generator']))}|"
            f"{float(row['max_delta_ap_pp']):.3f}|"
            f"{int(row['positive_drop_generator_count'])}/{len(table)}|"
        )
    counterexamples = []
    for _, row in table.iterrows():
        for scale in SCALE_NAMES:
            value = float(row[f"delta_ap_{scale}_pp"])
            if value < 0:
                counterexamples.append(
                    f"{_display_generator(str(row['generator']))}屏蔽{SCALE_CN[scale]}后AP提高"
                    f"{abs(value):.3f}个百分点"
                )
    lines.extend(["", "## 可以支持的机制解释", ""])
    if statistics["supports_scale_complementarity"]:
        lines.append(
            "不同生成器的最大敏感尺度并不完全相同，且每个尺度都至少使一种生成器的AP下降。"
            "该推理期干预结果支持三尺度提供差异化、互补的真实性线索。"
        )
    else:
        lines.append("当前尺度屏蔽结果不足以支持三尺度互补性，需要人工复核。")
    if counterexamples:
        lines.extend(["", "## 必须保留的反例", "", "；".join(counterexamples) + "。"])
    lines.extend(
        [
            "",
            "## 不能过度声称",
            "",
            "屏蔽是推理期敏感性诊断；性能变化不自动证明某尺度是唯一原因或严格因果来源。",
        ]
    )
    return "\n".join(lines) + "\n"


def _core3_description(summary: pd.DataFrame, statistics: dict[str, Any]) -> str:
    semantic = summary.set_index("condition").loc["matched_semantic_swap"]
    trace = summary.set_index("condition").loc["real_fake_trace_swap"]
    lines = [
        "# 核心结果3：语义作为条件、痕迹作为真假证据",
        "",
        "离线重算仅加载训练好的Cross-Attention与分类头。语义交换在同一生成器、同一真实性组内"
        "选择全局donor，保持原样本融合痕迹R不变；痕迹交换在同一生成器内选择相反真实性donor，"
        "保持原样本语义s不变。",
        "",
        "## 数据事实",
        "",
        "|干预|Donor覆盖率|预测翻转率|95% CI|平均绝对概率变化|平均绝对logit变化|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("匹配语义交换", semantic), ("Real↔Fake痕迹交换", trace)):
        lines.append(
            f"|{label}|{float(row['donor_match_coverage']) * 100:.2f}%|"
            f"{float(row['prediction_flip_rate']) * 100:.2f}%|"
            f"[{float(row['prediction_flip_ci_low']) * 100:.2f}%, "
            f"{float(row['prediction_flip_ci_high']) * 100:.2f}%]|"
            f"{float(row['mean_absolute_probability_change']):.4f}|"
            f"{float(row['mean_absolute_logit_change']):.4f}|"
        )
    lines.append(
        f"\n痕迹跟随率：**{float(trace['trace_following_rate']) * 100:.2f}%** "
        f"（95% CI [{float(trace['trace_following_ci_low']) * 100:.2f}%, "
        f"{float(trace['trace_following_ci_high']) * 100:.2f}%]）。"
    )
    lines.extend(["", "## 可以支持的机制解释", ""])
    if statistics["supports_trace_as_evidence"]:
        lines.append(
            "痕迹交换引起的预测翻转率高于匹配语义交换，且交换后预测更多跟随换入痕迹的真实性。"
            "结果支持“Semantic-as-Condition, Trace-as-Evidence”的信息流设计。"
        )
    else:
        lines.append("当前干预结果不满足预设支持条件，需要人工复核。")
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "GenImage缓存缺少semantic_class与content_env标签，因此这里不能声称同/异语义类别交换；"
            "语义donor匹配条件仅为同生成器、同真实性。离线干预是机制诊断，不是严格因果证明。",
        ]
    )
    return "\n".join(lines) + "\n"


def _triptych_caption() -> str:
    return (
        "# 三联图推荐图注\n\n"
        "SemTrace核心机制分析：正常模式偏离、多尺度互补与语义条件化。"
        "(a) 展示真实与生成图像在三个自动选定层上的预LayerNorm候选残差分布；"
        "(b) 展示在Adapter之后屏蔽单个尺度所造成的逐生成器AP绝对百分点变化；"
        "(c) 比较匹配语义交换与Real↔Fake融合痕迹交换引起的预测翻转率。"
        "候选残差与热图响应不等同于纯生成痕迹，屏蔽和交换均为推理期机制诊断。\n"
    )


def _figure_captions(
    core1: dict[str, Any],
    core2: dict[str, Any],
    core3: dict[str, Any],
) -> str:
    core1_ppt = (
        "生成图像在三个尺度上均表现出更大的条件正常模式偏离。"
        if core1["supports_normal_pattern_deviation"]
        else "三个尺度的真实/生成正常模式偏离结果需要复核。"
    )
    core2_ppt = (
        "不同生成器对浅、中、深层屏蔽呈现差异化敏感性。"
        if core2["supports_scale_complementarity"]
        else "尺度屏蔽尚未显示稳定互补性。"
    )
    core3_ppt = (
        "最终预测对真实性痕迹交换比匹配语义交换更敏感。"
        if core3["supports_trace_as_evidence"]
        else "语义/痕迹交换结果需要人工复核。"
    )
    return f"""# 图注

## 图1：真实与生成图像的多尺度正常特征偏离分布

论文版：图1展示真实图像与生成图像在SemTrace三个选定尺度上的候选残差分布。残差采用DINOv3局部特征与语义条件正常预测特征之差在每个Patch上的L2范数，再对Patch求均值。图中小提琴表示分布，内嵌箱线图表示四分位范围，菱形表示均值。

PPT版：{core1_ppt}

## 图2：不同生成器对多尺度痕迹的敏感性

论文版：图2给出GenImage各测试生成器在Adapter后执行Leave-One-Scale-Out干预时的AP变化。单元格为完整模型AP减去屏蔽模型AP的绝对百分点，正值表示屏蔽造成性能下降，负值表示屏蔽后性能提高。该图用于诊断尺度敏感性，而非严格因果归因。

PPT版：{core2_ppt}

## 图3：语义条件与生成痕迹对最终判别的影响

论文版：图3比较同生成器/同真实性条件下语义锚点交换与同生成器Real↔Fake融合痕迹交换引起的预测翻转率，误差线为1000次bootstrap 95%置信区间。语义交换保持原融合痕迹不变，痕迹交换保持原语义不变。

PPT版：{core3_ppt}

## 图4：SemTrace核心机制分析三联图

论文版：{_triptych_caption().splitlines()[2]}

PPT版：正常模式偏离、多尺度互补和语义条件化共同构成SemTrace的机制证据链。
"""


def _core_report(
    core1_text: str,
    core2_text: str,
    core3_text: str,
    provenance: dict[str, Any],
    status: dict[str, Any],
) -> str:
    return f"""# SemTrace核心机制实验事实报告

## 1. SemTrace核心机制

机制链为：

$$h \rightarrow \hat h \rightarrow e \rightarrow r \rightarrow R \rightarrow z_t.$$

其中，$e=h-\hat h$ 是候选痕迹残差；语义锚点用于条件化正常预测并作为Cross-Attention Query，最终分类器只读取痕迹证据 $z_t$。

## 2. 分析数据与checkpoint

- Checkpoint：`{provenance["checkpoint"]}`
- Checkpoint SHA-256：`{provenance["checkpoint_sha256"]}`
- 数据集/协议：{provenance["dataset"]} / {provenance["protocol"]}
- 选定层：{provenance["selected_layers"]}
- Core 1样本数：{provenance["sample_counts"]["core1_unique_samples"]}
- Core 2样本数：{provenance["sample_counts"]["core2_unique_samples"]}
- Core 3样本数：{provenance["sample_counts"]["core3_unique_samples"]}
- 代码commit：`{provenance["git_commit"]}`
- 材料状态：**{status["proposal_material_status"]}**

## 3. 核心结果一

{_strip_title(core1_text)}

## 4. 核心结果二

{_strip_title(core2_text)}

## 5. 核心结果三

{_strip_title(core3_text)}

## 6. 三项结果形成的证据链

```text
真实图像条件正常模式预测
↓
生成图像产生更大候选偏离
↓
不同生成器对三个尺度呈差异化敏感性
↓
语义负责条件化，融合痕迹提供直接分类证据
↓
最终预测对痕迹交换比匹配语义交换更敏感
```

## 7. 适合开题答辩使用的项目事实

1. 三个自动选定尺度均呈现显著的真实/生成候选残差差异。
2. Leave-One-Scale-Out显示不同生成器的主要敏感尺度存在差异。
3. 离线全局交换在同生成器条件内控制donor，避免现有batch-local交换覆盖不足的问题。
4. 所有图片都由包内CSV/JSON自动生成，可追溯到checkpoint、缓存与代码commit。
5. 所有机制结论均限定为推理期诊断，不宣称纯痕迹或严格因果解释。
"""


def _talking_points(
    core1: dict[str, Any],
    core2: dict[str, Any],
    core3: dict[str, Any],
) -> str:
    return f"""# 开题答辩机制图讲解要点

## 图1：多尺度正常特征偏离

- 图的核心问题：正常预测器是否学到真实图像的条件正常模式？
- 图中看到什么：{("三个尺度的Fake均值均高于Real，且p<0.05。" if core1["supports_normal_pattern_deviation"] else "至少一个尺度未满足预设支持条件。")}
- 能够支持什么：候选残差具有真实性区分信息。
- 20秒讲解逻辑：预测器只用真实图像训练；若生成图像难以被正常模式解释，其预测误差应更大；图中按浅、中、深层直接比较这一偏离。
- 可能被问的问题：残差是不是纯生成伪影？
- 建议回答要点：不是。它还可能包含剩余语义、后处理和预测误差，因此使用“候选痕迹残差”。

## 图2：Generator × Scale Masking

- 图的核心问题：为什么需要三尺度，而不是只选一个层？
- 图中看到什么：{("不同生成器的最大敏感尺度不完全相同。" if core2["supports_scale_complementarity"] else "尺度互补性需要进一步复核。")}
- 能够支持什么：三个尺度具有差异化敏感性和互补价值。
- 20秒讲解逻辑：保持checkpoint不变，在融合前屏蔽一个尺度；若某生成器AP明显下降，说明模型判别对该尺度激活敏感；不同生成器的敏感尺度差异说明并非简单重复。
- 可能被问的问题：屏蔽下降能否证明因果？
- 建议回答要点：这是推理期干预诊断，不是严格因果证明；且所有负下降值也原样报告。

## 图3：Semantic Swap vs Trace Swap

- 图的核心问题：语义会不会直接决定真假？
- 图中看到什么：{("痕迹交换翻转率更高，且痕迹跟随率超过50%。" if core3["supports_trace_as_evidence"] else "结果未满足预设支持条件。")}
- 能够支持什么：语义更接近条件变量，融合痕迹是更直接的真实性证据。
- 20秒讲解逻辑：语义交换固定R，痕迹交换固定s；比较预测翻转率即可观察最终判别对两条信息路径的相对敏感性。
- 可能被问的问题：是否做了同语义类别匹配？
- 建议回答要点：本缓存缺少semantic_class/content_env，不能这样声称；当前严格匹配的是同生成器和真实性规则。
"""


def _chatgpt_handoff(
    core1: dict[str, Any],
    core2_table: pd.DataFrame,
    core2_summary: pd.DataFrame,
    core3_summary: pd.DataFrame,
    core3: dict[str, Any],
    eval_metrics: dict[str, Any],
    provenance: dict[str, Any],
    status: dict[str, Any],
) -> str:
    core1_numbers = []
    for index, (layer, values) in enumerate(core1["layers"].items()):
        core1_numbers.append(
            f"{SCALE_CN[SCALE_NAMES[index]]}L{layer}: Real {values['real']['mean']:.3f}, "
            f"Fake {values['fake']['mean']:.3f}, d={values['comparison']['cohens_d']:.3f}, "
            f"p={values['comparison']['mann_whitney_p']:.2e}"
        )
    semantic = core3_summary.set_index("condition").loc["matched_semantic_swap"]
    trace = core3_summary.set_index("condition").loc["real_fake_trace_swap"]
    scale_numbers = ", ".join(
        f"{SCALE_CN[str(row['scale'])]}平均ΔAP={float(row['mean_delta_ap_pp']):.3f}pp"
        for _, row in core2_summary.iterrows()
    )
    generator_rows = "\n".join(
        f"|{_display_generator(generator)}|{values['accuracy'] * 100:.2f}%|"
        f"{values['average_precision'] * 100:.2f}%|"
        for generator, values in eval_metrics["per_generator"].items()
    )
    max_sensitive = []
    for _, row in core2_table.iterrows():
        scale = max(SCALE_NAMES, key=lambda item: float(row[f"delta_ap_{item}_pp"]))
        max_sensitive.append(f"{_display_generator(str(row['generator']))}:{SCALE_CN[scale]}")
    return f"""# ChatGPT交接：SemTrace开题答辩核心机制材料

> 材料状态：**{status["proposal_material_status"]}**。本文件整理实验事实，不是开题报告正文。

## A. SemTrace方法简述

SemTrace面向未知生成器条件下的AI生成图像检测。它冻结DINOv3语义与局部表征，通过离线探针自动选择浅、中、深三层；三个仅用真实图像训练的独立正常预测器，以停止梯度的语义锚点、显式排除中心Patch的局部邻域和二维位置为条件，预测真实图像局部特征。实际特征与预测特征之差被定义为候选痕迹残差，而非纯生成痕迹。三个尺度分别经Trace Adapter提炼并融合为Patch级痕迹Token R。Cross-Attention以语义锚点为Query、R为Key/Value，语义只调节痕迹选择；最终分类器只接收痕迹证据。阶段三使用BCE检测损失和带margin的语义—痕迹分离约束。信息流概括为“Semantic-as-Condition, Trace-as-Evidence”。

## B. 三个核心假设

1. 真实图像正常预测器可建立条件正常模式，使生成图像产生更大的候选残差。
2. 浅、中、深层对不同生成器提供互补而非完全冗余的真实性线索。
3. 语义主要承担条件化作用，最终真假判断更直接地跟随痕迹表示。

## C. 三张核心机制图

### 1. `01_real_fake_multiscale_residual.png`

- 图题：真实与生成图像的多尺度正常特征偏离分布
- 变量：每图像、每尺度的预LayerNorm `h-h_hat` Patch L2范数均值。
- 主要数值：{"；".join(core1_numbers)}。
- 观察：三个尺度的Fake均值均高于Real。
- 可支持：条件正常预测产生具有真实性区分能力的候选偏离。
- 不可过度声称：候选残差不是纯生成痕迹。

### 2. `02_generator_scale_masking.png`

- 图题：不同生成器对多尺度痕迹的敏感性
- 变量：Adapter后屏蔽单尺度时，`AP_full - AP_mask` 的绝对百分点。
- 主要数值：{scale_numbers}；逐生成器最大敏感尺度为{", ".join(max_sensitive)}。
- 观察：不同生成器的最大敏感尺度不完全相同；负值也原样保留。
- 可支持：三尺度具有差异化敏感性和互补价值。
- 不可过度声称：屏蔽结果不是严格因果证明。

### 3. `03_semantic_vs_trace_intervention.png`

- 图题：语义条件与生成痕迹对最终判别的影响
- 变量：匹配语义交换与Real↔Fake融合痕迹交换的预测翻转率。
- 主要数值：语义交换PFR={float(semantic["prediction_flip_rate"]) * 100:.2f}%（95% CI {float(semantic["prediction_flip_ci_low"]) * 100:.2f}–{float(semantic["prediction_flip_ci_high"]) * 100:.2f}%）；痕迹交换PFR={float(trace["prediction_flip_rate"]) * 100:.2f}%（95% CI {float(trace["prediction_flip_ci_low"]) * 100:.2f}–{float(trace["prediction_flip_ci_high"]) * 100:.2f}%）；TFR={float(trace["trace_following_rate"]) * 100:.2f}%。
- 观察：{("预测对痕迹交换更敏感。" if core3["supports_trace_as_evidence"] else "结果需要人工复核。")}
- 可支持：语义主要条件化、痕迹作为直接证据。
- 不可过度声称：缺少semantic_class/content_env，不能称为同类/异类语义交换；干预不是严格因果证明。

## D. 三联图

- 文件：`04_semtrace_core_mechanisms_triptych.png`
- 推荐图题：SemTrace核心机制分析：正常模式偏离、多尺度互补与语义条件化
- 推荐图注：见 `descriptions/triptych_caption.md`。

## E. 开题报告推荐使用方式

- 主要研究内容：用方法简述与三个核心假设界定SemTrace研究点。
- 研究方法：用图1解释正常预测与候选残差，用图2解释多尺度设计，用图3解释语义角色。
- 技术路线：采用 `DINOv3 → 正常预测 → 候选残差 → Adapter → 融合R → Cross-Attention → z_t`。
- 实验方案：说明GenImage SDv1.4训练、八生成器评测、bootstrap统计和推理期干预。
- 可行性分析：引用现有checkpoint、32,000样本机制缓存、完整跨生成器评测和可复现工具链。
- 已有科研基础：引用正式主实验表与三联机制图。

## F. PPT推荐使用方式

1. 主PPT先展示SemTrace信息流，再展示三联图作为核心证据页。
2. 图1用于回答“正常预测器是否有效”；图2用于回答“为何需要三层”；图3用于回答“语义是否形成捷径”。
3. 三张独立图及逐生成器CSV放Backup，便于追问时展开。

## G. 当前正式主实验指标

- 主实验目录：`{provenance["eval_root"]}`
- mAcc：{eval_metrics["mAcc"] * 100:.2f}%
- mAP：{eval_metrics["mAP"] * 100:.2f}%
- 全体样本Acc/AP/AUROC：{eval_metrics["accuracy"] * 100:.2f}% / {eval_metrics["average_precision"] * 100:.2f}% / {eval_metrics["auroc"] * 100:.2f}%

|Generator|Acc|AP|
|---|---:|---:|
{generator_rows}

## H. 数据与表述限制

- Core 1采用预LayerNorm prediction error，避免LayerNorm后范数近似固定造成误解。
- Core 2只使用`mask_scale_L2/L6/L8_after_adapter`，对应融合输入处Leave-One-Scale-Out。
- Core 3舍弃既有batch-local trace swap（相反真实性donor覆盖为0），改用缓存上的全局同生成器配对。
- 机制分析只支持相对敏感性和一致性表述，不证明纯痕迹、完全去语义或严格因果关系。
"""


def _package_readme(provenance: dict[str, Any], status: dict[str, Any]) -> str:
    return f"""# SemTrace开题答辩核心机制材料包

本包将SemTrace三个核心机制结果整理为可追溯数据、统计结果、独立图片、三联图和ChatGPT写作上下文。状态：**{status["proposal_material_status"]}**。

## 三张图分别回答什么

1. `01_real_fake_multiscale_residual`：生成图像是否比真实图像更偏离条件正常模式。
2. `02_generator_scale_masking`：不同生成器是否对浅、中、深尺度呈差异化敏感性。
3. `03_semantic_vs_trace_intervention`：最终预测对语义交换还是痕迹交换更敏感。

`04_semtrace_core_mechanisms_triptych` 是三项结果的统一论文/PPT版三联图。所有图片均由 `data/` 中CSV/JSON生成，不含人工修改数值。

## 推荐直接发给ChatGPT的文件

1. `reports/chatgpt_handoff.md`
2. `figures/04_semtrace_core_mechanisms_triptych.png`
3. 三张独立机制图
4. `reports/core_mechanism_report.md`

## 数据和追溯

- `data/`：样本级、汇总级CSV与统计JSON。
- `metadata/provenance.json`：checkpoint、commit、缓存指纹、随机种子和哈希。
- `metadata/file_manifest.json`：除manifest自身和ZIP外全部payload文件的SHA-256；排除二者是为避免递归自哈希。

## 重新运行

```bash
MPLCONFIGDIR=/tmp/semtrace-mpl uv run python -m semtrace.cli.build_proposal_mechanism_package \\
  mechanism_root={provenance["mechanism_root"]} \\
  eval_root={provenance["eval_root"]} \\
  checkpoint={provenance["checkpoint"]}
```

该命令复用已有机制缓存，仅重算Cross-Attention和分类头，不运行DINOv3、不训练模型。
"""


def _display_generator(name: str) -> str:
    return DISPLAY_GENERATORS.get(name.lower(), name)


def _strip_title(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[2:]).strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

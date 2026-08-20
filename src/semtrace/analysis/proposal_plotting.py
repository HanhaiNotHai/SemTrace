from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from semtrace.analysis.proposal_mechanisms import SCALE_NAMES

REAL_COLOR = "#3274A1"
FAKE_COLOR = "#E1812C"
MASK_CMAP = "RdBu_r"
DISPLAY_GENERATORS = {
    "midjourney": "Midjourney",
    "sdv1.4": "SDv1.4",
    "sdv1.5": "SDv1.5",
    "adm": "ADM",
    "glide": "GLIDE",
    "wukong": "Wukong",
    "vqdm": "VQDM",
    "biggan": "BigGAN",
}


def configure_publication_style() -> str:
    """Select an installed CJK font and configure editable PDF text.

    Matplotlib font discovery follows its documented font-manager API:
    https://matplotlib.org/stable/api/font_manager_api.html
    """
    preferred = (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
    )
    installed: dict[str, str] = {}
    for path in font_manager.findSystemFonts():
        try:
            installed[font_manager.FontProperties(fname=path).get_name()] = path
        except RuntimeError:
            continue
    selected = next((font for font in preferred if font in installed), "DejaVu Sans")
    if selected == "DejaVu Sans":
        warnings.warn(
            "No preferred Chinese font was found; Chinese glyphs may not render correctly.",
            stacklevel=2,
        )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return selected


def generate_proposal_figures(
    core1_samples: pd.DataFrame,
    core2_table: pd.DataFrame,
    core3_summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    selected_layers: Sequence[int],
    dpi: int,
    formats: Sequence[str],
) -> list[Path]:
    configure_publication_style()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    normalized_formats = tuple(item.lower() for item in formats)
    if not normalized_formats or set(normalized_formats) - {"png", "pdf"}:
        raise ValueError("image formats must be a non-empty subset of png and pdf")
    if dpi < 72:
        raise ValueError("DPI must be at least 72")

    outputs: list[Path] = []
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    _plot_core1(axis, core1_samples, selected_layers=selected_layers)
    figure.suptitle("真实与生成图像的多尺度正常特征偏离分布", fontsize=17)
    outputs.extend(
        _save_figure(
            figure, destination, "01_real_fake_multiscale_residual", dpi, normalized_formats
        )
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.6, 6.3), constrained_layout=True)
    _plot_core2(axis, core2_table)
    figure.suptitle("不同生成器对多尺度痕迹的敏感性", fontsize=17)
    outputs.extend(
        _save_figure(figure, destination, "02_generator_scale_masking", dpi, normalized_formats)
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    _plot_core3(axis, core3_summary)
    figure.suptitle("语义条件与生成痕迹对最终判别的影响", fontsize=17)
    outputs.extend(
        _save_figure(
            figure, destination, "03_semantic_vs_trace_intervention", dpi, normalized_formats
        )
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.8), constrained_layout=True)
    _plot_core1(axes[0], core1_samples, selected_layers=selected_layers, compact=True)
    axes[0].set_title("(a) 多尺度正常特征偏离")
    _plot_core2(axes[1], core2_table, compact=True)
    axes[1].set_title("(b) 不同生成器的尺度互补性")
    _plot_core3(axes[2], core3_summary, compact=True)
    axes[2].set_title("(c) 语义条件与痕迹证据")
    figure.suptitle("SemTrace核心机制分析：正常模式偏离、多尺度互补与语义条件化", fontsize=18)
    outputs.extend(
        _save_figure(
            figure, destination, "04_semtrace_core_mechanisms_triptych", dpi, normalized_formats
        )
    )
    plt.close(figure)
    return outputs


def _plot_core1(
    axis: Axes,
    samples: pd.DataFrame,
    *,
    selected_layers: Sequence[int],
    compact: bool = False,
) -> None:
    positions: list[float] = []
    values: list[np.ndarray] = []
    colors: list[str] = []
    for scale_index, layer in enumerate(selected_layers):
        for label_index, (label, color) in enumerate(((0, REAL_COLOR), (1, FAKE_COLOR))):
            group = samples[(samples["selected_layer"] == layer) & (samples["label"] == label)][
                "residual_l2_mean"
            ].to_numpy(dtype=np.float64)
            if group.size == 0 or not np.isfinite(group).all():
                raise ValueError(f"invalid core1 plot group: layer={layer}, label={label}")
            positions.append(scale_index + (-0.18 if label_index == 0 else 0.18))
            values.append(group)
            colors.append(color)
    violins = axis.violinplot(
        values,
        positions=positions,
        widths=0.3,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    bodies = cast(Sequence[PolyCollection], violins["bodies"])
    for index, body in enumerate(bodies):
        color = colors[index]
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.72)
        body.set_linewidth(0.7)
    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.10,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)
    axis.scatter(
        positions,
        [float(np.mean(group)) for group in values],
        marker="D",
        s=22,
        color="black",
        zorder=4,
        label="均值",
    )
    axis.set_xticks(range(3), ["浅层", "中层", "深层"])
    axis.set_ylabel("图像级候选残差L2均值")
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=REAL_COLOR, label="真实图像"),
        Patch(facecolor=FAKE_COLOR, label="生成图像"),
    ]
    axis.legend(handles=handles, loc="best", frameon=False, ncol=2 if compact else 3)
    if compact:
        axis.tick_params(axis="x", pad=2)


def _plot_core2(axis: Axes, table: pd.DataFrame, *, compact: bool = False) -> None:
    columns = [f"delta_ap_{scale}_pp" for scale in SCALE_NAMES]
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"core2 plot missing columns: {sorted(missing)}")
    matrix = table[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("core2 plot contains NaN or Inf")
    limit = max(float(np.abs(matrix).max()), 0.1)
    image = axis.imshow(matrix, cmap=MASK_CMAP, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(3), ["屏蔽浅层", "屏蔽中层", "屏蔽深层"])
    axis.set_yticks(
        range(len(table)),
        [DISPLAY_GENERATORS.get(str(value).lower(), str(value)) for value in table["generator"]],
    )
    threshold = limit * 0.55
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=8.5 if compact else 10,
                color="white" if abs(value) > threshold else "black",
            )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("AP下降（百分点）")
    axis.set_xlabel("推理期Leave-One-Scale-Out干预")
    axis.set_ylabel("测试生成器")


def _plot_core3(axis: Axes, summary: pd.DataFrame, *, compact: bool = False) -> None:
    order = ("matched_semantic_swap", "real_fake_trace_swap")
    labels = ("同生成器/同真实性\n语义交换", "同生成器Real↔Fake\n痕迹交换")
    selected = summary.set_index("condition").loc[list(order)]
    means = selected["prediction_flip_rate"].to_numpy(dtype=np.float64) * 100.0
    lower = selected["prediction_flip_ci_low"].to_numpy(dtype=np.float64) * 100.0
    upper = selected["prediction_flip_ci_high"].to_numpy(dtype=np.float64) * 100.0
    errors = np.vstack((means - lower, upper - means))
    bars = axis.bar(
        range(2),
        means,
        yerr=errors,
        capsize=5,
        color=("#55A868", "#C44E52"),
        edgecolor="#333333",
        linewidth=0.8,
    )
    for bar, value in zip(bars, means, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(float(upper.max()) * 0.025, 0.4),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9 if compact else 11,
        )
    trace_following = selected.loc["real_fake_trace_swap", "trace_following_rate"]
    if pd.notna(trace_following):
        axis.text(
            0.98,
            0.96,
            f"痕迹跟随率 = {float(trace_following) * 100.0:.2f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9 if compact else 11,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
        )
    axis.set_xticks(range(2), labels)
    axis.set_ylabel("预测翻转率（%）")
    axis.set_ylim(0, max(float(upper.max()) * 1.22, 5.0))
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)


def _save_figure(
    figure: Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
    formats: Sequence[str],
) -> list[Path]:
    """Use Figure.savefig's documented format inference from the suffix.

    Source: https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html
    """
    paths = []
    for image_format in formats:
        path = output_dir / f"{stem}.{image_format}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    return paths

"""Baseline vs Optimized KPI comparison charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

METRIC_SPECS: list[tuple[str, str, str, int]] = [
    ("avg_wait_time_min", "평균 대기시간", "분", 0),
    ("dispatch_success_rate", "배차 성공률", "%", 1),
    ("vehicle_utilization", "차량 이용률", "%", 1),
    ("total_distance_km", "총 운행거리", "km", 0),
]

DISPATCHER_LABELS = {
    "baseline": "Baseline",
    "optimized": "Optimized",
}

PALETTE = {
    "baseline": "#94A3B8",   # slate
    "optimized": "#0D9488",   # teal
}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FAFBFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#E2E8F0",
            "axes.labelcolor": "#334155",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.unicode_minus": False,
        }
    )
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for family in ("AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"):
        if family in available:
            plt.rcParams["font.family"] = family
            break


def _format_value(metric: str, value: float) -> str:
    if metric == "dispatch_success_rate" or metric == "vehicle_utilization":
        return f"{value * 100:.1f}%"
    if metric == "avg_wait_time_min":
        return f"{value:.1f}"
    return f"{value:,.0f}"


def _improvement_text(metric: str, baseline_mean: float, optimized_mean: float) -> str | None:
    if baseline_mean == 0:
        return None
    pct = (optimized_mean - baseline_mean) / abs(baseline_mean) * 100
    lower_is_better = metric in ("avg_wait_time_min", "total_distance_km")
    improved = pct < 0 if lower_is_better else pct > 0
    if not improved:
        sign = "↑" if pct > 0 else "↓"
        return f"{sign}{abs(pct):.1f}%"
    sign = "↓" if pct < 0 else "↑"
    color_note = "개선" if improved else ""
    return f"{sign}{abs(pct):.1f}% {color_note}".strip()


def plot_dispatcher_comparison(
    results_df: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str = "Baseline vs Optimized — KPI 비교 (영종 I-MOD)",
    dpi: int = 200,
) -> Path:
    """Save a 1×4 bar chart comparing baseline and optimized dispatchers."""
    _setup_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=False)
    fig.patch.set_facecolor("#FAFBFC")

    order = ["baseline", "optimized"]

    for ax, (metric, label_ko, unit, _decimals) in zip(axes, METRIC_SPECS):
        plot_df = results_df.copy()
        if metric in ("dispatch_success_rate", "vehicle_utilization"):
            plot_df = plot_df.assign(**{metric: plot_df[metric] * 100})
            y_label = unit
        else:
            y_label = unit

        sns.barplot(
            data=plot_df,
            x="dispatcher",
            y=metric,
            order=order,
            hue="dispatcher",
            palette=PALETTE,
            ax=ax,
            errorbar="sd",
            capsize=0.12,
            err_kws={"linewidth": 1.5, "color": "#334155"},
            legend=False,
            width=0.62,
        )

        ax.set_title(label_ko, pad=12, fontsize=13, fontweight="bold", color="#0F172A")
        ax.set_xlabel("")
        ax.set_ylabel(y_label, fontsize=10, color="#64748B")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([DISPATCHER_LABELS[d] for d in order], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        grouped = results_df.groupby("dispatcher")[metric]
        b_mean = float(grouped.mean().get("baseline", 0))
        o_mean = float(grouped.mean().get("optimized", 0))
        display_b = b_mean * 100 if metric in ("dispatch_success_rate", "vehicle_utilization") else b_mean
        display_o = o_mean * 100 if metric in ("dispatch_success_rate", "vehicle_utilization") else o_mean

        for patch, val in zip(ax.patches, [display_b, display_o]):
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + ax.get_ylim()[1] * 0.02,
                _format_value(metric, val if metric not in ("dispatch_success_rate", "vehicle_utilization") else val / 100),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="600",
                color="#334155",
            )

        delta = _improvement_text(metric, b_mean, o_mean)
        if delta:
            improved = (
                (o_mean < b_mean and metric in ("avg_wait_time_min", "total_distance_km"))
                or (o_mean > b_mean and metric not in ("avg_wait_time_min", "total_distance_km"))
            )
            badge_color = "#059669" if improved else "#64748B"
            ax.text(
                0.5,
                0.97,
                delta,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=badge_color, edgecolor="none", alpha=0.92),
            )

    legend_handles = [
        mpatches.Patch(color=PALETTE["baseline"], label="Baseline (가까운 차량 + 쌓임)"),
        mpatches.Patch(color=PALETTE["optimized"], label="Optimized (경로 삽입·제약·수요)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=1.02,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path

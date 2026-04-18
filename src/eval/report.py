"""Render the evaluation report (markdown) and Hit@k chart."""

from __future__ import annotations

import sys
from pathlib import Path

from src.config import settings
from src.eval.metrics import ModeMetrics, per_mode_metrics

MODE_ORDER = ["text_only", "vision_only", "hybrid"]
K_VALUES = [1, 3, 5]


def _format_float(v: float, digits: int = 3) -> str:
    """Format a float with fixed digits."""
    return f"{v:.{digits}f}"


def _build_retrieval_table(metrics: dict[str, ModeMetrics]) -> str:
    """Build the Hit@k + MRR comparison table."""
    header = "| Mode | N | Hit@1 | Hit@3 | Hit@5 | MRR |\n"
    sep = "|------|---|-------|-------|-------|-----|\n"
    rows: list[str] = []
    for mode in MODE_ORDER:
        if mode not in metrics:
            continue
        m = metrics[mode]
        rows.append(
            f"| {mode} | {m['n']} | "
            f"{_format_float(m['hit_at_1'])} | "
            f"{_format_float(m['hit_at_3'])} | "
            f"{_format_float(m['hit_at_5'])} | "
            f"{_format_float(m['mrr'])} |\n"
        )
    return header + sep + "".join(rows)


def _build_latency_table(metrics: dict[str, ModeMetrics]) -> str:
    """Build the latency table."""
    header = "| Mode | Mean (ms) | P50 (ms) | P95 (ms) |\n"
    sep = "|------|-----------|----------|----------|\n"
    rows: list[str] = []
    for mode in MODE_ORDER:
        if mode not in metrics:
            continue
        m = metrics[mode]
        rows.append(
            f"| {mode} | "
            f"{_format_float(m['mean_latency_ms'], 1)} | "
            f"{_format_float(m['p50_latency_ms'], 1)} | "
            f"{_format_float(m['p95_latency_ms'], 1)} |\n"
        )
    return header + sep + "".join(rows)


def render_markdown(metrics: dict[str, ModeMetrics], chart_rel_path: str) -> str:
    """Render the full report markdown."""
    if not metrics:
        return (
            "# Evaluation Report\n\n"
            "No run files were found under `data/eval/runs/`. "
            "Run `uv run python -m src.eval.run` first.\n"
        )

    lines: list[str] = []
    lines.append("# Evaluation Report\n\n")
    lines.append(
        "This report summarizes retrieval quality (Hit@k, MRR), latency, "
        "and Gemini cache-hit rate across the three retrieval modes.\n\n"
    )

    lines.append("## Retrieval Quality\n\n")
    lines.append(_build_retrieval_table(metrics))
    lines.append("\n")

    lines.append(f"![Hit@k per mode]({chart_rel_path})\n\n")

    lines.append("## Latency\n\n")
    lines.append(_build_latency_table(metrics))
    lines.append("\n")

    # Cache hit rate is only meaningful for hybrid (only mode that calls Gemini).
    hybrid = metrics.get("hybrid")
    if hybrid is not None:
        lines.append("## Cache Hit Rate (hybrid, Gemini)\n\n")
        lines.append(
            f"- Hybrid cache hit rate: **{_format_float(hybrid['cache_hit_rate'])}** "
            f"(n={hybrid['n']})\n\n"
        )

    lines.append("## Scope Notes\n\n")
    lines.append(
        "- Faithfulness and answer-relevancy (Ragas-style generation metrics) are "
        "**deferred to future work** for v1. They require heavyweight dependencies "
        "and additional LLM judge calls that we skip here to stay within the "
        "lightweight-stack constraint.\n"
        "- Top-k for retrieval evaluation is 5.\n"
        "- Generation is run on the hybrid result only to conserve Gemini free-tier "
        "quota (~10 RPM for 2.5 Flash).\n"
    )

    return "".join(lines)


def render_chart(metrics: dict[str, ModeMetrics], chart_path: Path) -> bool:
    """Render grouped bar chart: modes (x) x k values (grouped). Returns success."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib not available: {e}", file=sys.stderr)
        return False

    modes_present = [m for m in MODE_ORDER if m in metrics]
    if not modes_present:
        print("No modes present; skipping chart.", file=sys.stderr)
        return False

    # Prepare data: rows=modes, cols=k values.
    data: list[list[float]] = []
    for mode in modes_present:
        m = metrics[mode]
        data.append([m["hit_at_1"], m["hit_at_3"], m["hit_at_5"]])

    chart_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    n_modes = len(modes_present)
    n_k = len(K_VALUES)
    bar_width = 0.8 / n_modes
    x_positions = list(range(n_k))

    for i, mode in enumerate(modes_present):
        offsets = [x + (i - (n_modes - 1) / 2) * bar_width for x in x_positions]
        ax.bar(offsets, data[i], width=bar_width, label=mode)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Hit@{k}" for k in K_VALUES])
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Retrieval Hit@k by Mode")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Wrote chart: {chart_path}")
    return True


def main() -> None:
    """CLI entry point."""
    runs_dir = settings.eval_dir / "runs"
    report_path = settings.eval_dir / "report.md"
    charts_dir = settings.eval_dir / "charts"
    chart_path = charts_dir / "hit_at_k.png"

    metrics = per_mode_metrics(runs_dir)
    if not metrics:
        print(f"Warning: no runs found in {runs_dir}.", file=sys.stderr)

    render_chart(metrics, chart_path)

    # Use forward slash relative path so markdown renders on all platforms.
    rel_chart = "charts/hit_at_k.png"
    md = render_markdown(metrics, rel_chart)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()

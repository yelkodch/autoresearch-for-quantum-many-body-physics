from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "autoresearch_j1j2_mplconfig"),
)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = ROOT / "analysis" / "public_recipe_search_trajectory.json"
FIGURES_DIR = ROOT / "docs" / "figures"
OUTPUT_PATH = FIGURES_DIR / "recipe_search_summary.png"


def load_trajectory() -> dict:
    return json.loads(TRAJECTORY_PATH.read_text())


def fmt_score(value: float) -> str:
    return f"{value:.6f}"


def make_recipe_search_summary() -> Path:
    payload = load_trajectory()
    rows = payload["iterations"]
    iterations = [int(row["iteration"]) for row in rows]
    scores = [float(row["panel_score"]) for row in rows]
    statuses = [str(row["status"]) for row in rows]

    keep_x = [x for x, status in zip(iterations, statuses) if status == "keep"]
    keep_y = [y for y, status in zip(scores, statuses) if status == "keep"]
    discard_x = [x for x, status in zip(iterations, statuses) if status == "discard"]
    discard_y = [y for y, status in zip(scores, statuses) if status == "discard"]

    best_idx = min(range(len(scores)), key=scores.__getitem__)
    best_iteration = iterations[best_idx]
    best_score = scores[best_idx]
    initial_score = scores[0]
    improvement = best_score - initial_score

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 1.0,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
            "axes.labelcolor": "#0f172a",
            "font.family": "DejaVu Sans",
        }
    )

    fig = plt.figure(figsize=(11.8, 4.9), dpi=200)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.55])
    ax = fig.add_subplot(grid[0, 0])
    side = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.87, wspace=0.02)

    fig.suptitle("Fixed-Panel Recipe Search", x=0.065, y=0.985, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.065,
        0.93,
        f"{payload['system_label']}  •  {payload['benchmark_label']}  •  lower is better",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#475569",
    )

    ax.plot(iterations, scores, color="#2563eb", linewidth=2.2, zorder=2)
    ax.scatter(keep_x, keep_y, color="#16a34a", s=36, zorder=4)
    ax.scatter(discard_x, discard_y, color="#ef4444", s=28, marker="x", linewidths=1.6, zorder=4)
    ax.scatter([best_iteration], [best_score], color="#0f172a", s=68, zorder=5)

    ax.axhline(initial_score, color="#94a3b8", linestyle="--", linewidth=1.1, alpha=0.9)
    ax.axhline(best_score, color="#0f172a", linestyle=":", linewidth=1.2, alpha=0.9)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Panel score")
    ax.set_xlim(min(iterations) - 0.3, max(iterations) + 0.6)
    pad = (max(scores) - min(scores)) * 0.08
    ax.set_ylim(min(scores) - pad, max(scores) + pad)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.5, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    side.axis("off")
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.text(0.0, 0.95, "Curated snapshot", fontsize=12.5, fontweight="bold", va="top")
    side.scatter([0.02], [0.84], s=34, color="#16a34a", transform=side.transAxes)
    side.text(0.10, 0.84, "accepted", fontsize=10.2, va="center", transform=side.transAxes)
    side.scatter([0.02], [0.78], s=34, color="#ef4444", marker="x", linewidths=1.5, transform=side.transAxes)
    side.text(0.10, 0.78, "rejected", fontsize=10.2, va="center", transform=side.transAxes)
    side.scatter([0.02], [0.72], s=42, color="#0f172a", transform=side.transAxes)
    side.text(0.10, 0.72, "best", fontsize=10.2, va="center", transform=side.transAxes)
    side.text(
        0.0,
        0.64,
        "\n".join(
            [
                f"Iterations      {len(rows)}",
                f"Accepted        {sum(1 for status in statuses if status == 'keep')}",
                f"Rejected        {sum(1 for status in statuses if status == 'discard')}",
                "",
                f"Initial score   {fmt_score(initial_score)}",
                f"Best score      {fmt_score(best_score)}",
                f"Improvement     {fmt_score(improvement)}",
                f"Best iteration  {best_iteration}",
            ]
        ),
        family="DejaVu Sans Mono",
        fontsize=10.2,
        va="top",
        color="#1e293b",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        linespacing=1.45,
    )
    side.text(
        0.0,
        0.18,
        "The search improves a stable short-budget benchmark while keeping the physical setup fixed.",
        fontsize=9.5,
        color="#475569",
        va="top",
        wrap=True,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, facecolor="white")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = make_recipe_search_summary()
    print(path)

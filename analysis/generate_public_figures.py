from __future__ import annotations

import json
import os
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
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "DejaVu Sans",
        }
    )

    fig, ax = plt.subplots(figsize=(10.6, 5.15), dpi=200)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.79)

    fig.text(
        0.10,
        0.95,
        "Fixed-Panel Recipe Search",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.10,
        0.895,
        f"{payload['system_label']}  •  {payload['benchmark_label']}",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#4b5563",
    )

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.plot(iterations, scores, color="#374151", linewidth=2.0, zorder=2)
    ax.scatter(keep_x, keep_y, s=34, color="#f59e0b", edgecolors="white", linewidths=0.6, zorder=4)
    ax.scatter(discard_x, discard_y, s=28, color="#9ca3af", marker="x", linewidths=1.6, zorder=4)
    ax.scatter([best_iteration], [best_score], s=64, color="#111827", zorder=5)

    ax.axhline(initial_score, color="#cbd5e1", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(best_score, color="#f59e0b", linestyle=":", linewidth=1.2, zorder=1)

    ax.annotate(
        f"best  {fmt_score(best_score)}",
        xy=(best_iteration, best_score),
        xytext=(-8, -28),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9.2,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db"},
    )

    ax.set_xlabel("Iteration", color="#111827")
    ax.set_ylabel("Panel score", color="#111827")
    ax.set_xlim(min(iterations) - 0.3, max(iterations) + 0.4)
    pad = (max(scores) - min(scores)) * 0.08
    ax.set_ylim(min(scores) - pad, max(scores) + pad)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.grid(axis="x", color="#f3f4f6", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#374151")

    stats_text = "\n".join(
        [
            f"iterations   {len(rows)}",
            f"accepted     {sum(1 for status in statuses if status == 'keep')}",
            f"rejected     {sum(1 for status in statuses if status == 'discard')}",
            "",
            f"initial      {fmt_score(initial_score)}",
            f"best         {fmt_score(best_score)}",
            f"improvement  {fmt_score(improvement)}",
            f"best iter    {best_iteration}",
            "",
            "orange  accepted",
            "gray x  rejected",
        ]
    )
    ax.text(
        0.975,
        0.94,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        family="DejaVu Sans Mono",
        fontsize=9.1,
        color="#1f2937",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "#f9fafb", "edgecolor": "#d1d5db"},
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, facecolor="white")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(make_recipe_search_summary())

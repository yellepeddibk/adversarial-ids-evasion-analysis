"""
Generate visuals for Bhargav's slides in the midterm presentation.

Slides covered:
  - Slide 1  : Title card
  - Slide 2  : Motivation diagram (normal/attack traffic vs perturbed evasion)
  - Slide 3  : Project objectives pipeline
  - Slide 14 : Challenges & Limitations (icon + text layout)

Outputs PNGs into presentation/visuals/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
NAVY = "#0B1F3A"
ACCENT = "#1F8FFF"
DANGER = "#E63946"
SAFE = "#2A9D8F"
LIGHT = "#F5F7FA"
DARK_TEXT = "#1A1A1A"
MUTED = "#6C757D"

OUT_DIR = Path(__file__).resolve().parents[1] / "visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Slide 1 - Title card
# ---------------------------------------------------------------------------
def slide1_title() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Decorative grid lines (subtle cyber feel)
    for x in range(0, 17):
        ax.axvline(x, color="#FFFFFF", alpha=0.04, linewidth=0.5)
    for y in range(0, 10):
        ax.axhline(y, color="#FFFFFF", alpha=0.04, linewidth=0.5)

    # Accent bar
    ax.add_patch(mpatches.Rectangle((0.8, 6.5), 0.15, 1.6, color=ACCENT))

    ax.text(
        1.2, 7.6,
        "Adversarial Robustness Analysis of",
        color="white", fontsize=28, fontweight="bold", va="center",
    )
    ax.text(
        1.2, 6.85,
        "Neural Network-Based Intrusion Detection Systems",
        color="white", fontsize=28, fontweight="bold", va="center",
    )
    ax.text(
        1.2, 6.1,
        "Using FGSM and PGD Attacks",
        color=ACCENT, fontsize=24, fontweight="bold", va="center",
    )

    # Divider
    ax.plot([1.2, 14.8], [5.4, 5.4], color="white", alpha=0.3, linewidth=1)

    ax.text(1.2, 4.7, "Midterm Project Presentation",
            color="white", fontsize=18, va="center", style="italic")

    ax.text(1.2, 3.5, "Authors", color=MUTED, fontsize=14, va="center")
    ax.text(1.2, 2.8, "Conrad Miller   ·   Prathik Bengaluru Prabhakara   ·   Bhargav Yellepeddi",
            color="white", fontsize=18, va="center")

    # Footer tag
    ax.text(15.2, 0.6, "NSL-KDD  |  PyTorch  |  Adversarial ML",
            color=MUTED, fontsize=11, ha="right", va="center", style="italic")

    _save(fig, "slide01_title.png")


# ---------------------------------------------------------------------------
# Slide 2 - Motivation
# ---------------------------------------------------------------------------
def _draw_box(ax, x, y, w, h, text, fc, ec=DARK_TEXT, fontsize=12, color="white", bold=True):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.5, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", color=color,
        fontsize=fontsize, fontweight="bold" if bold else "normal",
    )


def _arrow(ax, x1, y1, x2, y2, color=DARK_TEXT, lw=2):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_width=8,head_length=10",
        color=color, linewidth=lw, mutation_scale=1,
    )
    ax.add_patch(a)


def slide2_motivation() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.4, "Why Adversarial Robustness in IDS?",
            ha="center", fontsize=24, fontweight="bold", color=DARK_TEXT)

    # Row 1: Normal traffic -> IDS -> Allowed
    _draw_box(ax, 0.6, 6.0, 3.2, 1.1, "Normal Traffic", SAFE)
    _arrow(ax, 3.9, 6.55, 5.8, 6.55)
    _draw_box(ax, 5.9, 6.0, 3.2, 1.1, "IDS Model", NAVY)
    _arrow(ax, 9.2, 6.55, 11.0, 6.55, color=SAFE, lw=2.5)
    _draw_box(ax, 11.1, 6.0, 4.0, 1.1, "Correctly Allowed  ✓", SAFE)

    # Row 2: Attack traffic -> IDS -> Detected
    _draw_box(ax, 0.6, 4.2, 3.2, 1.1, "Attack Traffic", DANGER)
    _arrow(ax, 3.9, 4.75, 5.8, 4.75)
    _draw_box(ax, 5.9, 4.2, 3.2, 1.1, "IDS Model", NAVY)
    _arrow(ax, 9.2, 4.75, 11.0, 4.75, color=SAFE, lw=2.5)
    _draw_box(ax, 11.1, 4.2, 4.0, 1.1, "Correctly Detected  ✓", SAFE)

    # Row 3: Perturbed attack -> IDS -> Evades
    _draw_box(ax, 0.6, 2.0, 3.2, 1.4,
              "Attack Traffic\n+ ε perturbation", DANGER, fontsize=11)
    _arrow(ax, 3.9, 2.7, 5.8, 2.7)
    _draw_box(ax, 5.9, 2.0, 3.2, 1.4, "IDS Model", NAVY, fontsize=12)
    _arrow(ax, 9.2, 2.7, 11.0, 2.7, color=DANGER, lw=2.5)
    _draw_box(ax, 11.1, 2.0, 4.0, 1.4,
              "Misclassified as Normal  ✗\n(Evasion)", DANGER, fontsize=11)

    # Bottom callout
    ax.text(
        8, 0.7,
        "Small, crafted perturbations preserve attack semantics but fool the model.",
        ha="center", fontsize=14, color=DARK_TEXT, style="italic",
    )

    _save(fig, "slide02_motivation.png")


# ---------------------------------------------------------------------------
# Slide 3 - Objectives pipeline
# ---------------------------------------------------------------------------
def slide3_objectives() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.2, "Project Objectives",
            ha="center", fontsize=26, fontweight="bold", color=DARK_TEXT)
    ax.text(8, 7.5, "A complete attack–defense pipeline",
            ha="center", fontsize=15, color=MUTED, style="italic")

    steps = [
        ("1", "Train\nBaseline MLP", NAVY),
        ("2", "Attack\nFGSM & PGD", DANGER),
        ("3", "Measure\nDegradation", "#F4A261"),
        ("4", "Defend\nAdv. Training", ACCENT),
        ("5", "Evaluate\nTrade-off", SAFE),
    ]

    box_w, box_h = 2.5, 2.0
    gap = 0.4
    total_w = len(steps) * box_w + (len(steps) - 1) * gap
    start_x = (16 - total_w) / 2
    y = 3.3

    for i, (num, label, color) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        _draw_box(ax, x, y, box_w, box_h, "", color)
        ax.text(x + box_w / 2, y + box_h - 0.45, num,
                ha="center", va="center", color="white",
                fontsize=22, fontweight="bold")
        ax.text(x + box_w / 2, y + box_h / 2 - 0.25, label,
                ha="center", va="center", color="white", fontsize=13,
                fontweight="bold")
        if i < len(steps) - 1:
            ax2 = x + box_w
            _arrow(ax, ax2 + 0.05, y + box_h / 2,
                   ax2 + gap - 0.05, y + box_h / 2,
                   color=DARK_TEXT, lw=2)

    # Bottom labels for each phase
    phase_labels = ["Setup", "Attack", "Analyze", "Defend", "Conclude"]
    for i, lbl in enumerate(phase_labels):
        x = start_x + i * (box_w + gap) + box_w / 2
        ax.text(x, y - 0.5, lbl, ha="center", color=MUTED,
                fontsize=12, style="italic")

    _save(fig, "slide03_objectives.png")


# ---------------------------------------------------------------------------
# Slide 14 - Challenges & Limitations
# ---------------------------------------------------------------------------
def _icon_circle(ax, cx, cy, r, color, glyph):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=color, zorder=2))
    ax.text(cx, cy, glyph, ha="center", va="center",
            color="white", fontsize=24, fontweight="bold", zorder=3)


def slide14_challenges() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.3, "Challenges and Limitations",
            ha="center", fontsize=26, fontweight="bold", color=DARK_TEXT)

    items = [
        (
            "1", NAVY, "Tabular Constraints",
            "Perturbations restricted to continuous features.\n"
            "May understate worst-case vulnerability.",
        ),
        (
            "2", ACCENT, "Single Dataset",
            "Evaluated only on NSL-KDD.\n"
            "Generalization to other IDS datasets unverified.",
        ),
        (
            "3", "#F4A261", "Single Architecture",
            "Only an MLP tested.\n"
            "CNN / transformer behavior on tabular data unexplored.",
        ),
        (
            "4", DANGER, "Fixed Training ε",
            "Adversarial training used a single ε = 0.1.\n"
            "May be suboptimal at very large ε values.",
        ),
        (
            "5", SAFE, "Runtime Cost",
            "PGD and adversarial training are expensive at scale.\n"
            "Limits real-time deployment scenarios.",
        ),
    ]

    # Two-column layout: 3 left, 2 right
    col_x = [1.0, 8.4]
    row_y = [6.0, 4.0, 2.0]

    positions = [
        (col_x[0], row_y[0]),
        (col_x[0], row_y[1]),
        (col_x[0], row_y[2]),
        (col_x[1], row_y[0]),
        (col_x[1], row_y[1]),
    ]

    for (glyph, color, title, body), (x, y) in zip(items, positions):
        # Icon circle
        _icon_circle(ax, x + 0.6, y + 0.75, 0.55, color, glyph)
        # Title
        ax.text(x + 1.5, y + 1.1, title, fontsize=15,
                fontweight="bold", color=DARK_TEXT, va="center")
        # Body
        ax.text(x + 1.5, y + 0.35, body, fontsize=11,
                color=MUTED, va="center")

    _save(fig, "slide14_challenges.png")


# ---------------------------------------------------------------------------
def main() -> None:
    slide1_title()
    slide2_motivation()
    slide3_objectives()
    slide14_challenges()


if __name__ == "__main__":
    main()

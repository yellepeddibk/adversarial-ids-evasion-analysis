"""
Generate visuals for Bhargav's slides in the project presentation.

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
# Style constants (matched to team slide style)
# ---------------------------------------------------------------------------
BG = "#050607"
TITLE_RED = "#FF2A2A"
DIVIDER_GOLD = "#C5A43B"

NAVY = "#0B1F3A"
PANEL_BASE = "#101A2F"
ACCENT = "#1F8FFF"
DANGER = "#E63946"
SAFE = "#00D084"
CYAN = "#00B8FF"
PINK = "#FF2D72"
YELLOW = "#D9B24C"

TEXT_MAIN = "#EAF0FF"
TEXT_MID = "#B5C2DE"
TEXT_MUTED = "#8F9BB5"

OUT_DIR = Path(__file__).resolve().parents[1] / "visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {path}")


def _header(ax, title: str) -> None:
    ax.text(0.3, 8.4, title, ha="left", va="center", fontsize=18, fontweight="bold", color=TITLE_RED)
    ax.plot([0.3, 15.7], [7.9, 7.9], color=DIVIDER_GOLD, alpha=0.9, linewidth=1)


def _panel(ax, x, y, w, h, edge_color, title=None, alpha=0.95):
    panel = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=edge_color,
        facecolor=PANEL_BASE,
        alpha=alpha,
    )
    ax.add_patch(panel)
    if title:
        ax.text(x + 0.25, y + h - 0.4, title, color=edge_color, fontsize=12, fontweight="bold", va="top")


# ---------------------------------------------------------------------------
# Slide 1 - Title card
# ---------------------------------------------------------------------------
def slide1_title() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    _header(ax, "Project Presentation")

    _panel(ax, 0.9, 2.2, 14.2, 4.9, CYAN)

    ax.text(
        1.4, 6.4,
        "Adversarial Robustness Analysis of",
        color=TEXT_MAIN, fontsize=28, fontweight="bold", va="center",
    )
    ax.text(
        1.4, 5.7,
        "Neural Network-Based Intrusion Detection Systems",
        color=TEXT_MAIN, fontsize=28, fontweight="bold", va="center",
    )
    ax.text(
        1.4, 5.0,
        "Using FGSM and PGD Attacks",
        color=ACCENT, fontsize=24, fontweight="bold", va="center",
    )

    ax.text(1.4, 3.8, "Authors", color=TEXT_MUTED, fontsize=13, va="center")
    ax.text(1.4, 3.2, "Conrad Miller   ·   Prathik Bengaluru Prabhakara   ·   Bhargav Yellepeddi",
            color=TEXT_MAIN, fontsize=16, va="center")

    ax.text(15.2, 0.6, "NSL-KDD   |   PyTorch   |   Adversarial ML",
            color=TEXT_MUTED, fontsize=11, ha="right", va="center")

    _save(fig, "slide01_title.png")


# ---------------------------------------------------------------------------
# Slide 2 - Motivation
# ---------------------------------------------------------------------------
def _draw_box(ax, x, y, w, h, text, fc, ec=CYAN, fontsize=12, color=TEXT_MAIN, bold=True):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", color=color,
        fontsize=fontsize, fontweight="bold" if bold else "normal",
    )


def _arrow(ax, x1, y1, x2, y2, color=TEXT_MID, lw=2):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_width=8,head_length=10",
        color=color, linewidth=lw, mutation_scale=1,
    )
    ax.add_patch(a)


def slide2_motivation() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    _header(ax, "Motivation & Relevance")

    ax.text(8, 7.3, "Why Adversarial Robustness in IDS?",
        ha="center", fontsize=20, fontweight="bold", color=TEXT_MAIN)

    # Row 1: Normal traffic -> IDS -> Allowed
    _draw_box(ax, 0.6, 5.6, 3.2, 1.1, "Normal Traffic", PANEL_BASE, ec=SAFE)
    _arrow(ax, 3.9, 6.15, 5.8, 6.15)
    _draw_box(ax, 5.9, 5.6, 3.2, 1.1, "IDS Model", PANEL_BASE, ec=CYAN)
    _arrow(ax, 9.2, 6.15, 11.0, 6.15, color=SAFE, lw=2.5)
    _draw_box(ax, 11.1, 5.6, 4.0, 1.1, "Correctly Allowed  ✓", PANEL_BASE, ec=SAFE)

    # Row 2: Attack traffic -> IDS -> Detected
    _draw_box(ax, 0.6, 3.8, 3.2, 1.1, "Attack Traffic", PANEL_BASE, ec=PINK)
    _arrow(ax, 3.9, 4.35, 5.8, 4.35)
    _draw_box(ax, 5.9, 3.8, 3.2, 1.1, "IDS Model", PANEL_BASE, ec=CYAN)
    _arrow(ax, 9.2, 4.35, 11.0, 4.35, color=SAFE, lw=2.5)
    _draw_box(ax, 11.1, 3.8, 4.0, 1.1, "Correctly Detected  ✓", PANEL_BASE, ec=SAFE)

    # Row 3: Perturbed attack -> IDS -> Evades
    _draw_box(ax, 0.6, 1.7, 3.2, 1.4,
          "Attack Traffic\n+ ε perturbation", PANEL_BASE, ec=DANGER, fontsize=11)
    _arrow(ax, 3.9, 2.4, 5.8, 2.4)
    _draw_box(ax, 5.9, 1.7, 3.2, 1.4, "IDS Model", PANEL_BASE, ec=CYAN, fontsize=12)
    _arrow(ax, 9.2, 2.4, 11.0, 2.4, color=DANGER, lw=2.5)
    _draw_box(ax, 11.1, 1.7, 4.0, 1.4,
          "Misclassified as Normal  ✗\n(Evasion)", PANEL_BASE, ec=DANGER, fontsize=11)

    # Bottom callout
    ax.text(
        8, 0.7,
        "Small, crafted perturbations preserve attack semantics but fool the model.",
        ha="center", fontsize=13, color=TEXT_MID, style="italic",
    )

    _save(fig, "slide02_motivation.png")


# ---------------------------------------------------------------------------
# Slide 3 - Objectives pipeline
# ---------------------------------------------------------------------------
def slide3_objectives() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    _header(ax, "Project Objectives")
    ax.text(8, 7.2, "A complete attack-defense pipeline",
        ha="center", fontsize=15, color=TEXT_MID, style="italic")

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
            ha="center", va="center", color=TEXT_MAIN,
                fontsize=22, fontweight="bold")
        ax.text(x + box_w / 2, y + box_h / 2 - 0.25, label,
            ha="center", va="center", color=TEXT_MAIN, fontsize=13,
                fontweight="bold")
        if i < len(steps) - 1:
            ax2 = x + box_w
            _arrow(ax, ax2 + 0.05, y + box_h / 2,
                   ax2 + gap - 0.05, y + box_h / 2,
               color=TEXT_MID, lw=2)

    # Bottom labels for each phase
    phase_labels = ["Setup", "Attack", "Analyze", "Defend", "Conclude"]
    for i, lbl in enumerate(phase_labels):
        x = start_x + i * (box_w + gap) + box_w / 2
        ax.text(x, y - 0.5, lbl, ha="center", color=TEXT_MUTED,
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
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    _header(ax, "Challenges and Limitations")

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
        _panel(ax, x, y, 6.5, 1.6, color)
        _icon_circle(ax, x + 0.6, y + 0.8, 0.45, color, glyph)
        # Title
        ax.text(x + 1.5, y + 1.1, title, fontsize=15,
            fontweight="bold", color=TEXT_MAIN, va="center")
        # Body
        ax.text(x + 1.5, y + 0.35, body, fontsize=11,
            color=TEXT_MID, va="center")

    _save(fig, "slide14_challenges.png")


# ---------------------------------------------------------------------------
def main() -> None:
    slide1_title()
    slide2_motivation()
    slide3_objectives()
    slide14_challenges()


if __name__ == "__main__":
    main()

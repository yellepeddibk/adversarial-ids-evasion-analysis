from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_PATH = Path(__file__).resolve().parents[1] / "visuals" / "architecture_diagram.png"

BG = "#FFFFFF"
BOX_BG = "#FBFBFB"
BOX_EDGE = "#A7A7A7"
TEXT = "#222222"
ARROW = "#666666"
ACCENT = "#7A7A7A"


def draw_box(ax, x, y, w, h, title, body, fontsize=8.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        facecolor=BOX_BG,
        edgecolor=BOX_EDGE,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        title + "\n" + body,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        linespacing=1.25,
    )


def draw_arrow(ax, start, end, dashed=False, label=None, label_xy=None):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        linestyle="--" if dashed else "-",
        color=ACCENT if dashed else ARROW,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(label_xy[0], label_xy[1], label, fontsize=7.5, color=TEXT)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 6.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    boxes = {
        "dataset": (0.1, 6.9, 2.6, 1.4),
        "prep": (3.1, 6.9, 3.1, 1.4),
        "split": (6.7, 6.9, 2.1, 1.4),
        "model": (9.3, 6.5, 3.7, 1.9),
        "baseline": (1.5, 4.6, 3.1, 1.4),
        "defense": (6.0, 4.0, 3.5, 1.8),
        "trained": (10.0, 4.6, 2.8, 1.4),
        "attacks": (0.1, 0.7, 3.8, 1.8),
        "validity": (4.5, 0.9, 3.6, 1.4),
        "eval": (9.3, 0.6, 3.0, 1.8),
        "report": (12.8, 0.6, 3.1, 1.8),
    }

    draw_box(ax, *boxes["dataset"], "Dataset", "NSL-KDD (tabular network traffic)\nLabel: Normal vs Attack (binary)")
    draw_box(ax, *boxes["prep"], "Preprocessing", "One-hot encode categorical\nStandardize continuous\nBinary label mapping")
    draw_box(ax, *boxes["split"], "Split", "Train / Validation / Test")
    draw_box(ax, *boxes["model"], "Model: MLP (Tabular Classifier)", "Input: processed features\nHidden1: 64 (ReLU)\nHidden2: 32 (ReLU)\nOutput: 1 (logit)\nLoss: BCEWithLogits | Adam", fontsize=8.0)
    draw_box(ax, *boxes["baseline"], "Train Baseline Model", "Train on normalized samples only")
    draw_box(ax, *boxes["defense"], "Train Defense Model", "Adversarial Training (FGSM)\nTrain on: clean + FGSM samples")
    draw_box(ax, *boxes["trained"], "Two Trained Models", "Baseline MLP\nAdv-trained MLP")
    draw_box(ax, *boxes["attacks"], "Adversarial Attacks (White-box)", "Generate x_adv on TEST data\nFGSM (single-step, projected, L∞, ε)\nPGD (multi-step, projected, L∞, ε)", fontsize=8.0)
    draw_box(ax, *boxes["validity"], "Validity Constraints (Tabular)", "Perturb continuous only\nKeep one-hot columns fixed\nClip continuous features to valid range", fontsize=8.0)
    draw_box(ax, *boxes["eval"], "Evaluation (for each ε)", "Run on BOTH models\nClean accuracy\nRobust accuracy (FGSM)\nRobust accuracy (PGD)", fontsize=8.0)
    draw_box(ax, *boxes["report"], "Report / Deliverables", "Table/plots: accuracy vs ε\nCompare baseline vs adv-trained\nKey takeaway: robustness gap", fontsize=8.0)

    draw_arrow(ax, (2.7, 7.6), (3.1, 7.6))
    draw_arrow(ax, (6.2, 7.6), (6.7, 7.6))
    draw_arrow(ax, (8.8, 7.6), (9.3, 7.6))

    draw_arrow(ax, (7.7, 6.9), (3.2, 6.0), label="train", label_xy=(6.2, 6.2))
    draw_arrow(ax, (7.8, 6.9), (7.8, 5.8), label="train", label_xy=(7.9, 6.2))
    draw_arrow(ax, (8.0, 6.9), (1.8, 2.5), label="test set", label_xy=(4.5, 4.7))

    draw_arrow(ax, (4.6, 5.3), (10.0, 5.3))
    draw_arrow(ax, (9.5, 5.0), (10.0, 5.2))
    draw_arrow(ax, (12.8, 4.6), (10.8, 2.4))
    draw_arrow(ax, (3.9, 1.6), (4.5, 1.6))
    draw_arrow(ax, (8.1, 1.6), (9.3, 1.6))
    draw_arrow(ax, (12.3, 1.6), (12.8, 1.6))
    draw_arrow(ax, (4.0, 2.0), (6.0, 4.3), dashed=True, label="FGSM samples", label_xy=(4.6, 2.9))

    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
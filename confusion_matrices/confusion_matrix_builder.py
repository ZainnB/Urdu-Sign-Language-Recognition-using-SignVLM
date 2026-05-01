import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

LABEL_MAP_FILE = Path(__file__).resolve().parent.parent / "dataset_split_text_files" / "label_map.json"


# ──────────────────────────────────────────────
# Custom diverging colourmap:
#   off-diagonal (wrong)  → shades of red
#   diagonal   (correct)  → shades of blue
#   zero cells            → near-black background
# ──────────────────────────────────────────────
def build_confusion_cmap():
    """
    A perceptually-tuned colourmap for confusion matrices.
    Low values (misclassifications) go from dark-background → deep red → bright red.
    High values (correct) go through blue-teal up to ice-white.
    We achieve this by building two separate maps and combining them.
    """
    # Wrong: 0 → dark bg → brick red → vivid red
    wrong_colors = [
        (0.08, 0.08, 0.12),   # near-black background
        (0.35, 0.05, 0.05),   # dark blood red
        (0.75, 0.10, 0.10),   # medium red
        (0.95, 0.30, 0.15),   # bright orange-red
    ]
    # Correct (diagonal override): deep blue → electric blue → ice white
    correct_colors = [
        (0.02, 0.10, 0.35),   # deep navy
        (0.05, 0.35, 0.75),   # royal blue
        (0.10, 0.65, 0.95),   # sky blue
        (0.80, 0.95, 1.00),   # ice white
    ]
    # Main colourmap used for the full matrix (reds for off-diag)
    wrong_cmap = LinearSegmentedColormap.from_list("wrong_cmap", wrong_colors, N=512)
    correct_cmap = LinearSegmentedColormap.from_list("correct_cmap", correct_colors, N=512)
    return wrong_cmap, correct_cmap


def load_labels_from_tsv(tsv_path: Path, n_classes: int):
    """
    Build an index->label-name list from a dataset TSV file.

    The TSV is expected to contain lines like:
        relative/path/to/classname/file.mp4\t<label_index>
    We extract the class name from the parent directory of the file path and map
    it to the integer label index. If the TSV is missing or an index is not
    present in the TSV, the numeric string for that index is used as fallback.
    """
    if not tsv_path.is_file():
        return None
    mapping = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            path_part = parts[0]
            idx_part = parts[-1]
            try:
                idx = int(idx_part)
            except ValueError:
                continue
            # class name is parent directory of the file path
            try:
                class_name = Path(path_part).parent.name
            except Exception:
                class_name = str(idx)
            if class_name:
                mapping.setdefault(idx, class_name)
    return [mapping.get(i, str(i)) for i in range(n_classes)]


def calculate_per_class_accuracy(cm):
    row_sums = np.sum(cm, axis=1)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return np.diag(cm) / row_sums


def normalize_cm(cm):
    """Row-normalise: each cell = fraction of true-class samples."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cm.astype(float) / row_sums


def plot_confusion_matrix(cm, title, filename, labels=None):
    n = cm.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]

    # ── Normalise ─────────────────────────────────────────────────────────
    cm_norm = normalize_cm(cm)
    per_class_acc = calculate_per_class_accuracy(cm)
    overall_acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

    wrong_cmap, correct_cmap = build_confusion_cmap()

    # ── Build RGBA image manually ─────────────────────────────────────────
    # Off-diagonal: wrong_cmap,  diagonal: correct_cmap
    rgba = np.zeros((n, n, 4))
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if i == j:
                rgba[i, j] = correct_cmap(v)
            else:
                rgba[i, j] = wrong_cmap(v)

    # ── Figure layout ──────────────────────────────────────────────────────
    # Main grid: [heatmap | accuracy bar] stacked with title + summary strip
    fig = plt.figure(figsize=(26, 24), facecolor="#0D0D14")

    gs_outer = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[0.06, 0.82, 0.12],
        hspace=0.04,
    )

    # ── Title strip ───────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs_outer[0])
    ax_title.set_facecolor("#0D0D14")
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5, title,
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=22, fontweight="bold",
        color="#E8EAF6",
        fontfamily="monospace",
    )
    ax_title.text(
        0.5, -0.15,
        f"Overall accuracy: {overall_acc:.2%}   |   {n} classes   |   "
        f"{int(cm.sum()):,} samples",
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=11, color="#7986CB",
        fontfamily="monospace",
    )

    # ── Inner grid: heatmap + side bar ────────────────────────────────────
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs_outer[1],
        width_ratios=[0.88, 0.12],
        wspace=0.02,
    )

    ax_cm   = fig.add_subplot(gs_inner[0])
    ax_bar  = fig.add_subplot(gs_inner[1])

    # ── Draw heatmap ──────────────────────────────────────────────────────
    ax_cm.imshow(rgba, aspect="auto", interpolation="nearest")

    # Grid lines (subtle)
    ax_cm.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_cm.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_cm.grid(which="minor", color="#1C1C28", linewidth=0.25)
    ax_cm.tick_params(which="minor", bottom=False, left=False)

    # All-label ticks (every label shown, small font)
    tick_positions = np.arange(n)
    ax_cm.set_xticks(tick_positions)
    ax_cm.set_yticks(tick_positions)
    ax_cm.set_xticklabels(
        labels, rotation=90, fontsize=4.5,
        color="#B0BEC5", fontfamily="monospace",
    )
    ax_cm.set_yticklabels(
        labels, fontsize=4.5,
        color="#B0BEC5", fontfamily="monospace",
    )
    ax_cm.tick_params(axis="both", which="major", length=0, pad=1.5)

    ax_cm.set_xlabel("Predicted Label", fontsize=11, color="#7986CB",
                      fontfamily="monospace", labelpad=8)
    ax_cm.set_ylabel("True Label", fontsize=11, color="#7986CB",
                      fontfamily="monospace", labelpad=8)
    ax_cm.set_facecolor("#0D0D14")

    # Diagonal highlight border
    for i in range(n):
        rect = mpatches.FancyBboxPatch(
            (i - 0.5, i - 0.5), 1, 1,
            linewidth=0.0,
            boxstyle="square,pad=0",
            facecolor="none",
        )
        ax_cm.add_patch(rect)

    # ── Per-class accuracy horizontal bar chart ───────────────────────────
    y_positions = np.arange(n)
    bar_colors = [
        correct_cmap(v) if v >= 0.5 else wrong_cmap(v)
        for v in per_class_acc
    ]
    ax_bar.barh(
        y_positions, per_class_acc,
        color=bar_colors, height=0.85,
        edgecolor="none",
    )
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_ylim(-0.5, n - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 0.5, 1.0])
    ax_bar.set_xticklabels(["0", ".5", "1"], fontsize=6, color="#7986CB",
                            fontfamily="monospace")
    ax_bar.set_xlabel("Accuracy", fontsize=8, color="#7986CB",
                      fontfamily="monospace", labelpad=6)
    ax_bar.set_facecolor("#0D0D14")
    ax_bar.spines[:].set_color("#2C2C3E")
    ax_bar.tick_params(axis="x", colors="#2C2C3E", length=2)

    # Mean accuracy line
    mean_acc = per_class_acc.mean()
    ax_bar.axvline(mean_acc, color="#FFD54F", linewidth=0.8,
                   linestyle="--", alpha=0.8)

    # ── Colour-scale legend strip ─────────────────────────────────────────
    ax_legend = fig.add_subplot(gs_outer[2])
    ax_legend.set_facecolor("#0D0D14")
    ax_legend.axis("off")

    # Wrong scale
    grad_w = np.linspace(0, 1, 256).reshape(1, -1)
    wrong_rgba = np.array([wrong_cmap(v) for v in grad_w[0]]).reshape(1, 256, 4)
    ax_legend.imshow(
        wrong_rgba, aspect="auto",
        extent=[0.02, 0.38, 0.2, 0.8],
        transform=ax_legend.transAxes,
    )
    ax_legend.text(0.02, 0.10, "0", transform=ax_legend.transAxes,
                   ha="left", va="top", fontsize=8, color="#EF9A9A",
                   fontfamily="monospace")
    ax_legend.text(0.38, 0.10, "1", transform=ax_legend.transAxes,
                   ha="right", va="top", fontsize=8, color="#EF9A9A",
                   fontfamily="monospace")
    ax_legend.text(0.20, 0.96, "Misclassification rate",
                   transform=ax_legend.transAxes,
                   ha="center", va="top", fontsize=9, color="#EF9A9A",
                   fontfamily="monospace")

    # Correct scale
    grad_c = np.linspace(0, 1, 256).reshape(1, -1)
    correct_rgba = np.array([correct_cmap(v) for v in grad_c[0]]).reshape(1, 256, 4)
    ax_legend.imshow(
        correct_rgba, aspect="auto",
        extent=[0.62, 0.98, 0.2, 0.8],
        transform=ax_legend.transAxes,
    )
    ax_legend.text(0.62, 0.10, "0", transform=ax_legend.transAxes,
                   ha="left", va="top", fontsize=8, color="#90CAF9",
                   fontfamily="monospace")
    ax_legend.text(0.98, 0.10, "1", transform=ax_legend.transAxes,
                   ha="right", va="top", fontsize=8, color="#90CAF9",
                   fontfamily="monospace")
    ax_legend.text(0.80, 0.96, "Correct classification rate",
                   transform=ax_legend.transAxes,
                   ha="center", va="top", fontsize=9, color="#90CAF9",
                   fontfamily="monospace")

    ax_legend.text(0.50, 0.55, "◆",
                   transform=ax_legend.transAxes,
                   ha="center", va="center", fontsize=14, color="#FFD54F",
                   fontfamily="monospace")
    ax_legend.text(0.50, 0.10, "mean class accuracy",
                   transform=ax_legend.transAxes,
                   ha="center", va="top", fontsize=8, color="#FFD54F",
                   fontfamily="monospace")

    # ── Save ──────────────────────────────────────────────────────────────
    plt.savefig(filename, dpi=220, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {filename}")


def print_per_class_accuracies(acc, labels, title):
    print(f"\n{title}")
    for i, value in enumerate(acc):
        label_name = labels[i] if i < len(labels) else str(i)
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {i:>3}  {label_name:<35}  {bar}  {value:.4f}")


def main():
    # ── Load matrices ─────────────────────────────────────────────────────
    train_cm = np.load("train_confusion_matrix.npy")
    val_cm   = np.load("validation_confusion_matrix.npy")
    test_cm  = np.load("test_confusion_matrix.npy")

    # Load labels per-split from TSV files (fallback to numeric labels)
    splits_dir = Path(__file__).resolve().parent.parent / "dataset_split_text_files"
    train_tsv = splits_dir / "train.tsv"
    val_tsv = splits_dir / "val.tsv"
    test_tsv = splits_dir / "test.tsv"

    labels_train = load_labels_from_tsv(train_tsv, train_cm.shape[0]) if train_tsv.is_file() else None
    if labels_train is None:
        print(f"train.tsv not found at {train_tsv}. Using numeric labels for training.")
        labels_train = [str(i) for i in range(train_cm.shape[0])]

    labels_val = load_labels_from_tsv(val_tsv, val_cm.shape[0]) if val_tsv.is_file() else None
    if labels_val is None:
        print(f"val.tsv not found at {val_tsv}. Using numeric labels for validation.")
        labels_val = [str(i) for i in range(val_cm.shape[0])]

    labels_test = load_labels_from_tsv(test_tsv, test_cm.shape[0]) if test_tsv.is_file() else None
    if labels_test is None:
        print(f"test.tsv not found at {test_tsv}. Using numeric labels for test.")
        labels_test = [str(i) for i in range(test_cm.shape[0])]

    print(f"Train CM shape : {train_cm.shape}")
    print(f"Val   CM shape : {val_cm.shape}")
    print(f"Test  CM shape : {test_cm.shape}")

    # ── Accuracies ────────────────────────────────────────────────────────
    train_acc = calculate_per_class_accuracy(train_cm)
    val_acc   = calculate_per_class_accuracy(val_cm)
    test_acc  = calculate_per_class_accuracy(test_cm)

    print_per_class_accuracies(train_acc, labels_train, "Train Per-Class Accuracies:")
    print_per_class_accuracies(val_acc,   labels_val, "Validation Per-Class Accuracies:")
    print_per_class_accuracies(test_acc,  labels_test, "Test Per-Class Accuracies:")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nRendering confusion matrices …")
    plot_confusion_matrix(train_cm, "Train Confusion Matrix",
                          "train_confusion_matrix.png",      labels=labels_train)
    plot_confusion_matrix(val_cm,   "Validation Confusion Matrix",
                          "validation_confusion_matrix.png", labels=labels_val)
    plot_confusion_matrix(test_cm,  "Test Confusion Matrix",
                          "test_confusion_matrix.png",       labels=labels_test)


if __name__ == "__main__":
    main()
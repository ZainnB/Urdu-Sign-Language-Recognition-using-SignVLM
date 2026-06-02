"""
compute_metrics.py
──────────────────
Loads train / validation / test confusion matrices (.npy) and computes:
  • Per-class  TP, TN, FP, FN  (one-vs-rest 2×2 breakdown)
  • Per-class  Precision, Recall, F1-score
  • Macro / Weighted averages
  • A rich PNG report per split  (metrics heatmap + bar charts + 2×2 grids)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR = Path(__file__).resolve().parent.parent / "dataset_split_text_files"


# ════════════════════════════════════════════════════════════════════════════
# Data helpers
# ════════════════════════════════════════════════════════════════════════════

def load_labels_from_tsv(tsv_path: Path, n_classes: int):
    """
    Build an index → label-name list from a dataset TSV file.

    Each line is expected to be:
        relative/path/to/ClassName/filename.mp4 <TAB> label_index

    The class name is taken from the *parent directory* of the file path
    (the folder directly containing the video), which matches how the dataset
    is structured:  <Split>/<ClassName>/<stem>.mp4

    Returns a list of length n_classes, or None if the file is missing.
    """
    if not tsv_path.is_file():
        return None
    mapping: dict = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()   # fallback: whitespace split
            if len(parts) < 2:
                continue
            path_part = parts[0]
            try:
                idx = int(parts[-1])
            except ValueError:
                continue
            try:
                class_name = Path(path_part).parent.name
            except Exception:
                class_name = str(idx)
            if class_name:
                mapping.setdefault(idx, class_name)
    return [mapping.get(i, str(i)) for i in range(n_classes)]


def compute_per_class_metrics(cm: np.ndarray):
    """
    For each class c (one-vs-rest):
        TP  = cm[c, c]
        FP  = col_sum[c] - cm[c, c]   (predicted c but not c)
        FN  = row_sum[c] - cm[c, c]   (true c but not predicted c)
        TN  = total - TP - FP - FN
    Returns dicts keyed by class index.
    """
    n = cm.shape[0]
    total = cm.sum()

    TP = np.diag(cm).astype(float)
    FP = cm.sum(axis=0).astype(float) - TP   # col sum − diag
    FN = cm.sum(axis=1).astype(float) - TP   # row sum − diag
    TN = total - TP - FP - FN

    precision = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
    recall    = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
    f1        = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )

    support = cm.sum(axis=1)   # true samples per class

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "support":   support,
    }


def macro_avg(metrics):
    return {
        "precision": metrics["precision"].mean(),
        "recall":    metrics["recall"].mean(),
        "f1":        metrics["f1"].mean(),
    }


def weighted_avg(metrics):
    w = metrics["support"] / metrics["support"].sum()
    return {
        "precision": (metrics["precision"] * w).sum(),
        "recall":    (metrics["recall"]    * w).sum(),
        "f1":        (metrics["f1"]        * w).sum(),
    }


# ════════════════════════════════════════════════════════════════════════════
# Terminal printing
# ════════════════════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def bar(value, width=20, full="█", empty="░"):
    filled = int(round(value * width))
    return full * filled + empty * (width - filled)

def colour_f1(v):
    if v >= 0.80: return GREEN
    if v >= 0.50: return YELLOW
    return RED

def print_metrics_table(metrics, labels, split_name):
    n = len(labels)
    print(f"\n{BOLD}{CYAN}{'═'*90}{RESET}")
    print(f"{BOLD}{CYAN}  {split_name}  —  Per-Class Metrics{RESET}")
    print(f"{BOLD}{CYAN}{'═'*90}{RESET}")
    header = (
        f"  {'#':>3}  {'Label':<28}  "
        f"{'TP':>6}  {'FP':>6}  {'FN':>6}  {'TN':>8}  "
        f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Supp':>5}  Bar"
    )
    print(f"{BOLD}{header}{RESET}")
    print(f"  {'─'*87}")

    for i in range(n):
        lbl = labels[i][:28]
        tp  = int(metrics["TP"][i])
        fp  = int(metrics["FP"][i])
        fn  = int(metrics["FN"][i])
        tn  = int(metrics["TN"][i])
        p   = metrics["precision"][i]
        r   = metrics["recall"][i]
        f   = metrics["f1"][i]
        sup = int(metrics["support"][i])
        c   = colour_f1(f)
        print(
            f"  {i:>3}  {lbl:<28}  "
            f"{tp:>6}  {fp:>6}  {fn:>6}  {tn:>8}  "
            f"{p:>6.3f}  {r:>6.3f}  {c}{f:>6.3f}{RESET}  "
            f"{sup:>5}  {c}{bar(f)}{RESET}"
        )

    ma  = macro_avg(metrics)
    wa  = weighted_avg(metrics)
    print(f"\n  {'─'*87}")
    print(f"  {BOLD}{'Macro avg':<32}{RESET}  "
          f"  {'':>6}  {'':>6}  {'':>6}  {'':>8}  "
          f"{ma['precision']:>6.3f}  {ma['recall']:>6.3f}  "
          f"{BOLD}{GREEN}{ma['f1']:>6.3f}{RESET}")
    print(f"  {BOLD}{'Weighted avg':<32}{RESET}  "
          f"  {'':>6}  {'':>6}  {'':>6}  {'':>8}  "
          f"{wa['precision']:>6.3f}  {wa['recall']:>6.3f}  "
          f"{BOLD}{GREEN}{wa['f1']:>6.3f}{RESET}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Colourmaps
# ════════════════════════════════════════════════════════════════════════════

def _cmap(colors, n=256):
    return LinearSegmentedColormap.from_list("_", colors, N=n)

BLUE_CMAP = _cmap(["#0D0D14", "#0D2B6B", "#1565C0", "#42A5F5", "#E3F2FD"])
RED_CMAP  = _cmap(["#0D0D14", "#4A0000", "#B71C1C", "#EF5350", "#FFCDD2"])
F1_CMAP   = _cmap(["#B71C1C", "#E65100", "#F9A825", "#2E7D32", "#1B5E20"])


# ════════════════════════════════════════════════════════════════════════════
# PNG report
# ════════════════════════════════════════════════════════════════════════════

def plot_metrics_report(metrics, labels, split_name, filename):
    n     = len(labels)
    prec  = metrics["precision"]
    rec   = metrics["recall"]
    f1    = metrics["f1"]
    tp    = metrics["TP"]
    fp    = metrics["FP"]
    fn    = metrics["FN"]
    tn    = metrics["TN"]
    sup   = metrics["support"]

    ma = macro_avg(metrics)
    wa = weighted_avg(metrics)

    BG    = "#0D0D14"
    PANEL = "#13131F"
    GRID  = "#1C1C2C"
    TICK  = "#546E7A"
    HEAD  = "#E8EAF6"
    SUB   = "#7986CB"

    fig = plt.figure(figsize=(28, 32), facecolor=BG)

    gs_root = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[0.045, 0.30, 0.28, 0.35],
        hspace=0.10,
    )

    # ── Title ────────────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs_root[0])
    ax_t.set_facecolor(BG); ax_t.axis("off")
    ax_t.text(0.5, 0.60, f"{split_name}  —  Per-Class Classification Report",
              transform=ax_t.transAxes, ha="center", va="center",
              fontsize=20, fontweight="bold", color=HEAD, fontfamily="monospace")
    ax_t.text(
        0.5, -0.10,
        f"Macro F1: {ma['f1']:.4f}   Macro Prec: {ma['precision']:.4f}   "
        f"Macro Rec: {ma['recall']:.4f}   |   "
        f"Weighted F1: {wa['f1']:.4f}   Weighted Prec: {wa['precision']:.4f}   "
        f"Weighted Rec: {wa['recall']:.4f}   |   {n} classes",
        transform=ax_t.transAxes, ha="center", va="center",
        fontsize=10, color=SUB, fontfamily="monospace",
    )

    # ── Row 1: Precision / Recall / F1 bar charts ────────────────────────
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_root[1], wspace=0.06,
    )
    y = np.arange(n)

    def _hbar(ax, values, cmap, xlabel, mean_line_color):
        colors = [cmap(v) for v in values]
        ax.barh(y, values, color=colors, height=0.80, edgecolor="none")
        ax.axvline(values.mean(), color=mean_line_color,
                   linewidth=1.0, linestyle="--", alpha=0.85, zorder=5)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=3.8, color=TICK,
                           fontfamily="monospace")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"],
                            fontsize=7, color=TICK, fontfamily="monospace")
        ax.set_xlabel(xlabel, fontsize=10, color=SUB,
                      fontfamily="monospace", labelpad=6)
        ax.set_facecolor(PANEL)
        ax.spines[:].set_color(GRID)
        ax.tick_params(axis="both", length=2, colors=GRID)
        ax.text(values.mean() + 0.01, n * 0.02,
                f"μ={values.mean():.3f}", fontsize=7,
                color=mean_line_color, fontfamily="monospace", va="top")

    _hbar(fig.add_subplot(gs1[0]), prec, BLUE_CMAP, "Precision", "#90CAF9")
    _hbar(fig.add_subplot(gs1[1]), rec,  BLUE_CMAP, "Recall",    "#80DEEA")
    _hbar(fig.add_subplot(gs1[2]), f1,   F1_CMAP,   "F1-Score",  "#FFD54F")

    # ── Row 2: Heatmap of all three metrics side-by-side ─────────────────
    gs2 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_root[2], wspace=0.04,
    )

    def _heatmap(ax, values, cmap, title_str):
        mat = values.reshape(-1, 1)
        rgba = np.array([[cmap(v) for v in row] for row in mat])
        ax.imshow(rgba, aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=3.8, color=TICK,
                           fontfamily="monospace")
        ax.set_title(title_str, fontsize=10, color=HEAD,
                     fontfamily="monospace", pad=6)
        ax.set_facecolor(PANEL)
        ax.spines[:].set_color(GRID)
        ax.tick_params(length=0)

    _heatmap(fig.add_subplot(gs2[0]), prec, BLUE_CMAP, "Precision strip")
    _heatmap(fig.add_subplot(gs2[1]), rec,  BLUE_CMAP, "Recall strip")
    _heatmap(fig.add_subplot(gs2[2]), f1,   F1_CMAP,   "F1 strip")

    # ── Row 3: TP / FP / FN / TN stacked bar (OvR counts) ───────────────
    gs3 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_root[3], wspace=0.06,
    )

    # Left: TP / FP / FN stacked (TN omitted — it's huge and trivially
    # dominates; show as annotation instead)
    ax_stk = fig.add_subplot(gs3[0])
    w = 0.72
    ax_stk.barh(y, tp,      height=w, color="#1565C0", label="TP",
                edgecolor="none")
    ax_stk.barh(y, fp, left=tp,              height=w, color="#C62828",
                label="FP", edgecolor="none")
    ax_stk.barh(y, fn, left=tp + fp,         height=w, color="#E65100",
                label="FN", edgecolor="none")
    ax_stk.set_ylim(-0.5, n - 0.5)
    ax_stk.invert_yaxis()
    ax_stk.set_yticks(y)
    ax_stk.set_yticklabels(labels, fontsize=3.8, color=TICK,
                            fontfamily="monospace")
    ax_stk.set_xlabel("Sample count (TP + FP + FN, one-vs-rest)",
                       fontsize=9, color=SUB, fontfamily="monospace")
    ax_stk.set_facecolor(PANEL)
    ax_stk.spines[:].set_color(GRID)
    ax_stk.tick_params(axis="both", length=2, colors=GRID)
    ax_stk.xaxis.set_minor_locator(MultipleLocator(50))
    ax_stk.set_title("TP / FP / FN breakdown (OvR)", fontsize=10,
                      color=HEAD, fontfamily="monospace", pad=6)
    ax_stk.legend(
        handles=[
            mpatches.Patch(color="#1565C0", label="TP — True Positive"),
            mpatches.Patch(color="#C62828", label="FP — False Positive"),
            mpatches.Patch(color="#E65100", label="FN — False Negative"),
        ],
        loc="lower right", fontsize=7,
        facecolor=PANEL, edgecolor=GRID, labelcolor=HEAD,
    )

    # Right: 2×2 summary table per class rendered as a colour grid
    # Columns: TP | FP | FN | TN  (log-scaled for visibility)
    ax_2x2 = fig.add_subplot(gs3[1])

    # Stack into (n, 4) matrix
    mat4 = np.column_stack([tp, fp, fn, tn])
    # Log-scale each column independently for colour mapping
    col_labels_4 = ["TP", "FP", "FN", "TN"]
    col_cmaps    = [BLUE_CMAP, RED_CMAP, RED_CMAP, BLUE_CMAP]

    rgba4 = np.zeros((n, 4, 4))
    for c in range(4):
        col = mat4[:, c].astype(float)
        vmax = col.max() if col.max() > 0 else 1.0
        normed = col / vmax
        for row in range(n):
            rgba4[row, c] = col_cmaps[c](normed[row])

    ax_2x2.imshow(rgba4, aspect="auto", interpolation="nearest")
    ax_2x2.set_xticks([0, 1, 2, 3])
    ax_2x2.set_xticklabels(col_labels_4, fontsize=9, color=HEAD,
                            fontfamily="monospace", fontweight="bold")
    ax_2x2.set_yticks(y)
    ax_2x2.set_yticklabels(labels, fontsize=3.8, color=TICK,
                            fontfamily="monospace")
    ax_2x2.set_title("2×2 OvR matrix per class  (colour = relative magnitude)",
                      fontsize=10, color=HEAD, fontfamily="monospace", pad=6)
    ax_2x2.spines[:].set_color(GRID)
    ax_2x2.tick_params(length=0)
    ax_2x2.set_facecolor(PANEL)

    # Column header colour-band
    for xi, (lbl, cm_) in enumerate(zip(col_labels_4, col_cmaps)):
        ax_2x2.axvline(xi - 0.5, color=GRID, linewidth=0.5)

    # ── Save ─────────────────────────────────────────────────────────────
    plt.savefig(filename, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {filename}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def process_split(cm: np.ndarray, labels: list, split_name: str, png_name: str):
    print(f"\n{'━'*90}")
    print(f"  Processing: {split_name}   shape={cm.shape}   "
          f"total samples={int(cm.sum()):,}")
    print(f"{'━'*90}")

    metrics = compute_per_class_metrics(cm)
    print_metrics_table(metrics, labels, split_name)
    plot_metrics_report(metrics, labels, split_name, png_name)
    return metrics


def main():
    # Load matrices
    # train_cm = np.load("train_confusion_matrix.npy")
    # val_cm   = np.load("validation_confusion_matrix.npy")
    test_cm  = np.load("test_confusion_matrix.npy")

    # Load per-split label names from TSV files; fall back to numeric strings
    splits = [
        #(train_cm, "train.tsv", "Train",      "train_metrics_report.png"),
        # (val_cm,   "val.tsv",   "Validation", "validation_metrics_report.png"),
        (test_cm,  "test.tsv",  "Test",       "test_metrics_report.png"),
    ]

    for cm, tsv_name, split_name, png_name in splits:
        tsv_path = SPLITS_DIR / tsv_name
        labels = load_labels_from_tsv(tsv_path, cm.shape[0])
        if labels is None:
            print(f"[warn] {tsv_name} not found at {tsv_path}. Using numeric labels.")
            labels = [str(i) for i in range(cm.shape[0])]
        process_split(cm, labels, split_name, png_name)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
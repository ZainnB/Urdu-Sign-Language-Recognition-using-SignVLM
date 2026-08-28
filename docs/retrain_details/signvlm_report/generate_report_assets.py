"""Generate all graphs + tables for the SignVLM signer-disjoint retrain report.

Driven entirely by the logged artifacts in docs/retrain_details/signVLM_lists/.
Confusion matrices are rendered with the project's own
confusion_matrices/confusion_matrix_builder.py.

Run from anywhere:  python generate_report_assets.py
Outputs land next to this script.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Class names include Urdu letters; give the builder's monospace family an
# Arabic-capable fallback so tick labels don't render as boxes.
plt.rcParams["font.monospace"] = ["DejaVu Sans Mono", "Tahoma", "Arial"]
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Tahoma", "Arial"]

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LISTS = HERE.parent / "signVLM_lists"

sys.path.insert(0, str(REPO / "confusion_matrices"))
from confusion_matrix_builder import plot_confusion_matrix  # noqa: E402

# Two-series palette (train vs validation), colorblind-safe pair.
C_TRAIN = "#1f77b4"
C_VAL = "#ff7f0e"
C_TEST = "#2ca02c"

plt.rcParams.update({
    "figure.figsize": (10, 7.5),
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.titlesize": 14,
})


VAL_EVERY = 5  # validation cadence; ad-hoc val logs off this grid are dropped


def load_history():
    with open(LISTS / "signvlm_loss_history.json", encoding="utf-8") as f:
        hist = json.load(f)
    # Validation is scheduled every 5th epoch; the run also logged ad-hoc
    # evals at epochs 1-3, which the report intentionally omits.
    for i, ep in enumerate(hist["epoch"]):
        if ep % VAL_EVERY:
            hist["val_loss"][i] = hist["val_acc1"][i] = hist["val_acc5"][i] = None
    return hist


def line_plot(x, series, title, ylabel, filename, percent=False):
    """series: list of (values, label, color). None values are skipped per-series."""
    fig, ax = plt.subplots()
    for values, label, color in series:
        pts = [(xi, v) for xi, v in zip(x, values) if v is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        if percent:
            ys = [v * 100 for v in ys]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8, label=label, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {filename}")


def step_loss_plot(filename):
    steps, losses = [], []
    with open(LISTS / "signvlm_step_log.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["loss"]:
                steps.append(int(row["global_step"]))
                losses.append(float(row["loss"]))
    steps = np.array(steps)
    losses = np.array(losses)
    # 218 steps/epoch -> smooth over roughly half an epoch
    win = 109
    kernel = np.ones(win) / win
    smooth = np.convolve(losses, kernel, mode="valid")
    fig, ax = plt.subplots()
    ax.plot(steps, losses, color=C_TRAIN, alpha=0.18, linewidth=0.6, label="per-step loss")
    ax.plot(steps[win - 1:], smooth, color=C_TRAIN, linewidth=1.8,
            label=f"moving average ({win} steps)")
    ax.set_xlabel("global step (218 steps/epoch)")
    ax.set_ylabel("training loss")
    ax.set_title("Step-level training loss (10,028 steps, 46 epochs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {filename}")


def lr_plot(hist, filename):
    fig, ax = plt.subplots()
    ax.plot(hist["epoch"], hist["lr"], marker="o", markersize=4,
            linewidth=1.8, color=C_TRAIN)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title("Cosine-annealed learning rate (4e-5 → ~1e-8)")
    fig.tight_layout()
    fig.savefig(HERE / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {filename}")


def read_final_metrics():
    rows = {}
    for fname in ["signvlm_final_metrics.csv", "signvlm_full_metrics_test.csv"]:
        with open(LISTS / fname, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["split"]] = row
    return rows


def split_accuracy_bar(final, filename):
    # test (same-signer pool) accuracy comes from its confusion matrix
    test_cm = np.load(LISTS / "test_confusion_matrix.npy")
    test_same = np.trace(test_cm) / test_cm.sum()
    labels = ["Train\n(n=4,368)", "Validation\n(n=1,244)",
              "Test same-pool\n(n=1,248)", "Test diff-signer\n(n=1,248)"]
    values = [float(final["train"]["accuracy"]) * 100,
              float(final["validation"]["accuracy"]) * 100,
              test_same * 100,
              float(final["test_diff_signer"]["accuracy"]) * 100]
    colors = [C_TRAIN, C_VAL, "#9467bd", C_TEST]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f}%",
                ha="center", fontsize=11)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Final top-1 accuracy per split (eval mode, multi-view) — best model")
    fig.tight_layout()
    fig.savefig(HERE / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {filename}")


def macro_metrics_bar(final, filename):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    splits = [("train", "Train", C_TRAIN),
              ("validation", "Validation", C_VAL),
              ("test_diff_signer", "Test (diff signer)", C_TEST)]
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for i, (key, label, color) in enumerate(splits):
        vals = [float(final[key][m]) for m in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                    ha="center", fontsize=8.5)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.09)
    ax.set_title("Final metrics: train vs validation vs signer-disjoint test — best model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {filename}")


def confusion_matrices():
    with open(LISTS / "label_map_auto.json", encoding="utf-8") as f:
        lm = json.load(f)
    labels = [lm[str(i)] for i in range(len(lm))]
    jobs = [
        ("train_confusion_matrix.npy", "SignVLM Train Confusion Matrix (n=4,368)",
         "signvlm_train_confusion_matrix.png"),
        ("validation_confusion_matrix.npy", "SignVLM Validation Confusion Matrix (n=1,244)",
         "signvlm_validation_confusion_matrix.png"),
        ("test_confusion_matrix.npy", "SignVLM Test Confusion Matrix - same signer pool (n=1,248)",
         "signvlm_test_confusion_matrix.png"),
        ("test_diff_signer_confusion_matrix.npy",
         "SignVLM Test Confusion Matrix - different signer (n=1,248)",
         "signvlm_test_diff_signer_confusion_matrix.png"),
    ]
    for src, title, out in jobs:
        cm = np.load(LISTS / src)
        plot_confusion_matrix(cm, title, str(HERE / out), labels=labels)


def per_epoch_markdown(hist, out_name):
    """Emit a per-epoch metrics table as markdown for the report."""
    lines = [
        "| epoch | lr | train_loss | train_acc1 | train_acc5 | val_loss | val_acc1 | val_acc5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    fmt = lambda v, pct=False: ("" if v is None else (f"{v*100:.2f}%" if pct else f"{v:.4f}"))
    for i, ep in enumerate(hist["epoch"]):
        lines.append(
            f"| {ep} | {hist['lr'][i]:.2e} | {fmt(hist['train_loss'][i])} | "
            f"{fmt(hist['train_acc1'][i], True)} | {fmt(hist['train_acc5'][i], True)} | "
            f"{fmt(hist['val_loss'][i])} | {fmt(hist['val_acc1'][i], True)} | "
            f"{fmt(hist['val_acc5'][i], True)} |"
        )
    (HERE / out_name).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved -> {out_name}")


def main():
    hist = load_history()
    final = read_final_metrics()

    print("Rendering training curves ...")
    line_plot(hist["epoch"],
              [(hist["train_loss"], "train", C_TRAIN), (hist["val_loss"], "validation", C_VAL)],
              "Loss per epoch", "loss", "signvlm_loss_per_epoch.png")
    line_plot(hist["epoch"],
              [(hist["train_acc1"], "train (aug, train-mode)", C_TRAIN),
               (hist["val_acc1"], "validation (eval-mode)", C_VAL)],
              "Top-1 accuracy per epoch", "top-1 accuracy (%)",
              "signvlm_top1_accuracy_per_epoch.png", percent=True)
    line_plot(hist["epoch"],
              [(hist["train_acc5"], "train (aug, train-mode)", C_TRAIN),
               (hist["val_acc5"], "validation (eval-mode)", C_VAL)],
              "Top-5 accuracy per epoch", "top-5 accuracy (%)",
              "signvlm_top5_accuracy_per_epoch.png", percent=True)
    lr_plot(hist, "signvlm_lr_schedule.png")
    step_loss_plot("signvlm_step_loss.png")

    print("Rendering summary bars ...")
    split_accuracy_bar(final, "signvlm_split_accuracy.png")
    macro_metrics_bar(final, "signvlm_final_metrics_bars.png")

    print("Rendering confusion matrices ...")
    confusion_matrices()

    per_epoch_markdown(hist, "per_epoch_table.md")
    print("Done.")


if __name__ == "__main__":
    main()

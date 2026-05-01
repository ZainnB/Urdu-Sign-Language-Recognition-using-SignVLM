from pathlib import Path
import importlib.util
import numpy as np
import sys


def load_confusion_builder_module():
    """Dynamically load the existing confusion_matrix_builder module from this folder."""
    base = Path(__file__).resolve().parent
    module_path = base / "confusion_matrix_builder.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"{module_path} not found")
    spec = importlib.util.spec_from_file_location("confusion_matrix_builder", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_unseen_tsv(tsv_path: Path):
    """Parse `unseen.tsv` and return sorted unique label indices found in the file.

    Each line is expected to contain the filename and the numeric label (tab or whitespace separated).
    """
    labels = set()
    with open(tsv_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # Try common separators; label should be the last token
            parts = ln.split()
            try:
                idx = int(parts[-1])
            except Exception:
                parts_tab = ln.split('\t')
                try:
                    idx = int(parts_tab[-1])
                except Exception:
                    continue
            labels.add(idx)
    return sorted(labels)


def main():
    base = Path(__file__).resolve().parent
    unseen_cm_path = base / "unseen_confusion_matrix.npy"
    unseen_tsv = base / "unseen.tsv"
    out_png = base / "unseen_confusion_matrix.png"

    if not unseen_cm_path.is_file():
        print(f"Error: {unseen_cm_path} not found.")
        sys.exit(1)
    if not unseen_tsv.is_file():
        print(f"Error: {unseen_tsv} not found.")
        sys.exit(1)

    cm_builder = load_confusion_builder_module()

    # Parse label indices present in the unseen split
    label_indices = parse_unseen_tsv(unseen_tsv)
    if len(label_indices) == 0:
        print("No label indices found in unseen.tsv; aborting.")
        sys.exit(1)

    # Load full confusion matrix and extract submatrix for these labels
    cm_full = np.load(unseen_cm_path, allow_pickle=False)
    n_all = cm_full.shape[0]
    label_indices = [i for i in label_indices if 0 <= i < n_all]
    if len(label_indices) == 0:
        print("No valid label indices found within range of the confusion matrix; aborting.")
        sys.exit(1)

    cm_sub = cm_full[np.ix_(label_indices, label_indices)]

    # Build human-readable label names using the same label map loader
    try:
        labels_full = cm_builder.load_label_names(cm_builder.LABEL_MAP_FILE)
    except Exception:
        labels_full = None

    if labels_full is None:
        labels = [str(i) for i in label_indices]
    else:
        labels = [labels_full[i] if i < len(labels_full) else str(i) for i in label_indices]

    # Plot using the same plotting routine from confusion_matrix_builder
    cm_builder.plot_confusion_matrix(cm_sub, "Unseen Confusion Matrix", str(out_png), labels=labels)

    # Print per-class accuracies for the unseen subset
    per_class_acc = cm_builder.calculate_per_class_accuracy(cm_sub)
    cm_builder.print_per_class_accuracies(per_class_acc, labels, "Unseen Per-Class Accuracies:")


if __name__ == "__main__":
    main()

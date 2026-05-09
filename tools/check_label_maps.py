#!/usr/bin/env python3
import re
from pathlib import Path


def load_label_map_local(path):
    id_to_name = {}
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Colon-separated
            if ':' in line:
                left, right = line.split(':', 1)
                left, right = left.strip(), right.strip()
                if left.isdigit():
                    id_to_name[int(left)] = right
                    continue
                if right.isdigit():
                    id_to_name[int(right)] = left
                    continue
                raise ValueError(f"Unrecognized label map line: {line!r}")

            # Whitespace / TSV style
            parts = line.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                idx = int(parts[-1])
                left_part = " ".join(parts[:-1])
                if "/" in left_part:
                    label = left_part.split("/", 1)[0].strip()
                else:
                    label = left_part
                    if "." in label:
                        label = label.split(".")[0]
                id_to_name[idx] = label
                continue

            # Fallback regex
            m = re.search(r'(.+?)\s+(\d+)$', line)
            if m:
                left_part = m.group(1)
                idx = int(m.group(2))
                if "/" in left_part:
                    label = left_part.split("/", 1)[0].strip()
                else:
                    label = left_part
                    if "." in label:
                        label = label.split(".")[0]
                id_to_name[idx] = label
                continue

            # skip unknown
    return id_to_name


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    psl_path = root / "trained_models_final" / "PSL_recognition_label_map.txt"
    tsv_path = root / "confusion_matrices" / "test.tsv"

    psl = load_label_map_local(psl_path)
    tsv = load_label_map_local(tsv_path)

    print("PSL map entries:", len(psl))
    print("PSL sample:", list(psl.items())[:8])
    print("TSV map entries:", len(tsv))
    print("TSV sample:", list(tsv.items())[:12])

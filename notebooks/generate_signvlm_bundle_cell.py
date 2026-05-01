"""Build an optional source zip for offline / manual copy (no notebook mutation).

The Colab notebook uses **git clone** in Cell 2b (`SIGNVLM_GIT_URL`); you normally do not need
a base64-in-notebook bundle. This script only writes a `.zip` next to this file for ad-hoc use.

Run from repo root:  python notebooks/generate_signvlm_bundle_cell.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "signvlm_sources_for_offline.zip"

FILES = [
    "checkpoint.py",
    "main.py",
    "model.py",
    "vision_transformer.py",
    "weight_loaders.py",
    "video_dataset/__init__.py",
    "video_dataset/dataloader.py",
    "video_dataset/dataset.py",
    "video_dataset/drive_to_local_cache.py",
    "video_dataset/transform.py",
    "video_dataset/rand_augment.py",
    "video_dataset/random_erasing.py",
]


def main():
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in FILES:
            p = ROOT / rel
            if not p.is_file():
                raise FileNotFoundError(f"Missing source file: {p}")
            zf.write(p, rel)
    n = OUT.stat().st_size
    print(f"Wrote {OUT} ({n} bytes). Unzip to /content and add to sys.path if needed; Colab Cell 2b uses git by default.")


if __name__ == "__main__":
    main()

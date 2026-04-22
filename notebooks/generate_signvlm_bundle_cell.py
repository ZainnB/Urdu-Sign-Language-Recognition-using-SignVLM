"""Regenerate Cell 2b in SignVLM_Colab_Training.ipynb from repo sources (run from repo root)."""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = Path(__file__).resolve().parent / "SignVLM_Colab_Training.ipynb"

FILES = [
    "checkpoint.py",
    "main.py",
    "model.py",
    "vision_transformer.py",
    "weight_loaders.py",
    "video_dataset/__init__.py",
    "video_dataset/dataloader.py",
    "video_dataset/dataset.py",
    "video_dataset/transform.py",
    "video_dataset/rand_augment.py",
    "video_dataset/random_erasing.py",
]

HEADER = """# Cell 2b: Extract bundled SignVLM Python sources into Colab (run after Cell 1; no repo copy on Drive needed for `import main` / `video_dataset`)
import base64
import io
import os
import zipfile

CODE_ROOT = globals().get("CODE_ROOT", "/content/signvlm_bundle")
REPO_ROOT = globals().get("REPO_ROOT", "")

# Alternative to bundle extraction: use repo sources directly when available.
_repo_main = os.path.join(REPO_ROOT, "main.py") if REPO_ROOT else ""
if _repo_main and os.path.isfile(_repo_main):
    CODE_ROOT = REPO_ROOT
    print("Using SignVLM sources from REPO_ROOT:", CODE_ROOT)
else:
    _ZIP_B64 = (
"""

FOOTER = """
)

    _buf = io.BytesIO(base64.b64decode(_ZIP_B64.encode("ascii")))
    os.makedirs(CODE_ROOT, exist_ok=True)
    with zipfile.ZipFile(_buf) as z:
        z.extractall(CODE_ROOT)
    print("Extracted SignVLM sources to", CODE_ROOT)
"""


def main():
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in FILES:
            p = ROOT / rel
            zf.write(p, rel)
    raw = base64.b64encode(bio.getvalue()).decode("ascii")
    # Chunk for readable notebook lines (~96 chars quoted per line)
    chunk = 96
    lines = [raw[i : i + chunk] for i in range(0, len(raw), chunk)]
    quoted = "\n".join(f'    "{line}"' for line in lines)

    cell_src = HEADER + quoted + FOOTER
    if not cell_src.endswith("\n"):
        cell_src += "\n"
    source_lines = cell_src.splitlines(keepends=True)

    nb = json.loads(NB.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        src = cell.get("source", [])
        first = src[0] if src else ""
        if isinstance(first, str) and first.startswith("# Cell 2b:"):
            nb["cells"][i]["source"] = source_lines
            if "outputs" in nb["cells"][i]:
                nb["cells"][i]["outputs"] = []
            nb["cells"][i]["execution_count"] = None
            break
    else:
        raise SystemExit("Could not find Cell 2b in notebook")

    NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {NB} ({len(raw)} b64 chars, zip {len(bio.getvalue())} bytes)")


if __name__ == "__main__":
    main()

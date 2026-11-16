r"""Download the trained models from the GitHub release into results/run3/ (needed to run the app).

Run: python scripts\download_models.py
Uses only the Python standard library.
"""
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/sfthanif/plant-disease/releases/download/v1.0"
FILES = ["small_cnn_best.pth", "mobilenetv3_small_best.pth", "efficientnet_b0_best.pth", "ood_bank.pt"]
DEST = Path(__file__).resolve().parent.parent / "results" / "run3"

DEST.mkdir(parents=True, exist_ok=True)
for name in FILES:
    out = DEST / name
    if out.exists():
        print(f"already present: {name}")
        continue
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(f"{RELEASE}/{name}", out)
print(f"done, models are in {DEST}")

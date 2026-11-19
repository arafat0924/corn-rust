r"""Evaluate the full app pipeline (gate -> leaf crop -> model) on the held-out test sets.

Run: .\.venv\Scripts\python.exe scripts\eval_app.py [output.csv]
Models and gate come from results/run3, or from the CORNRUST_RUN folder.
Output: reports/app_eval.csv (or the given path). "rejected" = the gate refused the image (counted as not correct).
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("app", ROOT / "app" / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

SETS = {"test (main source)": ROOT / "data/final/corn_rust_v3/test",
        "test_other (BCDD source)": ROOT / "data/final/corn_rust_v3/test_other"}

rows = []
for set_name, folder in SETS.items():
    items = [(p, p.parent.name) for p in sorted(folder.rglob("*.jpg"))]
    gated = []
    for p, truth in items:
        crop = app.leaf_crop(Image.open(p).convert("RGB"))
        group, view = app.gate_check(crop) if app.GATE else ("corn", crop)
        gated.append((group == "corn", view, truth))
    for model_name in app.MODELS:
        if model_name == getattr(app, "PUBLISHED_NAME", None):
            continue  # published model is evaluated in scripts/eval_published.py
        correct = rejected = 0
        for ok, view, truth in gated:
            if not ok:
                rejected += 1
                continue
            probs, _ = app.classify(view, model_name)
            correct += app.CLASSES[int(probs.argmax())] == truth
        n = len(gated)
        rows.append(dict(run=app.RUN.name, set=set_name, model=model_name.split(" (")[0], n=n,
                         rejected_by_gate=rejected / n, acc_end_to_end=correct / n,
                         acc_on_accepted=correct / max(1, n - rejected)))
        print(rows[-1])

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "app_eval.csv"
pd.DataFrame(rows).round(4).to_csv(out, index=False)
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))

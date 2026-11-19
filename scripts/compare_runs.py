r"""Compare a retrained run with the current one and say which models the app should serve.

Run: .\.venv\Scripts\python.exe scripts\compare_runs.py results\run4 results\run3_newgate
  new  : results/<new run>   (results.csv, crop_stats.csv, app_eval.csv from deploy/modal_pipeline.py)
  base : the current models judged with the same new gate (app_eval.csv), plus results/run3 for the rest
Rule: the new models win only if the app's default (the ensemble) is at least as accurate end to end
on BOTH test sets. Same split? Checked image for image via crop_stats.csv (same order, same crop sizes).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
new, base = (Path(a) for a in sys.argv[1:3])
old = ROOT / "results" / "run3"

a, b = pd.read_csv(new / "crop_stats.csv"), pd.read_csv(old / "crop_stats.csv")
same_rows = len(a) == len(b) and (a[["split", "cls"]].values == b[["split", "cls"]].values).all()
same_frac = same_rows and np.allclose(a.frac.fillna(-1), b.frac.fillna(-1), atol=0.02)
print(f"same split as run3: {'YES' if same_frac else 'NO'} "
      f"(rows {len(a)} vs {len(b)}, order+class match {same_rows}, crop sizes match {same_frac})")

cols = ["test_acc", "macro_f1", "test_other_acc"]
r = pd.concat({"new": pd.read_csv(new / "results.csv", index_col=0)[cols],
               "run3": pd.read_csv(old / "results.csv", index_col=0)[cols]}, axis=1)
print("\nnotebook test results (each run on its own split):\n", r.round(4).to_string())

e = pd.concat([pd.read_csv(new / "app_eval.csv").assign(models="new"),
               pd.read_csv(base / "app_eval.csv").assign(models="run3")])
t = e.pivot_table(index=["set", "model"], columns="models", values="acc_end_to_end")
print("\nwhole app (leaf crop -> new gate -> model), accuracy over ALL images:\n", t.round(4).to_string())

ens = t.xs("Ensemble: SmallCNN + MobileNetV3", level="model")
new_wins = bool((ens["new"] >= ens["run3"] - 1e-9).all())
print(f"\nDECISION: serve the {'NEW' if new_wins else 'CURRENT (run3)'} models "
      f"(ensemble new {ens['new'].round(4).tolist()} vs run3 {ens['run3'].round(4).tolist()})")
if not same_frac:
    print("note: the splits differ, so run3 may have trained on some of the new test images (favours run3)")

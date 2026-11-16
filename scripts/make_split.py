r"""Step 5: build a train/val/test split for a chosen set of labels and export it as image folders.

Run: .\.venv\Scripts\python.exe scripts\make_split.py --name corn_rust ^
        --labels Corn___Common_rust Corn___Healthy
Outputs:
  data/final/<name>/{train,val,test}/<class>/*.jpg   (ImageFolder layout, class = part after "___")
  data/final/<name>/split.csv                       (one row per image, with source path and split)
  data/final/<name>.zip                             (for upload to Colab / Kaggle)
Duplicates were already removed in Step 4, so a stratified random split has no near-copy leakage.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra source datasets (e.g. bcdd): their own train/val splits are added to "
                         "train/val and their test split becomes test_other (other-source test)")
    a = ap.parse_args()

    m = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")
    df = m[m.keep & m.label.isin(a.labels)].copy()
    missing = set(a.labels) - set(df.label)
    if missing:
        raise SystemExit(f"labels not found: {missing}")
    df["cls"] = df.label.str.split("___").str[1]

    rest, test = train_test_split(df, test_size=a.test, stratify=df.cls, random_state=a.seed)
    train, val = train_test_split(rest, test_size=a.val / (1 - a.test), stratify=rest.cls,
                                  random_state=a.seed)
    df["split"] = ""
    for name, part in (("train", train), ("val", val), ("test", test)):
        df.loc[part.index, "split"] = name

    if a.extra:  # other-source data keeps its own split (its rows may be keep=False as "redundant")
        ex = m[m.dataset.isin(a.extra) & m.label.isin(a.labels)
               & m.drop_reason.fillna("").isin(["", "redundant_source"])].copy()
        ex = ex[~ex.dataset.isin(df.dataset.unique())]
        ex["cls"] = ex.label.str.split("___").str[1]
        ex["split"] = ex.split.map({"train": "train", "val": "val", "valid": "val", "test": "test_other"})
        print(f"extra sources {a.extra}: {len(ex)} images")
        df = pd.concat([df, ex[ex.split.notna()]])

    out = ROOT / "data" / "final" / a.name
    if out.exists():
        shutil.rmtree(out)
    for _, r in df.iterrows():
        dest = out / r.split / r.cls / Path(r.proc_path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r.proc_path, dest)
    df[["proc_path", "path", "dataset", "label", "cls", "split"]].to_csv(out / "split.csv", index=False)
    zip_path = shutil.make_archive(str(out), "zip", out)

    print(pd.crosstab(df.cls, df.split, margins=True).to_string())
    print(f"\nexported to {out}\nzip: {zip_path} ({Path(zip_path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

r"""Step 3: map every raw image to one unified label.

Run (after scripts\explore_data.py):  .\.venv\Scripts\python.exe scripts\apply_labels.py
Inputs : data/inventory.csv, configs/label_map.csv
Outputs: data/labeled.csv (inventory rows that got a label, plus `label`, `crop`, `field_bg`)
         reports/labels/class_counts.csv (images per label per dataset)
Unmapped (dataset, class) pairs are printed so new downloads can be added to the map.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "labels"

# Only the original (non-augmented) images; augmentation is done at training time.
KEEP_ONLY_PATH_CONTAINING = {
    "veg_leaf_bd": "original",
    "riceleafbd": "original",
    "wheat_bd": "original",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(ROOT / "data" / "inventory.csv")
    lmap = pd.read_csv(ROOT / "configs" / "label_map.csv")

    for ds, needle in KEEP_ONLY_PATH_CONTAINING.items():
        drop = (inv.dataset == ds) & ~inv.path.str.lower().str.contains(needle)
        if drop.any():
            print(f"{ds}: skipping {drop.sum()} non-original (augmented) files")
        inv = inv[~drop]

    df = inv.merge(lmap, how="left", left_on=["dataset", "class"],
                   right_on=["dataset", "source_class"]).drop(columns="source_class")

    unmapped = df[df.label.isna()].groupby(["dataset", "class"]).size()
    if len(unmapped):
        print("\nUNMAPPED (add to configs/label_map.csv):")
        print(unmapped.to_string())

    df = df[df.label.notna() & df.ok]
    df["crop"] = df.label.str.split("___").str[0]
    df["field_bg"] = df.path.str.lower().str.contains("field background")
    df.to_csv(ROOT / "data" / "labeled.csv", index=False)

    counts = df.pivot_table(index="label", columns="dataset", values="path",
                            aggfunc="count", fill_value=0)
    counts["TOTAL"] = counts.sum(axis=1)
    counts = counts.sort_index()
    counts.to_csv(OUT / "class_counts.csv")
    print(f"\n{len(df)} labeled images, {df.label.nunique()} classes, "
          f"{df.crop.nunique()} crops\n")
    print(counts.to_string())


if __name__ == "__main__":
    main()

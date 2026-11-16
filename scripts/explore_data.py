r"""Step 2: inventory every raw image and make sample grids.

Run: .\.venv\Scripts\python.exe scripts\explore_data.py
Outputs:
  data/inventory.csv              one row per image (dataset, class, split, size, ok)
  reports/explore/summary.txt     per-dataset / per-class counts and image sizes
  reports/explore/grid_<ds>.jpg   4 sample images per class
"""
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "explore"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train", "val", "valid", "test"}


def inspect(path: Path):
    ds = path.relative_to(RAW).parts[0]
    rel = path.relative_to(RAW / ds)
    split = next((p for p in rel.parts[:-1] if p.lower() in SPLITS), "")
    row = {"dataset": ds, "class": path.parent.name, "split": split,
           "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
           "w": None, "h": None, "mode": None, "ok": False}
    try:
        with Image.open(path) as im:
            row["w"], row["h"], row["mode"] = im.width, im.height, im.mode
            im.verify()
        row["ok"] = True
    except Exception:
        pass
    return row


def make_grid(df, ds, per_class=4, tile=160):
    classes = sorted(df["class"].unique())
    grid = Image.new("RGB", (tile * per_class, (tile + 18) * len(classes)), "white")
    draw = ImageDraw.Draw(grid)
    rng = random.Random(0)
    for r, cls in enumerate(classes):
        paths = df[df["class"] == cls]["path"].tolist()
        y = r * (tile + 18)
        draw.text((4, y + 2), f"{cls} ({len(paths)})", fill="black")
        for c, p in enumerate(rng.sample(paths, min(per_class, len(paths)))):
            with Image.open(ROOT / p) as im:
                im = im.convert("RGB")
                im.thumbnail((tile, tile))
                grid.paste(im, (c * tile, y + 18))
    grid.save(OUT / f"grid_{ds}.jpg", quality=85)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [p for p in RAW.rglob("*") if p.suffix.lower() in EXTS and p.is_file()]
    print(f"found {len(files)} image files")
    with ThreadPoolExecutor(8) as ex:
        df = pd.DataFrame(list(ex.map(inspect, files)))
    df.to_csv(ROOT / "data" / "inventory.csv", index=False)

    lines = []
    for ds, g in df.groupby("dataset"):
        ok = g[g.ok]
        lines.append(f"\n=== {ds}: {len(g)} files, {len(g) - len(ok)} unreadable, "
                     f"{g.bytes.sum() / 1e9:.2f} GB")
        lines.append(f"  size (w x h): median {ok.w.median():.0f}x{ok.h.median():.0f}, "
                     f"min {ok.w.min()}x{ok.h.min()}, max {ok.w.max()}x{ok.h.max()}; "
                     f"modes {dict(ok['mode'].value_counts())}")
        counts = g.groupby(["class", "split"]).size().unstack(fill_value=0)
        lines.append(counts.to_string())
        make_grid(ok, ds)
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

r"""Step 4: resize, filter, and de-duplicate the labeled images.

Run (after apply_labels.py):  .\.venv\Scripts\python.exe scripts\clean_data.py [--threshold 4]
Incremental: images already processed are not re-read (cache: data/processed/hashes.csv).

Outputs:
  data/processed/images/<label>/<id>.jpg   shorter side = 288 px (never upscaled), EXIF-rotated
  data/processed/manifest.csv              one row per labeled image with keep / drop_reason / group_id
  reports/clean/summary.txt                per-class counts before -> after
  reports/clean/dup_pairs_d<T>.jpg         borderline duplicate pairs, for eyeballing the threshold
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
IMG_DIR = PROC / "images"
REPORT = ROOT / "reports" / "clean"
SHORT_SIDE = 288
MIN_SIDE = 64
# Prefer these sources when choosing which duplicate to keep (higher = better).
# Thin leaves on white paper look alike to phash, so rice needs a stricter duplicate threshold.
CROP_THRESHOLD = {"Rice": 2}
SOURCE_RANK = {"riceleafbd": 5, "dhan_shomadhan": 5, "veg_leaf_bd": 4, "wheat_bd": 4,
               "bcd_nafishamoin": 3, "bcdd": 1}


def proc_id(src_path: str) -> str:
    # Hash the Windows form of the path, so ids (= processed file names) match the originals
    # built on Windows no matter which OS runs this.
    return hashlib.sha1(src_path.replace("/", "\\").encode("utf-8")).hexdigest()[:16]


def dihedral(im):
    """The 8 flips/rotations of an image."""
    out = []
    for base in (im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
        for rot in (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                    Image.Transpose.ROTATE_270):
            out.append(base if rot is None else base.transpose(rot))
    return out


def process(args):
    src, label = args
    pid = proc_id(src)
    dest = IMG_DIR / label / f"{pid}.jpg"
    row = {"path": src, "proc_path": str(dest.relative_to(ROOT)), "pw": None, "ph": None,
           **{f"h{i}": None for i in range(8)}, "err": ""}
    try:
        with Image.open(ROOT / src) as im:
            im.draft("RGB", (SHORT_SIDE * 2, SHORT_SIDE * 2))  # fast JPEG decode at reduced scale
            im = ImageOps.exif_transpose(im).convert("RGB")
            s = min(im.size)
            if s > SHORT_SIDE:
                r = SHORT_SIDE / s
                im = im.resize((round(im.width * r), round(im.height * r)), Image.Resampling.LANCZOS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest, quality=92)
            row["pw"], row["ph"] = im.size
            small = im.copy()
            small.thumbnail((64, 64))
            for i, v in enumerate(dihedral(small)):
                row[f"h{i}"] = str(imagehash.phash(v))
    except Exception as e:
        row["err"] = repr(e)
    return row


def hash_all(df):
    cache_f = PROC / "hashes.csv"
    cache = pd.read_csv(cache_f, dtype=str) if cache_f.exists() else pd.DataFrame(columns=["path"])
    if len(cache):
        exists = cache.proc_path.map(lambda p: (ROOT / p).exists())
        cache = cache[exists & cache.err.fillna("").eq("")]
    todo = df[~df.path.isin(set(cache.path))]
    print(f"hashing {len(todo)} new images ({len(cache)} cached)")
    rows = []
    if len(todo):
        with ProcessPoolExecutor() as ex:
            for i, r in enumerate(ex.map(process, zip(todo.path, todo.label), chunksize=64), 1):
                rows.append(r)
                if i % 5000 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
    new = pd.DataFrame(rows, dtype=str)
    allh = pd.concat([cache, new], ignore_index=True)
    allh.to_csv(cache_f, index=False)
    return allh


def hashes(sub):
    """(n,) original hashes and (n, 8) flip/rotation hashes as uint64."""
    a = np.array([int(h, 16) for h in sub.h0], dtype=np.uint64)
    b = np.stack([np.array([int(h, 16) for h in sub[f"h{k}"]], dtype=np.uint64) for k in range(8)], 1)
    return a, b


def greedy_dedupe(sub, threshold):
    """Walk images best-first; keep one unless it is within `threshold` of an already-kept image.
    Direct comparison only (no transitive chaining). Returns (kept mask, matched-to index, border pairs)."""
    a, b = hashes(sub)
    n = len(sub)
    kept_idx, keep, match, border = [], np.zeros(n, bool), np.full(n, -1), []
    for i in range(n):
        if kept_idx:
            d = np.bitwise_count(a[kept_idx][:, None] ^ b[i][None, :]).min(axis=1)
            j = int(d.argmin())
            if d[j] <= threshold:
                match[i] = kept_idx[j]
                if d[j] == threshold:
                    border.append((kept_idx[j], i))
                continue
        keep[i] = True
        kept_idx.append(i)
    return keep, match, border


def conflict_pairs(sub, threshold):
    """Pairs with different labels that are near-identical (distance <= threshold)."""
    a, b = hashes(sub)
    labels = sub.label.to_numpy()
    out = set()
    for s in range(0, len(sub), 256):
        d = np.bitwise_count(a[s:s + 256, None, None] ^ b[None, :, :]).min(axis=2)
        for i, j in zip(*np.nonzero(d <= threshold)):
            if labels[i + s] != labels[j]:
                out.update((i + s, j))
    return sorted(out)


def pair_grid(pairs, out_file, tile=128):
    if not pairs:
        return
    pairs = pairs[:24]
    g = Image.new("RGB", (tile * 2 + 10, (tile + 16) * len(pairs)), "white")
    dr = ImageDraw.Draw(g)
    for r, (p1, l1, p2, l2) in enumerate(pairs):
        y = r * (tile + 16)
        dr.text((2, y + 2), f"{l1} | {l2}", fill="black")
        for c, p in enumerate((p1, p2)):
            with Image.open(ROOT / p) as im:
                im.thumbnail((tile, tile))
                g.paste(im, (c * (tile + 10), y + 16))
    g.save(out_file, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=4, help="max phash distance for same-label duplicates")
    ap.add_argument("--conflict", type=int, default=2, help="max phash distance for cross-label conflicts")
    args = ap.parse_args()
    t = args.threshold
    REPORT.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(ROOT / "data" / "labeled.csv")
    h = hash_all(lab[["path", "label"]])
    df = lab.merge(h, on="path", how="left")
    df["keep"], df["drop_reason"], df["group_id"] = True, "", ""

    bad = df.err.fillna("").ne("") | df.h0.isna()
    df.loc[bad, ["keep", "drop_reason"]] = [False, "unreadable"]
    small = ~bad & (df[["w", "h"]].min(axis=1) < MIN_SIDE)
    df.loc[small, ["keep", "drop_reason"]] = [False, "too_small"]

    # 0) BCDD's corn/potato/rice/wheat are 96px augmented copies of the nafishamoin images:
    #    drop BCDD only for labels nafishamoin also has (BCDD tomato / rice BLB / loose smut stay)
    naf_labels = set(df[df.dataset == "bcd_nafishamoin"].label)
    red = df.keep & (df.dataset == "bcdd") & df.label.isin(naf_labels)
    df.loc[red, ["keep", "drop_reason"]] = [False, "redundant_source"]

    # 1) same-label duplicates: greedy, best source / highest resolution first
    df["rank"] = df.dataset.map(SOURCE_RANK).fillna(0)
    df["area"] = df.w * df.h
    border_pairs = []
    for label, sub in df[df.keep].groupby("label"):
        sub = sub.sort_values(["rank", "area", "path"], ascending=[False, False, True])
        keep, match, border = greedy_dedupe(sub, CROP_THRESHOLD.get(label.split("___")[0], t))
        idx = sub.index.to_numpy()
        df.loc[idx, "group_id"] = [str(idx[m] if m >= 0 else idx[i]) for i, m in enumerate(match)]
        df.loc[idx[~keep], ["keep", "drop_reason"]] = [False, "duplicate"]
        border_pairs += [(df.at[idx[i], "proc_path"], label, df.at[idx[j], "proc_path"], label)
                         for i, j in border[:3]]

    # 2) cross-label conflicts: near-identical images under different labels -> drop both
    for crop, sub in df[df.keep].groupby("crop"):
        bad_i = conflict_pairs(sub, args.conflict)
        df.loc[sub.index[bad_i], ["keep", "drop_reason"]] = [False, "label_conflict"]

    cols = ["path", "proc_path", "dataset", "label", "crop", "field_bg", "split", "w", "h",
            "pw", "ph", "group_id", "keep", "drop_reason"]
    df[cols].to_csv(PROC / "manifest.csv", index=False)
    pair_grid(border_pairs, REPORT / f"dup_pairs_d{t}.jpg")

    before = df.groupby("label").size().rename("before")
    after = df[df.keep].groupby("label").size().rename("after")
    reasons = df[~df.keep].groupby(["label", "drop_reason"]).size().unstack(fill_value=0)
    table = pd.concat([before, after], axis=1).fillna(0).astype(int).join(reasons).fillna(0)
    by_ds = df.groupby(["dataset", "drop_reason"]).size().unstack(fill_value=0)
    text = (f"threshold={t} conflict={args.conflict}\nkept {df.keep.sum()} / {len(df)}\n\n{table.to_string()}\n\n"
            f"by dataset:\n{by_ds.to_string()}\n")
    (REPORT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

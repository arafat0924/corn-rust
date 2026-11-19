r"""Build and evaluate the "is this a supported corn leaf?" gate, v3 (relative kNN over 4 banks).

Run: $env:CORNRUST_RUN="results\run4"; .\.venv\Scripts\python.exe scripts\build_ood_gate.py
     (without CORNRUST_RUN the gate is written to results/run3)

v1 (absolute similarity threshold vs our training photos) rejected real corn leaves from other
sources (BCDD corn accepted only 11-37%). v2 asks *which reference group is the image closest to*:
  corn        : corn rust + healthy (nafishamoin train split + BCDD train split)  -> accepted
  corn_other  : corn gray leaf spot + northern leaf blight                       -> rejected
  other_crop  : leaves of rice, wheat, potato, tomato, eggplant, cauliflower     -> rejected
  not_leaf    : non-plant photos                                                 -> rejected
v3 adds synthetic graphics (logos, gradients, abstract wallpapers, UI-like text) to not_leaf: v2's
bank held only real photos, so smooth graphics such as a flat blue logo were accepted as corn.
Group score = mean cosine similarity to its K nearest members (ImageNet EfficientNet-B0 features).
Multi-view (full image + 5 zoomed crops) is evaluated too but stays off: it let junk through.
Everything is evaluated on held-out images that are NOT in the banks.
Outputs: <run>/ood_bank.pt, <run>/ood_gate.json, reports/ood/ood_eval.csv
"""
import colorsys
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data" / "final" / "corn_rust"
K, SEED, MAX_BANK = 5, 0, 1500
N_SYNTH_BANK, N_SYNTH_EVAL = 400, 200
GROUPS = ["corn", "corn_other", "other_crop", "not_leaf"]

spec = importlib.util.spec_from_file_location("app_core", ROOT / "app" / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)  # reuse leaf_crop, TF, views and the classifiers exactly as the app does
RUN = app.RUN                 # results/run3, or the CORNRUST_RUN folder

backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).eval()


@torch.no_grad()
def embed_tensors(x, bs=64):
    return torch.cat([F.normalize(backbone(x[i:i + bs]), dim=1) for i in range(0, len(x), bs)])


def load(paths):
    return [app.leaf_crop(Image.open(p).convert("RGB")) for p in paths]


def embed_imgs(imgs):
    return embed_tensors(torch.stack([app.TF(im) for im in imgs]))


def group_scores(f, banks):
    return torch.stack([(f @ banks[g].T).topk(K, dim=1).values.mean(1) for g in GROUPS], 1)


def synthetic_graphics(n, seed, size=256):
    """Non-photographic images: flat logos, gradients, abstract 'wallpaper' curves and UI/text cards."""
    rng = np.random.default_rng(seed)

    def colour(sat=(0.25, 1.0), val=(0.25, 1.0)):
        r, g, b = colorsys.hsv_to_rgb(rng.random(), rng.uniform(*sat), rng.uniform(*val))
        return int(r * 255), int(g * 255), int(b * 255)

    def gradient(a, b, radial=False):
        yy, xx = np.mgrid[0:size, 0:size] / size
        if radial:
            cx, cy = rng.random(2)
            t = np.clip(np.hypot(xx - cx, yy - cy) / rng.uniform(.5, 1.2), 0, 1)
        else:
            th = rng.uniform(0, 2 * np.pi)
            t = (xx * np.cos(th) + yy * np.sin(th))
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        arr = np.array(a, float) + (np.array(b, float) - np.array(a, float)) * t[..., None]
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    out = []
    for i in range(n):
        kind = i % 4
        if kind == 0:                                   # logo: flat background + flat shapes
            im = Image.new("RGB", (size, size), colour())
            d = ImageDraw.Draw(im)
            for _ in range(int(rng.integers(1, 5))):
                s = int(rng.integers(40, 180))
                x0, y0 = (int(v) for v in rng.integers(0, size - s // 2, 2))
                box, fill = [x0, y0, x0 + s, y0 + s], colour()
                shape = int(rng.integers(4))
                if shape == 0:
                    d.ellipse(box, fill=fill)
                elif shape == 1:
                    d.rectangle(box, fill=fill)
                elif shape == 2:
                    d.polygon([(x0, y0 + s), (x0 + s // 2, y0), (x0 + s, y0 + s)], fill=fill)
                else:
                    d.rounded_rectangle(box, radius=s // 4, outline=fill, width=int(rng.integers(6, 20)))
        elif kind == 1:                                 # plain linear / radial gradient
            im = gradient(colour(), colour(), radial=bool(rng.integers(2)))
        elif kind == 2:                                 # abstract wallpaper: soft overlapping bands
            im = gradient(colour(val=(.2, .9)), colour(val=(.2, .9)))
            d = ImageDraw.Draw(im)
            for _ in range(int(rng.integers(3, 8))):
                pts = [(float(x), float(y)) for x, y in rng.uniform(-60, size + 60, (4, 2))]
                d.line(pts, fill=colour(), width=int(rng.integers(15, 70)), joint="curve")
            im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(3, 12))))
        else:                                           # UI card / document: text-like bars on a flat page
            im = Image.new("RGB", (size, size), colour(sat=(0, .3), val=(.85, 1)))
            d = ImageDraw.Draw(im)
            y, ink = int(rng.integers(10, 40)), colour(val=(0, .5))
            while y < size - 20:
                h = int(rng.integers(6, 22))
                d.rectangle([16, y, 16 + int(rng.integers(60, size - 32)), y + h], fill=ink)
                d.text((20, y + h + 2), "Lorem ipsum 123", fill=ink)
                y += h + int(rng.integers(14, 36))
        out.append(im.convert("RGB"))
    return out


def main():
    rng = np.random.default_rng(SEED)
    m = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")
    split = pd.read_csv(SPLIT / "split.csv")

    def holdout(df, frac=0.2):
        idx = rng.permutation(len(df))
        n = int(len(df) * frac)
        return df.iloc[idx[n:]], df.iloc[idx[:n]]

    bcdd_corn = m[(m.dataset == "bcdd") & m.label.isin(["Corn___Common_rust", "Corn___Healthy"])]
    corn_other = m[m.label.isin(["Corn___Gray_leaf_spot", "Corn___Northern_leaf_blight"])
                   & (m.keep | (m.dataset == "bcdd"))]
    other_crop = m[m.keep & ~m.crop.isin(["Corn", "Not_a_leaf"])]
    not_leaf = m[m.keep & (m.label == "Not_a_leaf")]
    co_bank, co_eval = holdout(corn_other)
    oc_bank, oc_eval = holdout(other_crop)
    nl_bank, nl_eval = holdout(not_leaf)

    bank_paths = {
        "corn": [str(SPLIT / "train" / r.cls / Path(r.proc_path).name) for r in split[split.split == "train"].itertuples()]
                + [str(ROOT / p) for p in bcdd_corn[bcdd_corn.split == "train"].proc_path],
        "corn_other": [str(ROOT / p) for p in co_bank.proc_path],
        "other_crop": [str(ROOT / p) for p in oc_bank.proc_path],
        "not_leaf": [str(ROOT / p) for p in nl_bank.proc_path],
    }
    banks = {}
    for g in GROUPS:
        paths = bank_paths[g]
        if len(paths) > MAX_BANK:
            paths = list(rng.choice(paths, MAX_BANK, replace=False))
        imgs = load(paths)
        if g == "not_leaf":
            imgs += [app.leaf_crop(im) for im in synthetic_graphics(N_SYNTH_BANK, seed=1)]
        print(f"bank {g}: {len(imgs)} images", flush=True)
        banks[g] = embed_imgs(imgs)

    # held-out evaluation sets: (name, images, expected group, true corn class or None)
    def ev(df, n=400):
        return df.sample(min(n, len(df)), random_state=SEED)

    test = split[split.split == "test"]
    bcdd_test = bcdd_corn[bcdd_corn.split == "test"]
    eval_sets = [
        ("our corn rust/healthy TEST",
         lambda: load([str(SPLIT / "test" / r.cls / Path(r.proc_path).name) for r in test.itertuples()]),
         "corn", list(test.cls)),
        ("BCDD corn rust/healthy TEST (other source)", lambda: load([str(ROOT / p) for p in bcdd_test.proc_path]),
         "corn", [lbl.split("___")[1] for lbl in bcdd_test.label]),
        ("corn other diseases (held out)", lambda: load([str(ROOT / p) for p in ev(co_eval).proc_path]),
         "corn_other", None),
        ("other crops' leaves (held out)", lambda: load([str(ROOT / p) for p in ev(oc_eval).proc_path]),
         "other_crop", None),
        ("non-plant photos (held out)", lambda: load([str(ROOT / p) for p in ev(nl_eval).proc_path]),
         "not_leaf", None),
        ("synthetic graphics (held out)",
         lambda: [app.leaf_crop(im) for im in synthetic_graphics(N_SYNTH_EVAL, seed=2)], "not_leaf", None),
    ]
    cnn = app.MODELS[next(k for k in app.MODELS if k.startswith("SmallCNN"))][0][0]
    rows = []
    for name, get_imgs, expect, truth in eval_sets:
        imgs = get_imgs()
        r = {"set": name, "n": len(imgs)}
        for mode in ("single", "multi"):
            views = [app.views(im) if mode == "multi" else [im] for im in imgs]
            flat = [v for vs in views for v in vs]
            s = group_scores(embed_imgs(flat), banks)
            off, accepted, best_views = 0, [], []
            for vs in views:
                block = s[off:off + len(vs)]
                off += len(vs)
                is_corn = block.argmax(1) == 0
                accepted.append(bool(is_corn.any()))
                margin = block[:, 0] - block[:, 1:].max(1).values
                best_views.append(vs[int(margin.argmax())])
            r[f"accepted_{mode}"] = float(np.mean(accepted))
            if truth is not None:
                with torch.no_grad():
                    pred = cnn(torch.stack([app.TF(v) for v in best_views])).argmax(1).numpy()
                ok = np.array([app.CLASSES[p] == t for p, t in zip(pred, truth)])
                acc_mask = np.array(accepted)
                r[f"cnn_acc_{mode}"] = float(ok.mean())
                r[f"cnn_acc_on_accepted_{mode}"] = float(ok[acc_mask].mean()) if acc_mask.any() else np.nan
        rows.append(r)
        print(f"  {name}: accepted {r['accepted_single']:.1%} (single view)", flush=True)

    out = ROOT / "reports" / "ood"
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(out / "ood_eval.csv", index=False)
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save({g: banks[g].half() for g in GROUPS}, RUN / "ood_bank.pt")
    (RUN / "ood_gate.json").write_text(json.dumps(
        {"version": 2, "gate_build": 3, "backbone": "efficientnet_b0 (ImageNet, timm)", "k": K, "groups": GROUPS,
         "accept_group": "corn", "multiview": False, "synthetic_not_leaf": N_SYNTH_BANK,
         "bank_sizes": {g: len(banks[g]) for g in GROUPS}}, indent=1))
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(f"gate written to {RUN}")


if __name__ == "__main__":
    main()

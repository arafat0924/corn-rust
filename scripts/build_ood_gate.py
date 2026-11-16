r"""Build and evaluate the "is this a supported corn leaf?" gate, v2 (relative kNN over 4 banks).

Run: .\.venv\Scripts\python.exe scripts\build_ood_gate.py

v1 (absolute similarity threshold vs our training photos) rejected real corn leaves from other
sources (BCDD corn accepted only 11-37%). v2 asks *which reference group is the image closest to*:
  corn        : corn rust + healthy (nafishamoin train split + BCDD train split)  -> accepted
  corn_other  : corn gray leaf spot + northern leaf blight                       -> "unsupported corn disease"
  other_crop  : leaves of rice, wheat, potato, tomato, eggplant, cauliflower     -> rejected
  not_leaf    : non-plant photos                                                 -> rejected
Group score = mean cosine similarity to its K nearest members (ImageNet EfficientNet-B0 features).
Multi-view: the full image plus 5 zoomed crops (60%); an image is accepted if any view is "corn".
Everything is evaluated on held-out images that are NOT in the banks.
Outputs: results/run2/ood_bank.pt, results/run2/ood_gate.json, reports/ood/ood_eval.csv
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "run2"
SPLIT = ROOT / "data" / "final" / "corn_rust"
K, SEED, MAX_BANK = 5, 0, 1500
GROUPS = ["corn", "corn_other", "other_crop", "not_leaf"]

spec = importlib.util.spec_from_file_location("app_core", ROOT / "app" / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)  # reuse leaf_crop, TF, views and the classifiers exactly as the app does

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
        print(f"bank {g}: {len(paths)} images")
        banks[g] = embed_imgs(load(paths))

    # held-out evaluation sets: (name, paths, expected group, true corn class or None)
    def ev(df, n=400):
        return df.sample(min(n, len(df)), random_state=SEED)

    test = split[split.split == "test"]
    bcdd_test = bcdd_corn[bcdd_corn.split == "test"]
    eval_sets = [
        ("our corn rust/healthy TEST", [str(SPLIT / "test" / r.cls / Path(r.proc_path).name) for r in test.itertuples()],
         "corn", list(test.cls)),
        ("BCDD corn rust/healthy TEST (other source)", [str(ROOT / p) for p in bcdd_test.proc_path], "corn",
         [l.split("___")[1].replace("Common_rust", "Common_rust") for l in bcdd_test.label]),
        ("corn other diseases (held out)", [str(ROOT / p) for p in ev(co_eval).proc_path], "corn_other", None),
        ("other crops' leaves (held out)", [str(ROOT / p) for p in ev(oc_eval).proc_path], "other_crop", None),
        ("non-plant photos (held out)", [str(ROOT / p) for p in ev(nl_eval).proc_path], "not_leaf", None),
    ]
    cnn = app.MODELS[next(iter(app.MODELS))][0]
    rows = []
    for name, paths, expect, truth in eval_sets:
        imgs = load(paths)
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
        print(f"  {name}: done")

    out = ROOT / "reports" / "ood"
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(out / "ood_eval.csv", index=False)
    torch.save({g: banks[g].half() for g in GROUPS}, RUN / "ood_bank.pt")
    (RUN / "ood_gate.json").write_text(json.dumps(
        {"version": 2, "backbone": "efficientnet_b0 (ImageNet, timm)", "k": K, "groups": GROUPS,
         "accept_group": "corn", "multiview": False, "bank_sizes": {g: len(banks[g]) for g in GROUPS}},
        indent=1))
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

# %% [markdown]
# # Corn Common Rust Detection: CNN from scratch vs EfficientNet-B0 vs MobileNetV3
#
# Binary image classifier for corn (maize) leaves: **Common_rust** vs **Healthy**.
#
# | Model | Idea |
# |---|---|
# | SmallCNN | 4 conv blocks trained from scratch (baseline) |
# | EfficientNet-B0 | ImageNet-pretrained, 2-stage fine-tuning (main model) |
# | MobileNetV3-Small | ImageNet-pretrained, lightweight / mobile-friendly |
#
# **Data:** `corn_rust` split from the Bangladeshi Crops Disease Dataset, after de-duplication
# (train 1577 / val 338 / test 339). Kaggle settings: **Accelerator = GPU**, **Internet = On**.

# %%
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"   # quick CPU test: tiny subset, 1 epoch per stage


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__, "| timm", timm.__version__, "| smoke:", SMOKE)


DATA_DIR = os.environ.get("DATA_DIR", "data/final/corn_rust_v3")   # used only when running locally


def find_data_root():
    """Find the folder that contains train/ val/ test/ (works on Kaggle, Colab, or locally)."""
    bases = [Path("/kaggle/input"), Path("/content"), Path("/tmp/unzipped"), Path(DATA_DIR)]
    for base in bases:
        if base.exists():
            for p in sorted(base.rglob("train")):
                if p.is_dir() and (p.parent / "val").is_dir() and (p.parent / "test").is_dir():
                    return p.parent
    for z in Path("/kaggle/input").rglob("*.zip") if Path("/kaggle/input").exists() else []:
        import zipfile                                  # dataset uploaded as a zip that wasn't extracted
        zipfile.ZipFile(z).extractall(Path("/tmp/unzipped") / z.stem)
        return find_data_root()
    raise FileNotFoundError("No folder with train/val/test found. Did you add the dataset as input?")


RAW_DATA = find_data_root()
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(os.environ.get("OUT_DIR", "outputs"))
OUT.mkdir(exist_ok=True)
print("raw data:", RAW_DATA, "| outputs:", OUT)

# %% [markdown]
# ## 0. Removing a shortcut: the black background
# Run 1 hit **100% test accuracy, and the model was cheating**. 99.7% of the *Common_rust*
# images are cut-out leaves on a **black background**, while 99.9% of the *Healthy* images are
# full-frame leaf close-ups. A model can separate the classes just by checking for black corners,
# and Grad-CAM confirmed it was looking at the background, not the rust pustules.
#
# Fix: crop every image to the **largest square that contains only leaf** (no black pixels).
# Images with no black background get a random crop of the *same relative size* as the rust crops,
# so zoom level can't become a new shortcut either. Crops smaller than 35% of the image side are dropped.

# %%
from PIL import Image
from scipy.ndimage import distance_transform_cdt

BLACK_THR, MIN_FRAC, MASK_RES = 20, 0.35, 128


def black_fraction(img):
    a = np.asarray(img.convert("RGB").resize((64, 64)))
    return float((a.max(axis=2) < BLACK_THR).mean())


def leaf_square(img):
    """(left, top, side) of the largest all-leaf square, or None if the image has no black area."""
    w, h = img.size
    s = MASK_RES / max(w, h)
    a = np.asarray(img.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s)))))
    mask = np.pad(a.max(axis=2) >= BLACK_THR, 1)          # pad: square must stay inside the image
    if mask[1:-1, 1:-1].all():
        return None
    d = distance_transform_cdt(mask, metric="chessboard")[1:-1, 1:-1]
    best = d.max()
    ys, xs = np.nonzero(d == best)
    cy, cx = d.shape[0] / 2, d.shape[1] / 2
    k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)      # most central of the best squares
    half = (best - 1) / s + 0.5 / s
    side = int(2 * half * 0.92)                         # 8% safety margin away from the black edge
    cx_o, cy_o = (xs[k] + 0.5) / s, (ys[k] + 0.5) / s
    left = int(np.clip(cx_o - side / 2, 0, w - side))
    top = int(np.clip(cy_o - side / 2, 0, h - side))
    return left, top, side


def build_cropped(src, dst):
    rng = np.random.default_rng(SEED)
    files = sorted(src.rglob("*.jpg"))
    plan, fracs = [], []
    for f in files:
        with Image.open(f) as im:
            sq = leaf_square(im)
            frac = None if sq is None else sq[2] / min(im.size)
        plan.append((f, sq, frac))
        if frac is not None:
            fracs.append(frac)
    fracs = np.array(fracs) if fracs else np.array([1.0])
    stats = []
    for f, sq, frac in plan:
        rel = f.relative_to(src)
        with Image.open(f) as im:
            im = im.convert("RGB")
            w, h = im.size
            if sq is None:                              # no black: random crop with a matched size
                frac = float(np.clip(rng.choice(fracs), MIN_FRAC, 1.0))
                side = int(min(w, h) * frac)
                left, top = int(rng.integers(0, w - side + 1)), int(rng.integers(0, h - side + 1))
            else:
                left, top, side = sq
            kept = frac >= MIN_FRAC
            if kept:
                crop = im.crop((left, top, left + side, top + side))
                out = dst / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out, quality=95)
            stats.append(dict(split=rel.parts[0], cls=rel.parts[1], frac=frac, kept=kept,
                              black_before=black_fraction(im),
                              black_after=black_fraction(crop) if kept else np.nan))
    return pd.DataFrame(stats)


DATA = Path("/tmp/corn_rust_cropped") if Path("/kaggle/working").exists() else Path("outputs/cropped")
if DATA.exists():
    import shutil; shutil.rmtree(DATA)
crop_stats = build_cropped(RAW_DATA, DATA)
crop_stats.to_csv(OUT / "crop_stats.csv", index=False)
summary = crop_stats.groupby("cls").agg(
    images=("kept", "size"), kept=("kept", "sum"),
    with_black_before=("black_before", lambda v: (v > 0.02).mean()),
    with_black_after=("black_after", lambda v: (v > 0.02).mean()),
    mean_crop_frac=("frac", "mean")).round(3)
print(summary.to_string())
print(pd.crosstab(crop_stats[crop_stats.kept].split, crop_stats[crop_stats.kept].cls).to_string())

# %% [markdown]
# ## 1. Data and augmentation
# Training images get random crops, flips, rotation and color jitter so the model can't memorize
# exact pixels. Val/test images are only resized and center-cropped.

# %%
IMG = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(IMG),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])

SPLITS = ["train", "val", "test"] + (["test_other"] if (DATA / "test_other").is_dir() else [])
ds = {s: datasets.ImageFolder(DATA / s, train_tf if s == "train" else eval_tf) for s in SPLITS}
CLASSES = ds["train"].classes
if SMOKE:
    ds = {s: Subset(d, list(range(0, len(d), max(1, len(d) // 32)))) for s, d in ds.items()}
BATCH = 32
dl = {s: DataLoader(ds[s], batch_size=BATCH, shuffle=(s == "train"),
                    num_workers=int(os.environ.get("NUM_WORKERS", 0 if SMOKE else 2)),
                    pin_memory=DEVICE == "cuda")
      for s in ds}
print("classes:", CLASSES)
print({s: len(d) for s, d in ds.items()})

# %%
def denorm(x):
    return (x.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)


xb, yb = next(iter(dl["train"]))
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for ax, x, y in zip(axes.flat, xb, yb):
    ax.imshow(denorm(x)); ax.set_title(CLASSES[y]); ax.axis("off")
plt.suptitle("Augmented training samples"); plt.tight_layout()
plt.savefig(OUT / "samples.png", dpi=120); plt.show()

# %% [markdown]
# ## 2. Training loop
# AdamW + cosine learning-rate schedule, mixed precision on GPU, label smoothing,
# early stopping on validation loss, and the best checkpoint saved as `<model>_best.pth`.

# %%
def run_epoch(model, loader, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    total, ys, ps = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=0.05 if train else 0.0)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total += loss.item() * x.size(0)
            ys.append(y.cpu()); ps.append(logits.float().softmax(1).detach().cpu())
    ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return total / len(loader.dataset), float((ps.argmax(1) == ys).mean()), ys, ps


def fit(model, name, epochs, lr, history=None, patience=5, wd=1e-4):
    if SMOKE:
        epochs = 1
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    history = [] if history is None else history
    best = min([h["val_loss"] for h in history], default=float("inf"))
    bad = 0
    for _ in range(epochs):
        t = time.time()
        tl, ta, *_ = run_epoch(model, dl["train"], opt, scaler)
        vl, va, *_ = run_epoch(model, dl["val"])
        sched.step()
        history.append(dict(epoch=len(history) + 1, train_loss=tl, train_acc=ta,
                            val_loss=vl, val_acc=va, secs=time.time() - t))
        print(f"{name} ep{len(history):02d}  train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} | {time.time() - t:.0f}s")
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(model.state_dict(), OUT / f"{name}_best.pth")
        else:
            bad += 1
            if bad >= patience:
                print("early stopping"); break
    model.load_state_dict(torch.load(OUT / f"{name}_best.pth", map_location=DEVICE))
    return history


histories, train_secs, models = {}, {}, {}

# %% [markdown]
# ## 3. Model 1: SmallCNN from scratch (baseline)

# %%
class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        def block(i, o, pool=True):
            layers = [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
            return layers + ([nn.MaxPool2d(2)] if pool else [])

        self.features = nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128),
                                      *block(128, 256, pool=False))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


seed_all(SEED)
cnn = SmallCNN(len(CLASSES)).to(DEVICE)
print(f"SmallCNN params: {sum(p.numel() for p in cnn.parameters()) / 1e6:.2f}M")
t0 = time.time()
histories["small_cnn"] = fit(cnn, "small_cnn", epochs=30, lr=1e-3, patience=7)
train_secs["small_cnn"] = time.time() - t0
models["small_cnn"] = cnn

# %% [markdown]
# ## 4. Model 2: EfficientNet-B0 (transfer learning)
# Stage 1: freeze the pretrained backbone and train only the new classifier head.
# Stage 2: unfreeze everything and fine-tune with a 10x smaller learning rate.

# %%
def two_stage(arch, name, head_epochs=3, ft_epochs=12):
    seed_all(SEED)
    m = timm.create_model(arch, pretrained=True, num_classes=len(CLASSES)).to(DEVICE)
    print(f"{name} params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
    t = time.time()
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith("classifier")
    h = fit(m, name, epochs=head_epochs, lr=1e-3)
    for p in m.parameters():
        p.requires_grad = True
    h = fit(m, name, epochs=ft_epochs, lr=1e-4, history=h)
    histories[name], train_secs[name], models[name] = h, time.time() - t, m
    return m


eff = two_stage("efficientnet_b0", "efficientnet_b0")

# %% [markdown]
# ## 5. Model 3: MobileNetV3-Small (lightweight, phone-friendly)

# %%
mob = two_stage("mobilenetv3_small_100", "mobilenetv3_small")

# %% [markdown]
# ## 6. Learning curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name, h in histories.items():
    e = [r["epoch"] for r in h]
    axes[0].plot(e, [r["val_loss"] for r in h], label=f"{name} val")
    axes[0].plot(e, [r["train_loss"] for r in h], "--", alpha=0.5, label=f"{name} train")
    axes[1].plot(e, [r["val_acc"] for r in h], label=f"{name} val")
axes[0].set_title("Loss"); axes[1].set_title("Validation accuracy")
for ax in axes:
    ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "learning_curves.png", dpi=120); plt.show()
json.dump(histories, open(OUT / "histories.json", "w"), indent=1)

# %% [markdown]
# ## 7. Test-set evaluation
# The test set was never used for training or model selection.

# %%
rows, preds = [], {}
for name, m in models.items():
    t = time.time()
    _, acc, ys, ps = run_epoch(m, dl["test"])
    ms = (time.time() - t) / len(ds["test"]) * 1000
    pred = ps.argmax(1)
    prec, rec, f1, _ = precision_recall_fscore_support(ys, pred, average="macro", zero_division=0)
    preds[name] = (ys, ps)
    rows.append(dict(model=name, test_acc=acc, macro_precision=prec, macro_recall=rec, macro_f1=f1,
                     roc_auc=roc_auc_score(ys, ps[:, 1]) if len(set(ys)) > 1 else float("nan"),
                     params_M=sum(p.numel() for p in m.parameters()) / 1e6,
                     size_MB=(OUT / f"{name}_best.pth").stat().st_size / 1e6,
                     train_min=train_secs[name] / 60, ms_per_img=ms))
    if "test_other" in dl:   # images from a different source than the main test set
        _, acc_o, yo, po = run_epoch(m, dl["test_other"])
        rows[-1]["test_other_acc"] = acc_o
        rows[-1]["test_other_f1"] = precision_recall_fscore_support(
            yo, po.argmax(1), average="macro", zero_division=0)[2]
    print(f"\n=== {name} ===")
    print(classification_report(ys, pred, labels=range(len(CLASSES)), target_names=CLASSES,
                                digits=4, zero_division=0))
results = pd.DataFrame(rows).set_index("model").round(4)
results.to_csv(OUT / "results.csv")
results

# %%
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
for ax, (name, (ys, ps)) in zip(np.atleast_1d(axes), preds.items()):
    cm = confusion_matrix(ys, ps.argmax(1), labels=range(len(CLASSES)))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(name)
plt.tight_layout(); plt.savefig(OUT / "confusion_matrices.png", dpi=120); plt.show()

# %% [markdown]
# ## 8. Grad-CAM: where does the model look?
# Red = regions that pushed the model toward its prediction. For a trustworthy model the red
# areas should sit on the rust pustules, not on the background.

# %%
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out
        if out.requires_grad:  # skip during torch.no_grad() inference
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x):
        self.model.eval()
        logits = self.model(x)
        cls = logits.argmax(1)
        self.model.zero_grad()
        logits[torch.arange(len(x)), cls].sum().backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        lo, hi = cam.amin((1, 2), keepdim=True), cam.amax((1, 2), keepdim=True)
        return ((cam - lo) / (hi - lo + 1e-8)).detach().cpu(), logits.softmax(1).detach().cpu()


test_ds = ds["test"].dataset if isinstance(ds["test"], Subset) else ds["test"]
cam_layers = {"small_cnn": cnn.features[-1], "efficientnet_b0": eff.bn2, "mobilenetv3_small": mob.blocks[-1]}

for name, layer in cam_layers.items():
    cam = GradCAM(models[name], layer)
    ys, ps = preds[name]
    wrong = list(np.nonzero(ps.argmax(1) != ys)[0][:4])
    picks = [i for c in range(len(CLASSES)) for i in np.nonzero(ys == c)[0][:4]][:8 - len(wrong)] + wrong
    idx = [ds["test"].indices[i] if isinstance(ds["test"], Subset) else i for i in picks]
    x = torch.stack([test_ds[i][0] for i in idx]).to(DEVICE)
    heat, prob = cam(x)
    fig, axes = plt.subplots(2, len(idx), figsize=(2.6 * len(idx), 5.6), squeeze=False)
    for k, i in enumerate(idx):
        img, true = denorm(x[k].cpu()), test_ds[i][1]
        p = int(prob[k].argmax())
        axes[0, k].imshow(img); axes[0, k].axis("off")
        axes[0, k].set_title(f"true: {CLASSES[true]}", fontsize=8)
        axes[1, k].imshow(img); axes[1, k].imshow(heat[k], cmap="jet", alpha=0.45); axes[1, k].axis("off")
        axes[1, k].set_title(f"pred: {CLASSES[p]} {prob[k, p]:.2f}", fontsize=8,
                             color="green" if p == true else "red")
    plt.suptitle(f"{name} Grad-CAM (bottom row: red title = misclassified)")
    plt.tight_layout(); plt.savefig(OUT / f"gradcam_{name}.png", dpi=120); plt.show()

# %% [markdown]
# ## 9. Predict a single image
# Use this for the demo: give it any leaf photo path.

# %%
def predict(path, model=eff):
    model.eval()
    x = eval_tf(Image.open(path).convert("RGB"))[None].to(DEVICE)
    with torch.no_grad():
        p = model(x).softmax(1)[0].cpu()
    return {c: round(float(v), 4) for c, v in zip(CLASSES, p)}


example = test_ds.samples[0][0]
print(example, "->", predict(example))
json.dump({"classes": CLASSES, "img_size": IMG, "mean": MEAN, "std": STD},
          open(OUT / "model_info.json", "w"), indent=1)
print("saved outputs:", sorted(p.name for p in OUT.iterdir()))

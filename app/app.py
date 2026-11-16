r"""Step 10: Corn Common Rust detector, a Gradio demo app.

Run:  .\.venv\Scripts\python.exe app\app.py      then open http://127.0.0.1:7860
Uses the run-2 weights in results/run2/ (trained after removing the black-background shortcut).
"""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from scipy.ndimage import distance_transform_cdt
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "run3"   # run 3 = trained on two corn sources (nafishamoin + BCDD)
INFO = json.loads((RUN / "model_info.json").read_text())
CLASSES = INFO["classes"]
IMG = INFO["img_size"]
BLACK_THR, MASK_RES = 20, 128
UNCERTAIN = 0.75  # below this top-class probability, tell the user the result is unreliable

TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMG), transforms.ToTensor(),
                         transforms.Normalize(INFO["mean"], INFO["std"])])
VIEW = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMG)])


class SmallCNN(nn.Module):
    """Same architecture as in notebooks/train_corn_rust.py (needed to load the weights)."""

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


def _load(model, file):
    model.load_state_dict(torch.load(RUN / file, map_location="cpu"))
    return model.eval()


def load_models():
    cnn = _load(SmallCNN(len(CLASSES)), "small_cnn_best.pth")
    eff = _load(timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASSES)),
                "efficientnet_b0_best.pth")
    mob = _load(timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=len(CLASSES)),
                "mobilenetv3_small_best.pth")
    res = {}
    if (RUN / "results.csv").exists():
        import csv
        res = {r["model"]: r for r in csv.DictReader(open(RUN / "results.csv"))}

    def name(key, nice):
        r = res.get(key)
        return f"{nice} ({float(r['test_acc']):.1%}, other source {float(r['test_other_acc']):.1%})" \
            if r and r.get("test_other_acc") else nice

    # name -> list of (model, Grad-CAM layer); an ensemble averages its members, Grad-CAM from the first
    return {"Ensemble: SmallCNN + MobileNetV3 (recommended)": [(cnn, cnn.features[-1]), (mob, mob.blocks[-1])],
            name("mobilenetv3_small", "MobileNetV3-Small"): [(mob, mob.blocks[-1])],
            name("small_cnn", "SmallCNN"): [(cnn, cnn.features[-1])],
            name("efficientnet_b0", "EfficientNet-B0"): [(eff, eff.bn2)]}


MODELS = load_models()

# A published corn model (not trained by us), for comparison. Downloaded from Hugging Face on first run.
PUBLISHED_REPO = "Shaq2/chashibhai-corn-disease-cls"
PUBLISHED_NAME = "Published: corn YOLO classifier by Shaq2 (4 corn classes, not ours)"
PUBLISHED_LABELS = {"Common_Rust": "Common_rust", "Healthy": "Healthy",
                    "Gray_Leaf_Spot": "Gray_leaf_spot", "Northern_Leaf_Blight": "Northern_leaf_blight"}


def load_published():
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        return YOLO(hf_hub_download(PUBLISHED_REPO, "best.pt"))
    except Exception as e:  # no internet / ultralytics missing: the app still works with our models
        print(f"published model unavailable: {e!r}")
        return None


PUBLISHED = None if __name__.startswith("app_core") else load_published()
if PUBLISHED is not None:
    MODELS[PUBLISHED_NAME] = []


def classify_published(crop):
    r = PUBLISHED.predict(crop, imgsz=640, verbose=False)[0]
    return {PUBLISHED_LABELS.get(PUBLISHED.names[i], PUBLISHED.names[i]): float(p)
            for i, p in enumerate(r.probs.data)}


def classify(crop, model_name):
    """Average class probabilities over the chosen model(s); Grad-CAM from the first member."""
    members = MODELS[model_name]
    x = TF(crop)[None]
    probs, cam = gradcam(*members[0], x)
    with torch.no_grad():
        for model, _ in members[1:]:
            probs = probs + model(x).softmax(1)[0]
    return probs / len(members), cam


def views(img, frac=0.6):
    """The full image plus 5 zoomed square crops (center + 4 corners). Training photos are leaf
    close-ups, so a whole-leaf photo often matches better once zoomed in."""
    w, h = img.size
    s = int(min(w, h) * frac)
    pts = [((w - s) // 2, (h - s) // 2), (0, 0), (w - s, 0), (0, h - s), (w - s, h - s)]
    return [img] + [img.crop((x, y, x + s, y + s)) for x, y in pts]


def load_gate():
    """'Is this a supported corn leaf?' gate (v2) built by scripts/build_ood_gate.py."""
    f = RUN / "ood_gate.json"
    if not f.exists():
        return None
    cfg = json.loads(f.read_text(encoding="utf-8-sig"))
    if cfg.get("version") != 2:
        return None
    backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).eval()
    banks = {g: b.float() for g, b in torch.load(RUN / "ood_bank.pt").items()}
    return backbone, banks, cfg


GATE = None if __name__.startswith("app_core") else load_gate()

REJECT_MSG = {
    "corn_other": ("**This looks like a corn leaf with a disease this model doesn't cover** "
                   "(e.g. gray leaf spot or northern leaf blight). It only knows common rust vs healthy."),
    "other_crop": "**This looks like a leaf of another crop, not corn.** The model only knows corn leaves.",
    "not_leaf": "**This doesn't look like a plant leaf.** Upload a clear photo of a corn leaf.",
}


@torch.no_grad()
def gate_check(img):
    """Returns (group, best view or None). Accepted if any view is closest to the 'corn' group."""
    backbone, banks, cfg = GATE
    vs = views(img) if cfg.get("multiview") else [img]   # multi-view let junk through in eval
    f = F.normalize(backbone(torch.stack([TF(v) for v in vs])), dim=1)
    s = torch.stack([(f @ banks[g].T).topk(cfg["k"], dim=1).values.mean(1) for g in cfg["groups"]], 1)
    margin = s[:, 0] - s[:, 1:].max(1).values
    best = int(margin.argmax())
    if margin[best] >= 0:
        return "corn", vs[best]
    return cfg["groups"][1 + int(s[0, 1:].argmax())], None


def leaf_crop(img):
    """Same preprocessing as training: crop to the largest all-leaf square if there is a black background."""
    w, h = img.size
    s = MASK_RES / max(w, h)
    a = np.asarray(img.resize((max(1, round(w * s)), max(1, round(h * s)))))
    mask = np.pad(a.max(axis=2) >= BLACK_THR, 1)
    if mask[1:-1, 1:-1].all():
        return img
    d = distance_transform_cdt(mask, metric="chessboard")[1:-1, 1:-1]
    best = d.max()
    ys, xs = np.nonzero(d == best)
    k = np.argmin((ys - d.shape[0] / 2) ** 2 + (xs - d.shape[1] / 2) ** 2)
    side = int(2 * ((best - 1) / s + 0.5 / s) * 0.92)
    left = int(np.clip((xs[k] + 0.5) / s - side / 2, 0, w - side))
    top = int(np.clip((ys[k] + 0.5) / s - side / 2, 0, h - side))
    return img.crop((left, top, left + side, top + side)) if side > 16 else img


def gradcam(model, layer, x):
    store = {}

    def hook(module, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(hook)
    try:
        logits = model(x)
        model.zero_grad()
        logits[0, int(logits.argmax(1))].backward()
    finally:
        handle.remove()
    w = store["grad"].mean((2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return logits.softmax(1)[0].detach(), cam.detach().numpy()


def predict(img, model_name):
    if img is None:
        return None, None, "Upload a corn leaf photo first."
    img = img.convert("RGB")
    crop = leaf_crop(img)
    published = model_name == PUBLISHED_NAME
    if GATE is not None:
        group, view = gate_check(crop)
        # the published model knows other corn diseases, so let those through for it
        if group != "corn" and not (published and group == "corn_other"):
            return None, None, REJECT_MSG[group] + " It won't guess."
        crop = view if view is not None else crop
    if published:  # published models expect the full photo, not our leaf-only crop
        conf, overlay = classify_published(img), None
    else:
        probs, cam = classify(crop, model_name)
        view = np.asarray(VIEW(crop), dtype=np.float32) / 255
        overlay = Image.fromarray((np.clip(0.55 * view + 0.45 * colormaps["jet"](cam)[..., :3], 0, 1) * 255)
                                  .astype(np.uint8))
        conf = {c: float(p) for c, p in zip(CLASSES, probs)}
    top = max(conf, key=conf.get)
    if conf[top] < UNCERTAIN:
        note = (f"**Uncertain ({conf[top]:.0%}).** Try a closer, sharper photo of a single corn leaf "
                "in good light.")
    elif top == "Common_rust":
        note = (f"**Common rust detected ({conf[top]:.0%}).** Look for small brown, powdery pustules on "
                "both leaf surfaces. Consult a local agriculture officer about resistant varieties or "
                "fungicide timing.")
    elif top == "Healthy":
        note = f"**Leaf looks healthy ({conf[top]:.0%}).** No rust pustules found."
    else:
        note = (f"**Possible {top.replace('_', ' ')} ({conf[top]:.0%}).** Only the published model covers "
                "this disease; our models know common rust vs healthy only.")
    if published:
        note += "\n\n*Published model (not trained by us), so no Grad-CAM heatmap is available.*"
    return conf, overlay, note


def examples():
    test = ROOT / "data" / "final" / "corn_rust" / "test"
    picks = [p for c in CLASSES for p in sorted((test / c).glob("*.jpg"))[:3]] if test.exists() else []
    return [[str(p), next(iter(MODELS))] for p in picks]


with gr.Blocks() as demo:
    gr.Markdown("# Corn Common Rust Detector\n"
                "Upload a photo of a **corn (maize) leaf**. The model says whether it has **common rust** "
                "or is **healthy**, and the heatmap shows where it looked (red = most important).")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Corn leaf photo", height=360)
            model_dd = gr.Dropdown(list(MODELS), value=next(iter(MODELS)), label="Model")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=4, label="Prediction")
            out_note = gr.Markdown()
            out_cam = gr.Image(label="Grad-CAM: where the model looked", height=300)
    btn.click(predict, [inp, model_dd], [out_label, out_cam, out_note])
    if examples():
        gr.Examples(examples(), inputs=[inp, model_dd], label="Test-set examples (never seen in training)")
    gr.Markdown("*A gate checks first that the photo resembles our corn-leaf training data. On held-out "
                "tests it rejected 100% of non-plant photos and 99.7% of other crops' leaves, and accepted "
                "98% of real corn leaves. Corn leaves with **other** diseases slip through about 21% of the "
                "time and then get labelled rust or healthy. Course project, not a substitute for expert "
                "diagnosis.*")

if __name__ == "__main__":
    demo.launch()

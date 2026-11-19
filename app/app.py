r"""Corn common rust detector: web demo.

Run:  .\.venv\Scripts\python.exe app\app.py      then open http://127.0.0.1:7860
Serves a one-page frontend (app/static/) and a small JSON API (/api/models, /api/analyze).
Uses the models in results/live/ (see results/live/SOURCES.json), or the folder named by the
CORNRUST_RUN environment variable. Get the weights with scripts/download_models.py.
The model functions below are also imported by scripts/eval_app.py, eval_published.py and
build_ood_gate.py, and by deploy/modal_api.py, so keep their names and behaviour.
"""
import base64
import io
import json
import os
import threading
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_cdt
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
# model folder: weights, model_info.json, results.csv and the gate files (ood_gate.json, ood_bank.pt)
RUN = Path(os.environ.get("CORNRUST_RUN", ROOT / "results" / "live")).resolve()
INFO = json.loads((RUN / "model_info.json").read_text())
CLASSES = INFO["classes"]
IMG = INFO["img_size"]
BLACK_THR, MASK_RES = 20, 128
UNCERTAIN = 0.75  # below this top-class probability, tell the user the result is unreliable
MAX_BYTES = 15 * 1024 * 1024

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


# ---------------------------------------------------------------- web app

def nice(label):
    return label.replace("_", " ").capitalize()


def data_url(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def heat_overlay(crop, cam):
    view = np.asarray(VIEW(crop), dtype=np.float32) / 255
    return Image.fromarray((np.clip(0.55 * view + 0.45 * colormaps["jet"](cam)[..., :3], 0, 1) * 255)
                           .astype(np.uint8))


def analyze(img, model_name):
    """Full pipeline for one photo: leaf crop -> gate -> classifier -> Grad-CAM. Returns a JSON-ready dict."""
    img = ImageOps.exif_transpose(img).convert("RGB")   # phone photos carry their rotation in EXIF
    crop = leaf_crop(img)
    published = model_name == PUBLISHED_NAME
    if GATE is not None:
        group, view = gate_check(crop)
        # the published model knows other corn diseases, so let those through for it
        if group != "corn" and not (published and group == "corn_other"):
            return {"status": "rejected", "verdict": "rejected", "title": "This doesn't look like a corn leaf.",
                    "detail": "Upload a clear, close-up photo of a single corn (maize) leaf. "
                              "This detector checks for common rust and healthy leaves."}
        crop = view if view is not None else crop
    if published:  # published models expect the full photo, not our leaf-only crop
        conf, photo, heat = classify_published(img), VIEW(img), None
    else:
        probs, cam = classify(crop, model_name)
        conf = {c: float(p) for c, p in zip(CLASSES, probs)}
        photo, heat = VIEW(crop), heat_overlay(crop, cam)
    top = max(conf, key=conf.get)
    if conf[top] < UNCERTAIN:
        verdict, title = "uncertain", "Uncertain"
        detail = "Try a closer, sharper photo of a single corn leaf in good light."
    elif top == "Common_rust":
        verdict, title = "rust", "Common rust"
        detail = ("Look for small brown, powdery pustules on both leaf surfaces. Ask a local agriculture "
                  "officer about resistant varieties or fungicide timing.")
    elif top == "Healthy":
        verdict, title, detail = "healthy", "Healthy", "No rust pustules found."
    else:
        verdict, title = "other", f"Possible {nice(top).lower()}"
        detail = "Only the published model covers this disease. Our models know common rust and healthy only."
    return {"status": "ok", "verdict": verdict, "title": title, "detail": detail,
            "label": nice(top), "confidence": conf[top],
            "scores": [{"label": nice(c), "value": v} for c, v in sorted(conf.items(), key=lambda kv: -kv[1])],
            "photo": data_url(photo), "heatmap": data_url(heat) if heat is not None else None}


def model_catalog():
    items = []
    for key in MODELS:
        if key == PUBLISHED_NAME:
            label, detail = "Published YOLO", "Shaq2's corn classifier from Hugging Face · 4 classes · not ours"
        elif key.startswith("Ensemble"):
            label, detail = "Ensemble", "SmallCNN + MobileNetV3, averaged · recommended"
        else:
            label = key.split(" (")[0]
            acc = key[len(label) + 2:-1] if " (" in key else ""
            detail = "Test " + acc.replace(", other source", " · other source") if acc else ""
        items.append({"id": key, "label": label, "detail": detail})
    return items


def create_api(with_page=True):
    """The FastAPI app: /api/models and /api/analyze, plus the page itself when with_page is True
    (the Modal deployment serves only the API; nginx serves the page)."""
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    here = Path(__file__).resolve().parent
    api = FastAPI(title="Corn rust detector", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()   # Grad-CAM hooks live on shared models: one analysis at a time

    @api.get("/api/models")
    def models():
        return {"models": model_catalog(), "default": next(iter(MODELS))}

    @api.post("/api/analyze")
    def analyze_photo(image: UploadFile = File(...), model: str = Form("")):
        data = image.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            return JSONResponse({"status": "error", "detail": "That file is over 15 MB."}, 413)
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception:
            return JSONResponse({"status": "error",
                                 "detail": "That file isn't an image we can read. Try a JPG, PNG or WebP."}, 400)
        with lock:
            return analyze(img, model if model in MODELS else next(iter(MODELS)))

    if with_page:
        api.mount("/static", StaticFiles(directory=here / "static"), name="static")
        api.mount("/assets", StaticFiles(directory=here / "assets"), name="assets")

        @api.get("/")
        def index():
            return FileResponse(here / "static" / "index.html")

    return api


def serve(host="127.0.0.1", port=7860):
    import uvicorn

    api = create_api()
    print(f"Corn rust detector running at http://{host}:{port} (models: {RUN})", flush=True)
    uvicorn.run(api, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()

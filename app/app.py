r"""Step 10: Corn Common Rust detector, a Gradio demo app.

Run:  .\.venv\Scripts\python.exe app\app.py      then open http://127.0.0.1:7860
Uses the run-3 weights in results/run3/.
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

UNSUPPORTED_IMAGE_MSG = (
    "### This doesn't look like a corn leaf.\n\n"
    "Please upload a clear, close-up photo of a single corn (maize) leaf. "
    "This detector checks for common rust and healthy corn leaves."
)
# A nearest-reference group cannot establish that an arbitrary rejected image
# really is a leaf or another crop. Keep rejection messages neutral.
REJECT_MSG = dict.fromkeys(("corn_other", "other_crop", "not_leaf"), UNSUPPORTED_IMAGE_MSG)


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
            return None, None, REJECT_MSG.get(group, UNSUPPORTED_IMAGE_MSG)
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


READY_NOTE = "### A leaf check, whenever you're ready.\n\nUpload one corn leaf, pick a model, and select **Analyze leaf**. Your result and visual explanation will appear here."


def model_description(name):
    if name.startswith("Ensemble"):
        return "SmallCNN and MobileNetV3 working together. A good place to start."
    if name.startswith("MobileNet"):
        return "A compact model for an alternative check."
    if name.startswith("SmallCNN"):
        return "Our smallest classifier, trained specifically for this task."
    if name == PUBLISHED_NAME:
        return "An independent corn classifier with four classes. No heatmap is available."
    return "An alternative classifier for comparing your result."


def predict_ui(img, model_name):
    conf, overlay, note = predict(img, model_name)
    return (gr.update(value=conf, visible=conf is not None),
            gr.update(value=overlay, visible=overlay is not None), note)


def reset_results():
    return gr.update(value=None, visible=False), gr.update(value=None, visible=False), READY_NOTE


def photo_changed(img):
    return (*reset_results(), gr.update(interactive=img is not None))


def model_changed(name):
    return model_description(name), *reset_results()


def clear_photo():
    return None, *reset_results(), gr.update(interactive=False)


def sample_rust():
    with Image.open(Path(__file__).with_name("assets") / "sample-rust.png") as image:
        return image.convert("RGB")


def sample_healthy():
    with Image.open(Path(__file__).with_name("assets") / "sample-healthy.png") as image:
        return image.convert("RGB")


MODEL_CHOICES = []
for name in MODELS:
    label = ("Ensemble · Recommended" if name.startswith("Ensemble") else
             "MobileNetV3" if name.startswith("MobileNet") else
             "SmallCNN" if name.startswith("SmallCNN") else
             "Published corn model" if name == PUBLISHED_NAME else "EfficientNet B0")
    MODEL_CHOICES.append((label, name))

THEME = gr.themes.Soft(primary_hue="emerald", secondary_hue="teal", neutral_hue="slate",
                       font=["Segoe UI", "Arial", "sans-serif"], radius_size="lg").set(
    body_background_fill="#f4f6f1", body_background_fill_dark="#101a16",
    block_background_fill="#ffffff", block_background_fill_dark="#18251e",
    block_border_color="#dfe6dc", block_border_color_dark="#304438",
    button_primary_background_fill="#176143", button_primary_background_fill_dark="#9cdbb4",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#102b1b",
    button_primary_background_fill_hover="#104c34", button_primary_background_fill_hover_dark="#b2e8c5",
)
CSS = "\n".join(Path(__file__).with_name(name).read_text(encoding="utf-8")
                for name in ("ui.css", "scene.css"))
MOTION_JS = Path(__file__).with_name("ui.js").read_text(encoding="utf-8")
SCENE_HTML = Path(__file__).with_name("scene.html").read_text(encoding="utf-8")

with gr.Blocks(title="CornCare | Corn leaf health", analytics_enabled=False) as demo:
    gr.HTML(SCENE_HTML, elem_id="corncare-scene")
    with gr.Row(elem_id="workspace"):
        with gr.Column(elem_id="upload-panel", min_width=310):
            gr.Markdown("## Bring a leaf. We'll look closer.\nOne corn leaf, in focus and in good light.", elem_classes="section-heading")
            inp = gr.Image(type="pil", label="Corn leaf photo", height=310, sources=["upload", "webcam", "clipboard"], elem_id="leaf-upload")
            gr.Markdown("No photo handy? Try a presentation sample.", elem_classes="cc-sample-caption")
            with gr.Row():
                rust_sample = gr.Button("Rust sample", size="sm", elem_classes="cc-sample-btn")
                healthy_sample = gr.Button("Healthy sample", size="sm", elem_classes="cc-sample-btn")
            model_dd = gr.Radio(MODEL_CHOICES, value=next(iter(MODELS)), label="Choose a model", elem_id="model-choice")
            model_help = gr.Markdown(model_description(next(iter(MODELS))), elem_id="model-help")
            with gr.Row():
                btn = gr.Button("Analyze leaf", variant="primary", interactive=False, scale=3, elem_id="analyze")
                clear_btn = gr.Button("Clear", scale=1, elem_id="clear")
        with gr.Column(elem_id="results-panel", min_width=310):
            gr.Markdown("## The leaf report\nModel scores and the regions that influenced the result.", elem_classes="section-heading")
            out_note = gr.Markdown(READY_NOTE, elem_id="result-note")
            out_label = gr.Label(num_top_classes=4, label="Model scores", visible=False, elem_id="prediction")
            out_cam = gr.Image(label="Heatmap · regions that influenced the result", height=300, visible=False, elem_id="heatmap")
            with gr.Accordion("Tips for a useful check", open=False):
                gr.Markdown("Use one corn leaf in good light. Keep it in focus and fill most of the photo with the leaf.\n\n"
                            "The detector covers **common rust and healthy** leaves. Heatmap colors show influence on the prediction, not disease severity.")
    analysis = btn.click(predict_ui, [inp, model_dd], [out_label, out_cam, out_note],
                         concurrency_limit=1, api_name="analyze",
                         js="(image, model) => { window.cornCareMotion?.begin(); return [image, model]; }")
    analysis.success(fn=None, inputs=[out_note], outputs=[], queue=False, api_name=False,
                     js="(note) => { window.cornCareMotion?.finish(note); return []; }")
    analysis.failure(fn=None, inputs=[], outputs=[], queue=False, api_name=False,
                     js="() => { window.cornCareMotion?.fail(); return []; }")
    reset_js = "() => { window.cornCareMotion?.reset(); return []; }"
    inp.change(photo_changed, inp, [out_label, out_cam, out_note, btn], queue=False, api_name=False,
               cancels=[analysis]).then(fn=None, outputs=[], queue=False, api_name=False, js=reset_js)
    model_dd.change(model_changed, model_dd, [model_help, out_label, out_cam, out_note], queue=False, api_name=False,
                    cancels=[analysis]).then(fn=None, outputs=[], queue=False, api_name=False, js=reset_js)
    clear_btn.click(clear_photo, outputs=[inp, out_label, out_cam, out_note, btn], queue=False, api_name=False,
                    cancels=[analysis]).then(fn=None, outputs=[], queue=False, api_name=False, js=reset_js)
    rust_sample.click(sample_rust, outputs=inp, queue=False, api_name=False)
    healthy_sample.click(sample_healthy, outputs=inp, queue=False, api_name=False)
    if examples():
        gr.Examples(examples(), inputs=[inp, model_dd], label="Try a test photo")
    gr.Markdown("A course-project prototype for close-up corn leaf photos. The input check and predictions can be wrong. "
                "For a field diagnosis, consult an agriculture professional.", elem_id="footer-note")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=MOTION_JS, footer_links=[])

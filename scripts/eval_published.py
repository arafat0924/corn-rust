r"""Compare our models with published, NOT fine-tuned models on the same test images.

Run: .\.venv\Scripts\python.exe scripts\eval_published.py
Published models (used exactly as released, weights downloaded from Hugging Face):
  rodynaemad/plant-disease-dinov2-small  DINOv2-small, 38 PlantVillage classes (CC BY-SA 4.0)
  neghani/plantvillage-resnet18          ResNet-18, 38 PlantVillage classes (Apache-2.0)
  Shaq2/chashibhai-corn-disease-cls      YOLO classifier, 4 corn classes, ~11 MB (MIT)
Inputs:
  original : the photo as-is (what published models expect)  <- primary for published models
  leaf_crop: our leaf-only square crop (our pipeline's input)
Scoring:
  as_published : argmax over all the model's classes; correct only if it is the right corn class
  restricted   : compare only its corn-rust vs corn-healthy scores (fair binary comparison)
Caveat: test_other is BCDD corn = PlantVillage images, which the PlantVillage models were TRAINED on.
Output: reports/published_comparison.csv
"""
import importlib.util
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("app_core", ROOT / "app" / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

SETS = {"test (main source)": ROOT / "data/final/corn_rust_v3/test",
        "test_other (BCDD = PlantVillage)": ROOT / "data/final/corn_rust_v3/test_other"}
HF_MODELS = ["rodynaemad/plant-disease-dinov2-small", "neghani/plantvillage-resnet18"]
YOLO_REPO = "Shaq2/chashibhai-corn-disease-cls"
OUR = {"Ours: Ensemble SmallCNN+MobileNetV3": "Ensemble", "Ours: MobileNetV3-Small": "MobileNetV3",
       "Ours: SmallCNN": "SmallCNN"}


def corn_ids(id2label):
    rust = next(i for i, l in id2label.items() if "corn" in l.lower() and "rust" in l.lower())
    healthy = next(i for i, l in id2label.items() if "corn" in l.lower() and "healthy" in l.lower())
    return {"Common_rust": int(rust), "Healthy": int(healthy)}


def score(logit_rows, ids, truth):
    pub = res = 0
    for lg, t in zip(logit_rows, truth):
        pub += int(torch.as_tensor(lg).argmax()) == ids[t]
        res += ("Common_rust" if lg[ids["Common_rust"]] > lg[ids["Healthy"]] else "Healthy") == t
    return pub / len(truth), res / len(truth)


def main():
    data = {}
    for name, folder in SETS.items():
        items = sorted(folder.rglob("*.jpg"))
        orig = [Image.open(p).convert("RGB") for p in items]
        data[name] = {"original": orig, "leaf_crop": [app.leaf_crop(im) for im in orig],
                      "truth": [p.parent.name for p in items]}
    rows = []

    def add(model, set_name, inp, pub, res, n):
        rows.append(dict(model=model, set=set_name, input=inp, n=n, acc_as_published=pub, acc_restricted=res))
        print(rows[-1])

    from transformers import AutoImageProcessor, AutoModelForImageClassification
    for repo in HF_MODELS:
        proc = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForImageClassification.from_pretrained(repo).eval()
        ids = corn_ids(model.config.id2label)
        for set_name, d in data.items():
            for inp in ("original", "leaf_crop"):
                logits = []
                for i in range(0, len(d[inp]), 32):
                    with torch.no_grad():
                        logits += list(model(**proc(images=d[inp][i:i + 32], return_tensors="pt")).logits)
                add(f"Published: {repo.split('/')[1]}", set_name, inp, *score(logits, ids, d["truth"]),
                    len(d["truth"]))

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    yolo = YOLO(hf_hub_download(YOLO_REPO, "best.pt"))
    ids = corn_ids({i: f"corn {n}" for i, n in yolo.names.items()})
    for set_name, d in data.items():
        for inp in ("original", "leaf_crop"):
            probs = [r.probs.data for r in yolo.predict(d[inp], imgsz=640, verbose=False, batch=32)]
            add("Published: corn YOLO (Shaq2)", set_name, inp, *score(probs, ids, d["truth"]), len(d["truth"]))

    for label, key in OUR.items():
        name = next(k for k in app.MODELS if k.startswith(key))
        for set_name, d in data.items():
            ok = sum(app.CLASSES[int(app.classify(im, name)[0].argmax())] == t
                     for im, t in zip(d["leaf_crop"], d["truth"])) / len(d["truth"])
            add(label, set_name, "leaf_crop", ok, ok, len(d["truth"]))

    df = pd.DataFrame(rows)
    df.round(4).to_csv(ROOT / "reports" / "published_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

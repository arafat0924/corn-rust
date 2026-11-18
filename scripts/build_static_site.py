r"""Build a static copy of the demo page into site/, for plain web hosting with no Python backend.

Run: .\.venv\Scripts\python.exe scripts\build_static_site.py
The two sample leaves (app/assets/sample-*.png) are analysed now with every model, and the results
(scores, verdict, photo and Grad-CAM heatmap) are saved as JSON in site/demo/. When the page finds no
/api server, app.js switches to demo mode and shows these saved results. Uploads need the real server.
Upload the CONTENTS of site/ to the web root (see the deployment steps in the README).
"""
import importlib.util
import json
import re
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
OUT = ROOT / "site"
SAMPLES = ("rust", "healthy")

spec = importlib.util.spec_from_file_location("app", APP / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)   # loads the models and the gate exactly as the live app does


def slug(label):
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(APP / "static", OUT / "static")
    shutil.move(OUT / "static" / "index.html", OUT / "index.html")   # served at /
    shutil.copytree(APP / "assets", OUT / "assets")
    demo = OUT / "demo"
    demo.mkdir()

    catalog = [m for m in app.model_catalog() if m["id"] != app.PUBLISHED_NAME]
    for m in catalog:
        m["slug"] = slug(m["label"])
        for sample in SAMPLES:
            with Image.open(APP / "assets" / f"sample-{sample}.png") as im:
                res = app.analyze(im, m["id"])
            (demo / f"{sample}-{m['slug']}.json").write_text(json.dumps(res), encoding="utf-8")
            print(f"  {sample:8s} {m['label']:18s} -> {res['title']} ({res.get('confidence', 0):.1%})")
    (demo / "models.json").write_text(json.dumps({"models": catalog, "default": catalog[0]["id"]}, indent=1),
                                      encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"static site written to {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

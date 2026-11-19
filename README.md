# plant-disease

Deep-learning detector for **corn (maize) common rust** from leaf photos, built as a course project with a
Bangladesh focus. A binary classifier (Common rust vs Healthy) with a one-page web demo, Grad-CAM explanations,
and an "is this a corn leaf?" gate that refuses to guess on unrelated images.

**Live demo: https://cornrust.arafatcodes.dev** (upload a close-up of one corn leaf, or try a sample).

## Results (the models the app serves, held-out test sets)

| Model | Params | Main test acc | Other-source test acc | Trained in |
|---|---|---|---|---|
| SmallCNN (from scratch) | 0.39M | 99.7% | 93.5% | run 3 |
| MobileNetV3-Small (ImageNet, fine-tuned) | 1.5M | 99.4% | 98.9% | run 3 |
| EfficientNet-B0 (ImageNet, fine-tuned) | 4.0M | 99.4% | 96.8% | run 4 |
| **App ensemble** (SmallCNN + MobileNetV3), incl. gate | n/a | **99.4%** of all images (0.6% rejected, 100% of accepted correct) | **97.9%** (none rejected) |

The ensemble row comes from `scripts/eval_app.py` (whole app: leaf crop, gate, model; `reports/app_eval.csv`).
"Other source" = BCDD (PlantVillage-style) corn images, a different photo source than the main set.
`results/live/SOURCES.json` lists which run each served file comes from.

## What we learned along the way
1. **Shortcut learning (run 1).** The first models hit 100% because 99.7% of rust photos had a black cut-out
   background and healthy photos didn't. Grad-CAM showed the models looking at the background.
   Fix: crop every image to the largest all-leaf square (notebook section 0).
2. **Single-source overfitting (run 2).** After the fix, accuracy on corn from another source was only 65.6%.
   Fix: train on two corn sources (run 3), giving 93.5–98.9%.
3. **Out-of-distribution inputs.** A 2-class model labels anything (e.g. a logo) as rust or healthy.
   Fix: a kNN gate on ImageNet features that compares an image with 4 reference groups (corn,
   corn with other diseases, other crops, non-plant). Its first version still let flat graphics through
   (a blue logo, an abstract wallpaper), so gate v3 adds synthetic graphics to the non-plant group.
   Held-out results (`reports/ood/ood_eval.csv`): corn test leaves accepted 99.4% (other source 100%);
   rejected: other corn diseases 97.0%, other crops 97.3%, non-plant photos 97.4%, synthetic graphics 100%.
4. **Reproducible retraining (run 4).** The whole data pipeline was re-run from the original downloads on
   Linux and reproduced the run-3 split image for image (`scripts/compare_runs.py` checks this). The models
   were retrained on a cloud GPU; on the same test sets each model was kept or replaced by whichever did
   better through the whole app: EfficientNet-B0 improved (other source 91.7% -> 97.9% end to end), SmallCNN
   and MobileNetV3 did not, so the served ensemble keeps the run-3 versions.

## Project layout
```
scripts/            data pipeline (run in this order)
  download_mendeley.py, fetch_mendeley_all.ps1   download raw datasets
  explore_data.py   -> data/inventory.csv, reports/explore/
  apply_labels.py   -> unified labels via configs/label_map.csv
  clean_data.py     -> resize, drop tiny images, perceptual-hash de-duplication
  make_split.py     -> train/val/test (+ test_other) folders and zip
  build_ood_gate.py -> corn-leaf gate (feature banks + held-out evaluation)
  eval_app.py       -> end-to-end evaluation of the app pipeline
  compare_runs.py   -> new run vs current models: same split? which is better?
  build_static_site.py -> backend-free copy of the page with pre-computed sample results
  py_to_ipynb.py    -> builds the notebook from the .py source
  download_models.py -> fetches the trained models from the GitHub release
deploy/modal_pipeline.py      the whole pipeline on Modal (download, clean, split, GPU training, gate, eval)
deploy/modal_api.py           the live model API on Modal (scales to zero when idle)
notebooks/train_corn_rust.py  training notebook source (edit this, then regenerate the .ipynb)
notebooks/kaggle/             Kaggle kernel config (GPU, dataset input), the original training route
app/app.py                    web demo: model pipeline + FastAPI server
app/static/                   the demo page (index.html, style.css, app.js)
results/run1..4/              metrics, learning curves, confusion matrices, Grad-CAM per training run
results/live/                 the models the app serves (weights via scripts/download_models.py)
reports/                      exploration, cleaning, gate and app evaluation reports
```

## Quick start (use the trained models, no dataset or GPU needed)
```powershell
git clone https://github.com/arafat0924/corn-rust.git
cd corn-rust
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install timm fastapi uvicorn python-multipart pillow scipy matplotlib
.\.venv\Scripts\python.exe scripts\download_models.py   # ~40 MB of trained models from the v2.0 release
.\.venv\Scripts\python.exe app\app.py                    # open http://127.0.0.1:7860
```
The trained models are attached to the [v2.0 release](https://github.com/arafat0924/corn-rust/releases/tag/v2.0)
(`small_cnn_best.pth`, `mobilenetv3_small_best.pth`, `efficientnet_b0_best.pth`, `ood_bank.pt`). The first run
also downloads ImageNet weights for the gate (internet needed once). Runs on a normal CPU.

## How the live site works
nginx on a small VPS serves the page (`site/`, built by `scripts\build_static_site.py`) and forwards `/api/*`
to the model API on Modal (`deploy/modal_api.py`, CPU, scales to zero; the first check after a quiet spell
waits ~15 s while it wakes up). The two sample leaves use pre-computed results, so they are instant.
Uploads are switched on and off with `UPLOAD_ENABLED` in `app/static/app.js`.
```powershell
modal deploy deploy\modal_api.py                          # API (needs a logged-in Modal profile)
.\.venv\Scripts\python.exe scripts\build_static_site.py   # page -> site/
tar -czf $env:TEMP\cornrust-site.tgz -C site .
scp $env:TEMP\cornrust-site.tgz <user>@<server>:/tmp/
ssh <user>@<server> "rm -rf /var/www/cornrust/* && tar -xzf /tmp/cornrust-site.tgz -C /var/www/cornrust"
```

## Retraining (cloud, no local dataset needed)
`deploy/modal_pipeline.py` runs every step on Modal and keeps the data in a Modal Volume; see its docstring:
`prepare` (download the 5 datasets, explore, label, clean, split), `train` (T4 GPU), `gate`, `evaluate`.
Then copy `results/<run>` back, compare with `scripts\compare_runs.py`, and point `results/live` at the
winners. The original route (local pipeline + Kaggle notebook via `notebooks/kaggle`) still works too.

## Data sources
- [Bangladeshi Crops Disease Dataset](https://www.kaggle.com/datasets/nafishamoin/bangladeshi-crops-disease-dataset) (main corn source)
- [Bangladeshi Crops Disease Dataset (BCDD)](https://www.kaggle.com/datasets/musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd) (second corn source, CC BY 4.0)
- Mendeley Data: Dhan-Shomadhan, RiceLeafBD, Plant Leaf Freshness (BD vegetables), used for the gate's
  "other crops" group and for the planned multi-crop version

## Limitations
- Trained on close-up leaf photos. Whole-plant field photos are outside the training distribution.
- Only two classes. Corn leaves with other diseases are mostly rejected by the gate, but not always.
- BCDD is pre-augmented, so the other-source accuracy may be somewhat optimistic.
- Not tested on new field photos yet. Course project, not a substitute for expert diagnosis.

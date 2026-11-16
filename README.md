# plant-disease

Deep-learning detector for **corn (maize) common rust** from leaf photos, built as a course project with a
Bangladesh focus. A binary classifier (Common rust vs Healthy) with a Gradio demo, Grad-CAM explanations,
and an "is this a corn leaf?" gate that refuses to guess on unrelated images.

## Results (run 3, held-out test sets)

| Model | Params | Main test acc | Other-source test acc |
|---|---|---|---|
| SmallCNN (from scratch) | 0.39M | 99.7% | 93.5% |
| MobileNetV3-Small (ImageNet, fine-tuned) | 1.5M | 99.4% | 98.9% |
| EfficientNet-B0 (ImageNet, fine-tuned) | 4.0M | 98.5% | 90.3% |
| **App ensemble** (SmallCNN + MobileNetV3), incl. gate | n/a | 100% of accepted (1.5% rejected) | 97.9% of accepted (1.0% rejected) |

The ensemble row comes from `scripts/eval_app.py`. "Other source" = BCDD (PlantVillage-style) corn images,
a different photo source than the main set.

## What we learned along the way
1. **Shortcut learning (run 1).** The first models hit 100% because 99.7% of rust photos had a black cut-out
   background and healthy photos didn't. Grad-CAM showed the models looking at the background.
   Fix: crop every image to the largest all-leaf square (notebook section 0).
2. **Single-source overfitting (run 2).** After the fix, accuracy on corn from another source was only 65.6%.
   Fix: train on two corn sources (run 3), giving 93.5–98.9%.
3. **Out-of-distribution inputs.** A 2-class model labels anything (e.g. a logo) as rust or healthy.
   Fix: a kNN gate on ImageNet features that compares an image with 4 reference groups (corn,
   corn with other diseases, other crops, non-plant). Held-out rejection: non-plant 99%, other crops 98%,
   other corn diseases 97%.

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
  py_to_ipynb.py    -> builds the notebook from the .py source
  download_models.py -> fetches the trained models from the GitHub release
notebooks/train_corn_rust.py   training notebook source (edit this, then regenerate the .ipynb)
notebooks/kaggle/              Kaggle kernel config (GPU, dataset input)
app/app.py                     Gradio demo
results/run1..3/               metrics, learning curves, confusion matrices, Grad-CAM
reports/                       exploration, cleaning, gate and app evaluation reports
```

## Quick start (use the trained models, no dataset or GPU needed)
```powershell
git clone https://github.com/sfthanif/plant-disease.git
cd plant-disease
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install timm gradio pillow scipy matplotlib
.\.venv\Scripts\python.exe scripts\download_models.py   # 37 MB of trained models from the v1.0 release
.\.venv\Scripts\python.exe app\app.py                    # open http://127.0.0.1:7860
```
The trained models are attached to the [v1.0 release](https://github.com/sfthanif/plant-disease/releases/tag/v1.0)
(`small_cnn_best.pth`, `mobilenetv3_small_best.pth`, `efficientnet_b0_best.pth`, `ood_bank.pt`). The first run
also downloads ImageNet weights for the gate (internet needed once). Runs on a normal CPU.

## Retraining (needs the datasets)
Install the full dependencies (`imagehash pandas scikit-learn tqdm kaggle` as well), run the pipeline in
`scripts/`, then train with the Kaggle notebook (`kaggle kernels push -p notebooks\kaggle`), put the outputs
in `results/run3/` and rebuild the gate with `scripts\build_ood_gate.py`.

## Data sources
- [Bangladeshi Crops Disease Dataset](https://www.kaggle.com/datasets/nafishamoin/bangladeshi-crops-disease-dataset) (main corn source)
- [Bangladeshi Crops Disease Dataset (BCDD)](https://www.kaggle.com/datasets/musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd) (second corn source, CC BY 4.0)
- Mendeley Data: Dhan-Shomadhan, RiceLeafBD, Plant Leaf Freshness (BD vegetables), used for the gate's
  "other crops" group and for the planned multi-crop version

## Limitations
- Trained on close-up leaf photos. Whole-plant field photos are outside the training distribution.
- Only two classes. Corn leaves with other diseases are mostly rejected by the gate, but not always.
- BCDD is pre-augmented, so the other-source accuracy may be somewhat optimistic.
- Course project, not a substitute for expert diagnosis.

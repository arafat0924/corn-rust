r"""Rebuild the corn-rust data and retrain the models on Modal (cloud CPUs and a GPU).

Needs `pip install modal` and a logged-in Modal profile. Run from the project root, one stage at a
time. Every stage saves its outputs in the Modal Volume "cornrust-data", so stages can be re-run alone:

  modal run deploy/modal_pipeline.py::prepare --naf-url "<url>" --bcdd-url "<url>"
        download the 5 raw datasets, then explore -> labels -> clean -> split   (CPU)
  modal run deploy/modal_pipeline.py::train --run run4       train the 3 models (T4 GPU)
  modal run deploy/modal_pipeline.py::gate --run run4        build + evaluate the corn-leaf gate (CPU)
  modal run deploy/modal_pipeline.py::evaluate --run run4    whole-app evaluation, new vs run3 models
  modal volume get cornrust-data results/run4 results/      copy the results back

<url>: Kaggle's public download endpoint https://www.kaggle.com/api/v1/datasets/download/<owner>/<name>
redirects to a signed storage link (valid 3 days). Resolve it on a normal PC and pass the final link:
cloud machines are often refused by the endpoint itself, but not by the storage link.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
PROJ_S = "/root/proj"           # container paths: always POSIX, also when deploying from Windows
PROJ = Path(PROJ_S)
VOL = Path("/vol")
PY = sys.executable

app = modal.App("cornrust-pipeline")
vol = modal.Volume.from_name("cornrust-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torchvision==0.29.2", "timm==1.0.31", "numpy", "pandas", "scipy",
                 "scikit-learn", "matplotlib", "pillow", "imagehash", "tqdm")
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "scripts", f"{PROJ_S}/scripts", ignore=["__pycache__"])
    .add_local_dir(LOCAL / "configs", f"{PROJ_S}/configs")
    .add_local_file(LOCAL / "notebooks" / "train_corn_rust.py", f"{PROJ_S}/notebooks/train_corn_rust.py")
    .add_local_dir(LOCAL / "app", f"{PROJ_S}/app", ignore=["__pycache__"])
    .add_local_dir(LOCAL / "results" / "run3", f"{PROJ_S}/results/run3")
)
MENDELEY = [("dhan_shomadhan", "znsxdctwtt", 1), ("riceleafbd", "kx9rx8p2mz", 1), ("veg_leaf_bd", "n67gctmjyj", 3)]
SPLIT_TARGET = {("test", "Common_rust"): 172, ("test", "Healthy"): 167, ("test_other", "Common_rust"): 48,
                ("test_other", "Healthy"): 48, ("train", "Common_rust"): 1160, ("train", "Healthy"): 1137,
                ("val", "Common_rust"): 244, ("val", "Healthy"): 238}   # the original run-3 split


def sh(*cmd, env=None, cwd=PROJ, log=None):
    """Run a command, stream its output, optionally tee it into a log file; fail loudly."""
    print("$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env={**os.environ, **(env or {})},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    fh = open(log, "a", encoding="utf-8") if log else None
    for line in p.stdout:
        print(line, end="", flush=True)
        if fh:
            fh.write(line)
    if fh:
        fh.close()
    if p.wait():
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")


def download(url, dest):
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    print(f"downloaded {dest.name}: {dest.stat().st_size / 1e9:.2f} GB", flush=True)


def unzip(zip_path, dest, delete=True):
    import zipfile
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    if delete:
        zip_path.unlink()


def restore(name):
    """Unpack a saved split (data/final/<name>.zip in the volume) into the project tree."""
    unzip(VOL / "data" / "final" / f"{name}.zip", PROJ / "data" / "final" / name, delete=False)


def save_reports(subdirs):
    for d in subdirs:
        src = PROJ / "reports" / d
        if src.exists():
            shutil.copytree(src, VOL / "reports" / d, dirs_exist_ok=True)


@app.function(image=image, volumes={"/vol": vol}, cpu=16, memory=32768, timeout=4 * 3600)
def prepare(naf_url: str, bcdd_url: str):
    import collections
    import tarfile

    import pandas as pd

    raw = PROJ / "data" / "raw"
    # Kaggle: same folder layout as the original Windows run (data/raw/<dataset>/extracted/...)
    for name, url in (("bcdd", bcdd_url), ("bcd_nafishamoin", naf_url)):
        z = raw / name / "archive.zip"
        download(url, z)
        unzip(z, raw / name / "extracted")
        tops = collections.Counter(p.name for p in (raw / name / "extracted").iterdir())
        print(name, "top-level:", dict(tops), flush=True)
    assert (raw / "bcd_nafishamoin/extracted/CropDisease/Crop___DIsease/Corn___Common_Rust").is_dir(), \
        "unexpected nafishamoin layout"
    assert (raw / "bcdd/extracted/Bangladeshi_Crops_Disease_Dataset/test").is_dir(), "unexpected BCDD layout"
    # Mendeley: public API, then unpack every zip next to itself (like scripts/fetch_mendeley_all.ps1)
    for name, ds_id, ver in MENDELEY:
        sh(PY, "scripts/download_mendeley.py", ds_id, ver, raw / name, "--workers", 8)
        for z in sorted((raw / name).rglob("*.zip")):
            unzip(z, z.parent / z.stem)
    for d in sorted(raw.iterdir()):
        n = sum(1 for p in d.rglob("*") if p.is_file())
        print(f"raw {d.name}: {n} files", flush=True)

    sh(PY, "scripts/explore_data.py")
    sh(PY, "scripts/apply_labels.py")
    sh(PY, "scripts/clean_data.py")
    labels = ["Corn___Common_rust", "Corn___Healthy"]
    sh(PY, "scripts/make_split.py", "--name", "corn_rust", "--labels", *labels)
    sh(PY, "scripts/make_split.py", "--name", "corn_rust_v3", "--labels", *labels, "--extra", "bcdd")

    # does the new split match the original one image for image (counts here, crop stats after training)?
    split = pd.read_csv(PROJ / "data/final/corn_rust_v3/split.csv")
    counts = split.groupby(["split", "cls"]).size().to_dict()
    for key, want in SPLIT_TARGET.items():
        print(f"split {key}: {counts.get(key, 0)} (original {want})", flush=True)
    print("SPLIT COUNTS MATCH" if counts == SPLIT_TARGET else "SPLIT COUNTS DIFFER", flush=True)

    out = VOL / "data"
    (out / "final").mkdir(parents=True, exist_ok=True)
    for name in ("corn_rust", "corn_rust_v3"):
        shutil.copy2(PROJ / "data" / "final" / f"{name}.zip", out / "final" / f"{name}.zip")
    for f in ("inventory.csv", "labeled.csv"):
        shutil.copy2(PROJ / "data" / f, out / f)
    with tarfile.open(out / "processed.tar", "w") as t:   # ~22k small images: one file in the volume
        t.add(PROJ / "data" / "processed", arcname="processed")
    save_reports(["explore", "labels", "clean"])
    vol.commit()
    print("PREPARE DONE", flush=True)


@app.function(image=image, volumes={"/vol": vol}, gpu="T4", cpu=4, memory=16384, timeout=3 * 3600)
def train(run: str = "run4"):
    restore("corn_rust_v3")
    out, work = Path("/tmp/out"), Path("/tmp/work")
    work.mkdir(exist_ok=True)
    env = {"DATA_DIR": str(PROJ / "data/final/corn_rust_v3"), "OUT_DIR": str(out), "NUM_WORKERS": "4"}
    sh(PY, PROJ / "notebooks/train_corn_rust.py", env=env, cwd=work, log="/tmp/training.log")
    dest = VOL / "results" / run
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(out, dest)
    shutil.copy2("/tmp/training.log", dest / "training.log")
    vol.commit()
    print(f"TRAIN DONE -> {dest}: {sorted(p.name for p in dest.iterdir())}", flush=True)


@app.function(image=image, volumes={"/vol": vol}, cpu=16, memory=32768, timeout=3 * 3600)
def gate(run: str = "run4"):
    import tarfile
    with tarfile.open(VOL / "data" / "processed.tar") as t:
        t.extractall(PROJ / "data")
    restore("corn_rust")
    vol.reload()
    sh(PY, "scripts/build_ood_gate.py", env={"CORNRUST_RUN": str(VOL / "results" / run)})
    shutil.copy2(PROJ / "reports/ood/ood_eval.csv", VOL / "results" / run / "ood_eval.csv")
    vol.commit()
    print("GATE DONE", flush=True)


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def checksums(paths: str):
    """sha256 of comma-separated volume paths, to verify downloads (`modal volume get` can fail silently)."""
    import hashlib
    for p in paths.split(","):
        h = hashlib.sha256()
        with open(VOL / p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        print(f"{h.hexdigest()}  {p}", flush=True)


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=16384, timeout=2 * 3600)
def evaluate(run: str = "run4", baseline: str = "run3"):
    """Judge the new models and the baseline models with the same pipeline and the same (new) gate."""
    restore("corn_rust_v3")
    vol.reload()
    new = VOL / "results" / run
    base = VOL / "results" / f"{baseline}_newgate"
    shutil.copytree(PROJ / "results" / baseline, base, dirs_exist_ok=True)
    for f in ("ood_gate.json", "ood_bank.pt"):
        shutil.copy2(new / f, base / f)
    for d in (new, base):
        sh(PY, "scripts/eval_app.py", d / "app_eval.csv", env={"CORNRUST_RUN": str(d)})
    vol.commit()
    print("EVALUATE DONE", flush=True)

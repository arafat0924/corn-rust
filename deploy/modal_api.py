r"""The live model API on Modal: /api/models and /api/analyze.

cornrust.arafatcodes.dev (nginx on the VPS) serves the page and forwards /api/* here.
Deploy from the project root:  modal deploy deploy/modal_api.py
URL: https://<modal-workspace>--cornrust-api.modal.run
Ships the weights and gate files of results/<MODEL_RUN>. Scales to zero when idle, so the first
request after a quiet period waits for a cold start; at most 2 containers run, which caps the cost.
"""
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
MODEL_RUN = "live"   # the results/<folder> whose models are served (see results/live/SOURCES.json)
RUN_FILES = ["model_info.json", "results.csv", "small_cnn_best.pth", "mobilenetv3_small_best.pth",
             "efficientnet_b0_best.pth", "ood_gate.json", "ood_bank.pt"]
REMOTE_RUN = "/root/proj/results/model"


def cache_gate_backbone():
    """Bake the gate's ImageNet weights into the image, so a cold start needs no download."""
    import timm
    timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "torchvision==0.29.2", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("timm==1.0.31", "numpy", "scipy", "matplotlib", "pillow", "fastapi", "python-multipart")
    .run_function(cache_gate_backbone)
    .env({"CORNRUST_RUN": REMOTE_RUN, "MPLBACKEND": "Agg", "HF_HUB_OFFLINE": "1"})
    .add_local_file(LOCAL / "app" / "app.py", "/root/proj/app/app.py")
)
for name in RUN_FILES:
    image = image.add_local_file(LOCAL / "results" / MODEL_RUN / name, f"{REMOTE_RUN}/{name}")

app = modal.App("cornrust-api")


@app.function(image=image, cpu=2, memory=2048, min_containers=0, max_containers=2,
              scaledown_window=300, timeout=120)
@modal.concurrent(max_inputs=4)
@modal.asgi_app(label="cornrust-api")
def web():
    import sys
    sys.path.insert(0, "/root/proj/app")
    import app as core          # loads the models and the gate once per container
    return core.create_api(with_page=False)

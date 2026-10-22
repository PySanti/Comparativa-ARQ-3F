"""Evaluate the trained RT-DETRv2-L model (train_rtdetrv2.py output) following section 5 of the comparison doc.

    venv\\Scripts\\python.exe scripts\\eval_rtdetrv2.py [--run <dir>] [--recompute]

<dir> must contain best_model/ (from_pretrained dir), training_config.json and metrics.csv. By default
resultados/RT-DETRv2-L is used if it exists, otherwise runs/rtdetrv2_l_s42. Output: <dir>/evaluacion/
The metrics, thresholds, confusion matrix and collages are produced by eval_core.run_evaluation.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval_core import PRED_FLOOR, run_evaluation

ROOT = Path(__file__).resolve().parent.parent
SIZE = 640  # same input size as in training
TOP_K = 300  # same number of detections kept per image as in training-time validation

p = argparse.ArgumentParser()
p.add_argument("--run", default=None)
p.add_argument("--recompute", action="store_true", help="ignore cached predictions")
args = p.parse_args()
default_run = ROOT / "resultados" / "RT-DETRv2-L"
RUN = Path(args.run) if args.run else (default_run if default_run.exists() else ROOT / "runs" / "rtdetrv2_l_s42")
if not RUN.is_absolute():
    RUN = ROOT / RUN
OUT = RUN / "evaluacion"

_model = None  # loaded lazily so cached predictions need no GPU


def predict_split(split, gt):
    import torch
    from transformers import RTDetrV2ForObjectDetection

    global _model
    if not torch.cuda.is_available():
        raise SystemExit(f"torch {torch.__version__} sees no CUDA GPU")
    dev = torch.device("cuda")
    if _model is None:
        _model = RTDetrV2ForObjectDetection.from_pretrained(RUN / "best_model").to(dev).eval()
    stems = list(gt)
    preds = {}
    with torch.inference_mode():
        for start in range(0, len(stems), 8):
            batch = stems[start:start + 8]
            xs, sizes = [], []
            for stem in batch:
                img = Image.open(gt[stem][0]).convert("RGB")
                sizes.append(img.size)
                img = img.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                xs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0)
            out = _model(pixel_values=torch.stack(xs).to(dev))
            # top-K (query, class) pairs by sigmoid score, boxes are normalised cx, cy, w, h
            prob = out.logits.float().sigmoid()
            _, Q, C = prob.shape
            s, i = prob.flatten(1).topk(min(TOP_K, Q * C), dim=1)
            q, c = i // C, i % C
            bx = out.pred_boxes.float().gather(1, q[..., None].expand(-1, -1, 4))
            for k, stem in enumerate(batch):
                w, h = sizes[k]
                cx, cy, bw, bh = bx[k].unbind(-1)
                boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
                keep = (s[k] >= PRED_FLOOR).cpu().numpy()
                preds[stem] = (boxes.cpu().numpy().astype(np.float64)[keep], s[k].cpu().numpy().astype(np.float64)[keep],
                               c[k].cpu().numpy().astype(int)[keep])
            if (start // 8 + 1) % 20 == 0:
                print(f"  {split}: {min(start + 8, len(stems))}/{len(stems)}", flush=True)
    return preds


def hyperparameters():
    # hyperparameters and timings written by train_rtdetrv2.py
    cfg = json.loads((RUN / "training_config.json").read_text())
    best = json.loads((RUN / "best_model" / "best_epoch.json").read_text())
    with open(RUN / "metrics.csv", encoding="utf-8") as f:
        epochs_done = max(int(r["epoch"]) for r in csv.DictReader(f)) + 1
    t = cfg.get("total_training_time_s")
    return dict(model=cfg["model_hf"], epochs=cfg["epochs_effective"], epochs_logged=epochs_done, best_epoch=best["epoch"] + 1,
                batch_size=cfg["batch_size"], grad_accum_steps=cfg["grad_accum"], effective_batch=cfg["effective_batch"],
                lr=cfg["lr"], lr_backbone=cfg["lr_backbone"], optimizer=cfg["optimizer"], weight_decay=cfg["weight_decay"],
                lr_scheduler=cfg["lr_scheduler"], ema_decay=cfg["ema_decay"], input_size=cfg["input_size"],
                multi_scale=cfg["multi_scale"], bf16=cfg["bf16"], seed=cfg["seed"], gpu=cfg["gpu"],
                training_time_s=t, training_time_h=round(t / 3600, 2) if t else "no registrado")


if __name__ == "__main__":
    run_evaluation(OUT, predict_split, hyperparameters(), recompute=args.recompute)

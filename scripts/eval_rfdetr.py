"""Evaluate the trained RF-DETR Medium checkpoint following section 5 of the comparison doc.

Run with the project venv:  venv\\Scripts\\python.exe scripts\\eval_rfdetr.py [--recompute]
Output: resultados/RF-DETR Medium/evaluacion/
The metrics, thresholds, confusion matrix and collages are produced by eval_core.run_evaluation; this script
only loads the RF-DETR checkpoint, predicts, and reads the training hyperparameters.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval_core import PRED_FLOOR, run_evaluation

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "resultados" / "RF-DETR Medium"
CKPT = RES / "checkpoint_best_total.pth"
OUT = RES / "evaluacion"

p = argparse.ArgumentParser()
p.add_argument("--recompute", action="store_true", help="ignore cached predictions")
args = p.parse_args()

_model = None  # loaded lazily so cached predictions need no GPU


def predict_split(split, gt):
    global _model
    if _model is None:
        from rfdetr import RFDETRMedium
        _model = RFDETRMedium(pretrain_weights=str(CKPT))
    import torch
    preds = {}
    with torch.inference_mode():
        for i, (stem, (path, _, _)) in enumerate(gt.items()):
            d = _model.predict(Image.open(path).convert("RGB"), threshold=PRED_FLOOR)
            preds[stem] = (np.asarray(d.xyxy, dtype=np.float64).reshape(-1, 4),
                           np.asarray(d.confidence, dtype=np.float64), np.asarray(d.class_id, dtype=int))
            if (i + 1) % 100 == 0:
                print(f"  {split}: {i + 1}/{len(gt)}", flush=True)
    return preds


def hyperparameters():
    # hyperparameters from the training run (done elsewhere, on a RTX 3060)
    cfg = json.loads((RES / "training_config.json").read_text())
    tc, mc = cfg["train_config"], cfg["model_config"]
    with open(RES / "metrics.csv", encoding="utf-8") as f:
        epochs_done = max(int(r["epoch"]) for r in csv.DictReader(f)) + 1
    return dict(epochs=tc["epochs"], epochs_logged=epochs_done, batch_size=tc["batch_size"],
                grad_accum_steps=tc["grad_accum_steps"], effective_batch=tc["batch_size"] * tc["grad_accum_steps"],
                lr=tc["lr"], lr_encoder=tc["lr_encoder"], optimizer=tc["optimizer"], weight_decay=tc["weight_decay"],
                lr_scheduler=tc["lr_scheduler"], ema=tc["use_ema"], ema_decay=tc["ema_decay"],
                resolution=mc["resolution"], encoder=mc["encoder"], multi_scale=tc["multi_scale"], seed=tc["seed"],
                early_stopping=tc["early_stopping"], gpu="RTX 3060 12 GB (dato del usuario)",
                training_time="no registrado en metrics.csv ni training_config.json")


if __name__ == "__main__":
    run_evaluation(OUT, predict_split, hyperparameters(), recompute=args.recompute)

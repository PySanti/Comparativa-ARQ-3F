"""Train RF-DETR Medium on dataset_split/ and report mAP / F1 + best checkpoint.

    python scripts/train_rfdetr.py --classes all      # 6 classes
    python scripts/train_rfdetr.py --classes bache    # only class 0

Written against the rfdetr 1.x API and NOT executed when authored (nothing was installed).
First-run things to verify are marked FIXME-VERIFY.
Output: runs/rfdetr_medium_<classes>_s<seed>/{coco/, checkpoint_best_total.pth, metrics.json}
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "dataset_split"
NAMES = ["bache", "semaforo", "alumbrado", "basura", "arbol", "contenedor"]

p = argparse.ArgumentParser()
p.add_argument("--classes", choices=["all", "bache"], default="all")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--grad-accum", type=int, default=4)  # effective batch = batch-size * grad-accum
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--eval-only", action="store_true", help="skip training, evaluate existing checkpoint")
args = p.parse_args()

KEEP = [0] if args.classes == "bache" else list(range(len(NAMES)))
CLASS_NAMES = [NAMES[c] for c in KEEP]
OUT = ROOT / "runs" / f"rfdetr_medium_{args.classes}_s{args.seed}"
COCO = OUT / "coco"


def check_env():
    try:
        import rfdetr  # noqa: F401
    except ImportError:
        sys.exit("rfdetr not installed (script does not install anything). pip install rfdetr")
    import torch
    if not torch.cuda.is_available():
        sys.exit(f"torch {torch.__version__} sees no CUDA GPU. Install a CUDA build of PyTorch "
                 "(RTX 5070 needs CUDA >= 12.8).")


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_coco():
    """Convert dataset_split/ (YOLO) to COCO, keeping only KEEP classes, ids remapped to 0..n-1.

    RF-DETR's folder convention is train/ + valid/, each with _annotations.coco.json.
    """
    remap = {c: i for i, c in enumerate(KEEP)}
    for split, folder in (("train", "train"), ("val", "valid")):
        d = COCO / folder
        d.mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        for img in sorted((SPLIT / "images" / split).iterdir()):
            link_or_copy(img, d / img.name)
            w, h = Image.open(img).size
            iid = len(images) + 1
            images.append({"id": iid, "file_name": img.name, "width": w, "height": h})
            for line in (SPLIT / "labels" / split / f"{img.stem}.txt").read_text().splitlines():
                parts = line.split()
                if len(parts) != 5 or int(parts[0]) not in remap:
                    continue
                cx, cy, bw, bh = (float(x) for x in parts[1:])
                bw, bh = bw * w, bh * h
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": remap[int(parts[0])],
                             "bbox": [cx * w - bw / 2, cy * h - bh / 2, bw, bh], "area": bw * bh, "iscrowd": 0})
        cats = [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]
        (d / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
        print(f"{folder}: {len(images)} images, {len(anns)} boxes")


def iou_matrix(a, b):
    """a: (n,4) xyxy, b: (m,4) xyxy."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def evaluate(ckpt):
    from pycocotools.coco import COCO as CocoGT
    from pycocotools.cocoeval import COCOeval
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights=str(ckpt))
    gt = CocoGT(str(COCO / "valid" / "_annotations.coco.json"))
    n = len(KEEP)
    dets = []                       # coco-format predictions (low threshold, for mAP)
    per_img = {}                    # image_id -> (gt boxes by class, preds by class)
    for iid, info in gt.imgs.items():
        pred = model.predict(Image.open(COCO / "valid" / info["file_name"]).convert("RGB"), threshold=0.01)
        # FIXME-VERIFY: class_id offset. Assumes predictions use the same 0..n-1 ids as the COCO json.
        if len(pred.class_id) and int(pred.class_id.max()) >= n:
            print("WARNING: predicted class_id >= n_classes; class id offset needs adjusting", file=sys.stderr)
        g = {c: [] for c in range(n)}
        for a in gt.loadAnns(gt.getAnnIds(imgIds=iid)):
            x, y, w, h = a["bbox"]
            g[a["category_id"]].append([x, y, x + w, y + h])
        pr = {c: [] for c in range(n)}
        for box, s, c in zip(pred.xyxy, pred.confidence, pred.class_id):
            c = int(c)
            if 0 <= c < n:
                dets.append({"image_id": iid, "category_id": c, "score": float(s),
                             "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]})
                pr[c].append((float(s), box))
        per_img[iid] = (g, pr)

    res = {"mAP50_95": 0.0, "mAP50": 0.0, "per_class": {}}
    if dets:
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        res["mAP50_95"], res["mAP50"] = float(ev.stats[0]), float(ev.stats[1])
        for i, name in enumerate(CLASS_NAMES):
            e = COCOeval(gt, dt, "bbox"); e.params.catIds = [i]
            e.evaluate(); e.accumulate(); e.summarize()
            res["per_class"][name] = {"mAP50_95": float(e.stats[0]), "mAP50": float(e.stats[1])}

    # F1 @ IoU 0.5, greedy matching by confidence, swept over confidence thresholds.
    thrs = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tp = np.zeros((n, len(thrs))); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    for g, pr in per_img.values():
        for c in range(n):
            gb = np.array(g[c]).reshape(-1, 4)
            for ti, t in enumerate(thrs):
                pb = sorted((x for x in pr[c] if x[0] >= t), key=lambda x: -x[0])
                used = set(); hit = 0
                if len(pb) and len(gb):
                    iou = iou_matrix(np.array([x[1] for x in pb]), gb)
                    for row in iou:
                        j = int(np.argmax(np.where([k in used for k in range(len(gb))], -1, row)))
                        if row[j] >= 0.5 and j not in used:
                            used.add(j); hit += 1
                tp[c, ti] += hit; fp[c, ti] += len(pb) - hit; fn[c, ti] += len(gb) - hit

    def f1(t_, f_, n_):
        pr_ = t_ / np.maximum(t_ + f_, 1e-9); rc = t_ / np.maximum(t_ + n_, 1e-9)
        return 2 * pr_ * rc / np.maximum(pr_ + rc, 1e-9)

    for i, name in enumerate(CLASS_NAMES):
        f = f1(tp[i], fp[i], fn[i]); k = int(np.argmax(f))
        res["per_class"].setdefault(name, {}).update({"F1": float(f[k]), "F1_threshold": float(thrs[k])})
    fg = f1(tp.sum(0), fp.sum(0), fn.sum(0)); k = int(np.argmax(fg))
    res["F1_micro"], res["F1_micro_threshold"] = float(fg[k]), float(thrs[k])
    return res


def main():
    check_env()
    import torch
    from rfdetr import RFDETRMedium

    OUT.mkdir(parents=True, exist_ok=True)
    if not COCO.exists():
        build_coco()
    if not args.eval_only:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model = RFDETRMedium()
        # FIXME-VERIFY: argument names follow rfdetr 1.x TrainConfig.
        model.train(dataset_dir=str(COCO), epochs=args.epochs, batch_size=args.batch_size,
                    grad_accum_steps=args.grad_accum, lr=args.lr, output_dir=str(OUT), seed=args.seed)
    ckpt = OUT / "checkpoint_best_total.pth"
    if not ckpt.exists():
        sys.exit(f"{ckpt} not found; check the files rfdetr wrote in {OUT}")
    metrics = evaluate(ckpt)
    metrics["config"] = {k: v for k, v in vars(args).items()} | {"class_names": CLASS_NAMES}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

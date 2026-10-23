"""Model-independent evaluation following section 5 of the comparison doc, shared by eval_rfdetr.py and eval_rtdetrv2.py.

A model script only has to provide `predict_split(split, gt) -> {stem: (boxes_xyxy_px, scores, class_ids)}` returning
detections at score >= PRED_FLOOR, plus a dict of training hyperparameters; run_evaluation does the rest:

1. Per class, pick the confidence threshold that maximises F1 on val (sweep 0.001..1.0, step 0.001, IoU 0.5).
2. On test, with those fixed per-class thresholds: P/R/F1 per class and macro-averaged,
   AP@50 per class and mean (COCO-style 101-point AP, own implementation: pycocotools is not installed),
   confusion matrix (counts and row-normalised), and 100 bache collages (see collage.py).
"""
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import collage
from collage import NAMES, make_collage

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "dataset_split"
NC = len(NAMES)
PRED_FLOOR = 0.001
IOU_THR = 0.5  # default; run_evaluation(iou_thr=...) overrides it for the whole run
MAX_DET = 100  # per image and class, as in COCO
THRS = np.round(np.arange(1, 1001) / 1000, 3)
SEED = 42
N_COLLAGES = 100


# ---------------------------------------------------------------- data
def load_gt(split):
    gt = {}
    for img in sorted((SPLIT / "images" / split).iterdir()):
        w, h = Image.open(img).size
        boxes, cls = [], []
        for line in (SPLIT / "labels" / split / f"{img.stem}.txt").read_text().splitlines():
            q = line.split()
            if len(q) != 5:
                continue
            c, cx, cy, bw, bh = int(q[0]), *(float(x) for x in q[1:])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            cls.append(c)
        gt[img.stem] = (img, np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(cls, dtype=int))
    return gt


def cached_predictions(out, split, gt, predict_split, recompute):
    """Predictions are expensive: reuse out/predictions_<split>.pkl unless recompute."""
    cache = out / f"predictions_{split}.pkl"
    if cache.exists() and not recompute:
        return pickle.load(open(cache, "rb"))
    preds = predict_split(split, gt)
    pickle.dump(preds, open(cache, "wb"))
    return preds


# ---------------------------------------------------------------- metrics
def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-12)


def greedy_match(pb, ps, gb):
    """pb/ps predictions of one class in one image, gb its GT. Returns scores desc and tp flags."""
    order = np.argsort(-ps, kind="stable")[:MAX_DET]
    pb, ps = pb[order], ps[order]
    tp = np.zeros(len(ps), dtype=bool)
    if len(ps) and len(gb):
        m = iou(pb, gb)
        used = np.zeros(len(gb), dtype=bool)
        for i in range(len(ps)):
            row = np.where(used, -1.0, m[i])
            j = int(np.argmax(row))
            if row[j] >= IOU_THR:
                used[j] = True
                tp[i] = True
    return ps, tp


def class_tables(gt, preds):
    """Per class: all predicted scores, tp flags, and number of GT boxes (matching at floor 0.001)."""
    S = [[] for _ in range(NC)]
    T = [[] for _ in range(NC)]
    ngt = np.zeros(NC, dtype=int)
    for stem, (_, gb, gc) in gt.items():
        pb, ps, pc = preds[stem]
        for c in range(NC):
            ngt[c] += int((gc == c).sum())
            ms = pc == c
            s, t = greedy_match(pb[ms], ps[ms], gb[gc == c])
            S[c].append(s)
            T[c].append(t)
    return ([np.concatenate(x) if x else np.zeros(0) for x in S],
            [np.concatenate(x) if x else np.zeros(0, dtype=bool) for x in T], ngt)


def prf_at(s, t, ngt, thr):
    keep = s >= thr
    tp = int(t[keep].sum())
    n = int(keep.sum())
    pr = tp / n if n else 0.0
    rc = tp / ngt if ngt else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return dict(TP=tp, FP=n - tp, FN=int(ngt) - tp, precision=pr, recall=rc, F1=f1)


def sweep_best(s, t, ngt):
    order = np.argsort(-s, kind="stable")
    ss, cum = s[order], np.cumsum(t[order])
    n_ge = np.searchsorted(-ss, -THRS, side="right")
    tp = np.where(n_ge > 0, cum[np.maximum(n_ge - 1, 0)], 0)
    pr = np.where(n_ge > 0, tp / np.maximum(n_ge, 1), 0.0)
    rc = tp / max(ngt, 1)
    f1 = np.where(pr + rc > 0, 2 * pr * rc / np.maximum(pr + rc, 1e-12), 0.0)
    best = f1.max()
    plateau = THRS[f1 >= best - 1e-12]
    return float(np.median(plateau)), float(best), float(plateau.min()), float(plateau.max())


def ap50(s, t, ngt):
    """COCO-style AP: 101 recall points, monotone precision envelope."""
    if ngt == 0 or len(s) == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")
    tp = np.cumsum(t[order])
    fp = np.cumsum(~t[order])
    rc, pr = tp / ngt, tp / np.maximum(tp + fp, 1)
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    idx = np.searchsorted(rc, np.linspace(0, 1, 101), side="left")
    return float(np.mean([pr[i] if i < len(pr) else 0.0 for i in idx]))


def confusion(gt, preds, thr):
    cm = np.zeros((NC + 1, NC + 1), dtype=int)  # rows: true (last = background), cols: predicted
    for stem, (_, gb, gc) in gt.items():
        pb, ps, pc = preds[stem]
        keep = ps >= thr[np.clip(pc, 0, NC - 1)]
        pb, ps, pc = pb[keep], ps[keep], pc[keep]
        order = np.argsort(-ps, kind="stable")
        pb, pc = pb[order], pc[order]
        used = np.zeros(len(gb), dtype=bool)
        m = iou(pb, gb) if len(pb) and len(gb) else None
        for i in range(len(pb)):
            if m is not None:
                row = np.where(used, -1.0, m[i])
                j = int(np.argmax(row))
                if row[j] >= IOU_THR:
                    used[j] = True
                    cm[gc[j], pc[i]] += 1
                    continue
            cm[NC, pc[i]] += 1
        for j in np.where(~used)[0]:
            cm[gc[j], NC] += 1
    return cm


def fix_class_offset(gt, preds):
    """Check that predicted class ids line up with the YOLO ids; a shift is applied only if it clearly beats 0."""
    agree = {o: 0 for o in (-1, 0, 1)}
    for stem, (_, gb, gc) in gt.items():
        pb, ps, pc = preds[stem]
        k = ps >= 0.5
        if k.any() and len(gb):
            m = iou(pb[k], gb)
            j = m.argmax(1)
            ok = m.max(1) >= IOU_THR
            for o in agree:
                agree[o] += int((pc[k][ok] + o == gc[j[ok]]).sum())
    best = 0 if agree[0] >= max(agree.values()) else max(agree, key=agree.get)
    print("class id agreement by offset:", agree)
    if best != 0:
        print(f"WARNING: predicted class ids are shifted by {best:+d}; applying it", file=sys.stderr)
    return best


# ---------------------------------------------------------------- full evaluation
def run_evaluation(out, predict_split, hyper, recompute=False, iou_thr=0.5, cache_dir=None):
    """iou_thr: IoU used to match detections with labels everywhere (thresholds, metrics, confusion, collage colours).
    cache_dir: where predictions_<split>.pkl live (they do not depend on the IoU), defaults to out."""
    global IOU_THR
    IOU_THR = collage.IOU_THR = iou_thr
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else out
    (out / "collages").mkdir(exist_ok=True)
    gt = {s: load_gt(s) for s in ("val", "test")}
    print("predicting val/test (cached if available)")
    preds = {s: cached_predictions(cache_dir, s, gt[s], predict_split, recompute) for s in gt}
    off = fix_class_offset(gt["val"], preds["val"])
    for s in preds:
        preds[s] = {k: (b, sc, c + off) for k, (b, sc, c) in preds[s].items()}
        for k, (b, sc, c) in list(preds[s].items()):
            ok = (c >= 0) & (c < NC)
            preds[s][k] = (b[ok], sc[ok], c[ok])

    # 1. per-class thresholds on val
    Sv, Tv, Nv = class_tables(gt["val"], preds["val"])
    thr = np.zeros(NC)
    thr_info = {}
    for c in range(NC):
        t, f1, lo, hi = sweep_best(Sv[c], Tv[c], Nv[c])
        thr[c] = t
        thr_info[NAMES[c]] = dict(threshold=t, val_F1=f1, plateau_min=lo, plateau_max=hi, val_boxes=int(Nv[c]))
        print(f"  {NAMES[c]:11s} thr={t:.3f} (plateau {lo:.3f}-{hi:.3f}) val F1={f1:.4f}")
    (out / "umbrales_val.json").write_text(json.dumps(thr_info, indent=2))

    # 2. test metrics with the fixed thresholds
    St, Tt, Nt = class_tables(gt["test"], preds["test"])
    per = {}
    for c in range(NC):
        m = prf_at(St[c], Tt[c], Nt[c], thr[c])
        m["AP50"] = ap50(St[c], Tt[c], Nt[c])
        m["threshold"] = float(thr[c])
        m["test_boxes"] = int(Nt[c])
        per[NAMES[c]] = m
    macro = {k: float(np.mean([per[n][k] for n in NAMES])) for k in ("precision", "recall", "F1", "AP50")}
    tot = {k: sum(per[n][k] for n in NAMES) for k in ("TP", "FP", "FN")}
    pr = tot["TP"] / max(tot["TP"] + tot["FP"], 1)
    rc = tot["TP"] / max(tot["TP"] + tot["FN"], 1)
    micro = dict(precision=pr, recall=rc, F1=2 * pr * rc / max(pr + rc, 1e-12))

    # 3. confusion matrix (csv + two png)
    cm = confusion(gt["test"], preds["test"], thr)
    labels = NAMES + ["fondo"]
    with open(out / "matriz_confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["real \\ predicho"] + labels)
        for i, r in enumerate(cm):
            w.writerow([labels[i]] + r.tolist())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for norm in (False, True):
        fig, ax = plt.subplots(figsize=(8, 7))
        data = cm / np.maximum(cm.sum(1, keepdims=True), 1) if norm else cm
        ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(NC + 1), labels, rotation=45, ha="right")
        ax.set_yticks(range(NC + 1), labels)
        ax.set_xlabel("predicho")
        ax.set_ylabel("real")
        ax.set_title("Matriz de confusion (test, " + ("normalizada por fila)" if norm else "conteos)"))
        for i in range(NC + 1):
            for j in range(NC + 1):
                v = data[i, j]
                ax.text(j, i, f"{v:.2f}" if norm else str(v), ha="center", va="center",
                        color="white" if v > data.max() * 0.55 else "black", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / ("matriz_confusion_normalizada.png" if norm else "matriz_confusion.png"), dpi=150)
        plt.close(fig)

    result = dict(umbral_por_clase={n: per[n]["threshold"] for n in NAMES}, por_clase=per,
                  global_macro=macro, global_micro_referencia=micro, bache=per["bache"],
                  mAP50_global=macro["AP50"], iou_thr=iou_thr, hiperparametros=hyper,
                  notas=["AP50 estilo COCO (101 puntos) implementado a mano; pycocotools no esta instalado",
                         "umbral = mediana de la meseta de F1 maximo en val (barrido 0.001..1, paso 0.001)",
                         f"IoU {iou_thr:g} para emparejar detecciones con labels (las claves AP50/mAP50 son el AP a ese IoU); maximo 100 detecciones por imagen y clase"])
    (out / "metrics_test.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # 4. collages
    stems = sorted(gt["test"])
    pick = random.Random(SEED).sample(stems, min(N_COLLAGES, len(stems)))
    for i, stem in enumerate(pick):
        path, gb, gc = gt["test"][stem]
        pb, ps, pc = preds["test"][stem]
        make_collage(path, gb, gc, pb, ps, pc, thr, out / "collages" / f"{i:03d}_{stem[:40]}.jpg")
    (out / "collages_lista.txt").write_text("\n".join(f"{i:03d} {s}" for i, s in enumerate(pick)))

    print(json.dumps({k: result[k] for k in ("umbral_por_clase", "global_macro", "global_micro_referencia")}, indent=2))
    print("bache:", {k: round(v, 4) for k, v in per["bache"].items()})
    print("done ->", out)
    return result

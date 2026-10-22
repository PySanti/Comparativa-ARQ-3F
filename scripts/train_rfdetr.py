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

# Objetivo: fijar las rutas del proyecto y los nombres de clase, que son la referencia común de todo el script.
ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "dataset_split"
NAMES = ["bache", "semaforo", "alumbrado", "basura", "arbol", "contenedor"]

# Objetivo: leer los parámetros de la corrida por línea de comandos (clases a usar, semilla, hiperparámetros, modo solo-evaluación).
p = argparse.ArgumentParser()
p.add_argument("--classes", choices=["all", "bache"], default="all")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--grad-accum", type=int, default=4)  # effective batch = batch-size * grad-accum
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--eval-only", action="store_true", help="skip training, evaluate existing checkpoint")
args = p.parse_args()

# Objetivo: derivar de los argumentos las clases que se conservan (todas o solo bache) y las carpetas de salida de esta corrida.
KEEP = [0] if args.classes == "bache" else list(range(len(NAMES)))
CLASS_NAMES = [NAMES[c] for c in KEEP]
OUT = ROOT / "runs" / f"rfdetr_medium_{args.classes}_s{args.seed}"
COCO = OUT / "coco"


def check_env():
    # Objetivo: comprobar que rfdetr está instalado y que torch ve una GPU CUDA; si no, detenerse con un mensaje claro sin instalar nada.
    try:
        import rfdetr  # noqa: F401
    except ImportError:
        sys.exit("rfdetr not installed (script does not install anything). pip install rfdetr")
    import torch
    if not torch.cuda.is_available():
        sys.exit(f"torch {torch.__version__} sees no CUDA GPU. Install a CUDA build of PyTorch "
                 "(RTX 5070 needs CUDA >= 12.8).")


def link_or_copy(src, dst):
    # Objetivo: poner la imagen en la carpeta COCO sin duplicar espacio (hardlink); si el sistema de archivos no lo permite, copiarla.
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_coco():
    """Convert dataset_split/ (YOLO) to COCO, keeping only KEEP classes, ids remapped to 0..n-1.

    RF-DETR's folder convention is train/ + valid/, each with _annotations.coco.json.
    """
    # Objetivo: mapear los ids de clase originales a ids contiguos 0..n-1 (necesario cuando solo se conserva bache).
    remap = {c: i for i, c in enumerate(KEEP)}
    # Objetivo: generar una carpeta COCO por split; RF-DETR espera los nombres train/ y valid/.
    for split, folder in (("train", "train"), ("val", "valid")):
        d = COCO / folder
        d.mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        for img in sorted((SPLIT / "images" / split).iterdir()):
            # Objetivo: registrar la imagen (archivo, tamaño e id) para el JSON de COCO.
            link_or_copy(img, d / img.name)
            w, h = Image.open(img).size
            iid = len(images) + 1
            images.append({"id": iid, "file_name": img.name, "width": w, "height": h})
            # Objetivo: convertir cada caja YOLO (centro, tamaño, normalizada) a caja COCO (esquina, tamaño, en píxeles), descartando las clases no conservadas.
            for line in (SPLIT / "labels" / split / f"{img.stem}.txt").read_text().splitlines():
                parts = line.split()
                if len(parts) != 5 or int(parts[0]) not in remap:
                    continue
                cx, cy, bw, bh = (float(x) for x in parts[1:])
                bw, bh = bw * w, bh * h
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": remap[int(parts[0])],
                             "bbox": [cx * w - bw / 2, cy * h - bh / 2, bw, bh], "area": bw * bh, "iscrowd": 0})
        # Objetivo: escribir el JSON de anotaciones del split con sus imágenes, cajas y categorías.
        cats = [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]
        (d / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
        print(f"{folder}: {len(images)} images, {len(anns)} boxes")


def iou_matrix(a, b):
    """a: (n,4) xyxy, b: (m,4) xyxy."""
    # Objetivo: calcular el IoU entre cada caja de a y cada caja de b (matriz n x m) para emparejar predicciones con labels.
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def evaluate(ckpt):
    from pycocotools.coco import COCO as CocoGT
    from pycocotools.cocoeval import COCOeval
    from rfdetr import RFDETRMedium

    # Objetivo: cargar el mejor checkpoint y las anotaciones de validación contra las que se evaluará.
    model = RFDETRMedium(pretrain_weights=str(ckpt))
    gt = CocoGT(str(COCO / "valid" / "_annotations.coco.json"))
    n = len(KEEP)
    dets = []                       # coco-format predictions (low threshold, for mAP)
    per_img = {}                    # image_id -> (gt boxes by class, preds by class)
    # Objetivo: predecir cada imagen de validación y guardar, por clase, las labels reales y las predicciones (para mAP y para F1).
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

    # Objetivo: calcular mAP@50:95 y mAP@50, globales y por clase, con la evaluación estándar de COCO.
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
    # Objetivo: contar verdaderos positivos, falsos positivos y falsos negativos de cada clase para cada umbral de confianza (IoU 0.5, emparejamiento voraz por confianza).
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
        # Objetivo: calcular el F1 a partir de TP, FP y FN (precision y recall intermedios).
        pr_ = t_ / np.maximum(t_ + f_, 1e-9); rc = t_ / np.maximum(t_ + n_, 1e-9)
        return 2 * pr_ * rc / np.maximum(pr_ + rc, 1e-9)

    # Objetivo: quedarse, por clase, con el umbral que da el mejor F1, y guardar también el mejor F1 global (micro).
    for i, name in enumerate(CLASS_NAMES):
        f = f1(tp[i], fp[i], fn[i]); k = int(np.argmax(f))
        res["per_class"].setdefault(name, {}).update({"F1": float(f[k]), "F1_threshold": float(thrs[k])})
    fg = f1(tp.sum(0), fp.sum(0), fn.sum(0)); k = int(np.argmax(fg))
    res["F1_micro"], res["F1_micro_threshold"] = float(fg[k]), float(thrs[k])
    return res


def main():
    # Objetivo: verificar el entorno antes de hacer trabajo costoso.
    check_env()
    import torch
    from rfdetr import RFDETRMedium

    # Objetivo: preparar la carpeta de salida y el dataset en formato COCO (solo si aún no existe).
    OUT.mkdir(parents=True, exist_ok=True)
    if not COCO.exists():
        build_coco()
    # Objetivo: entrenar RF-DETR Medium con semilla fija y los hiperparámetros indicados (se omite con --eval-only).
    if not args.eval_only:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model = RFDETRMedium()
        # FIXME-VERIFY: argument names follow rfdetr 1.x TrainConfig.
        model.train(dataset_dir=str(COCO), epochs=args.epochs, batch_size=args.batch_size,
                    grad_accum_steps=args.grad_accum, lr=args.lr, output_dir=str(OUT), seed=args.seed)
    # Objetivo: localizar el mejor checkpoint que dejó rfdetr; si no está, avisar en lugar de evaluar con nada.
    ckpt = OUT / "checkpoint_best_total.pth"
    if not ckpt.exists():
        sys.exit(f"{ckpt} not found; check the files rfdetr wrote in {OUT}")
    # Objetivo: evaluar el mejor checkpoint en validación y guardar las métricas junto con la configuración de la corrida.
    metrics = evaluate(ckpt)
    metrics["config"] = {k: v for k, v in vars(args).items()} | {"class_names": CLASS_NAMES}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

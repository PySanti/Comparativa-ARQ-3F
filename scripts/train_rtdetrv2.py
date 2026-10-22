"""Train RT-DETRv2-L (Hugging Face transformers) on dataset_split/ for the architecture comparison.

    venv\\Scripts\\python.exe scripts\\train_rtdetrv2.py                 # 50 epochs
    venv\\Scripts\\python.exe scripts\\train_rtdetrv2.py --smoke         # 8 batches, 1 epoch, to check the pipeline
    venv\\Scripts\\python.exe scripts\\train_rtdetrv2.py --resume        # continue from runs/<out>/last.pth

Best checkpoint = highest val mAP@50:95 averaged over the 6 classes (same criterion as RF-DETR), evaluated on the EMA model.
Output (runs/rtdetrv2_l_s<seed>/): best_model/ (from_pretrained dir), last.pth, metrics.csv, training_config.json (with timings).
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Objetivo: fijar rutas, nombres de clase, tamaño de entrada y los umbrales de IoU (0.5 a 0.95) que usa el mAP@50:95.
ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "dataset_split"
NAMES = ["bache", "semaforo", "alumbrado", "basura", "arbol", "contenedor"]
NC = len(NAMES)
SIZE = 640
IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)

# Objetivo: leer los hiperparámetros y modos de la corrida por línea de comandos (--smoke para probar, --resume para continuar).
p = argparse.ArgumentParser()
p.add_argument("--model", default="PekingU/rtdetr_v2_r50vd")  # r50vd = RT-DETRv2-L
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--grad-accum", type=int, default=4)  # effective batch = batch-size * grad-accum
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--lr-backbone", type=float, default=1e-5)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--warmup-steps", type=int, default=300)  # optimizer steps
p.add_argument("--lr-drop-epoch", type=int, default=40)  # lr x0.1 from this epoch
p.add_argument("--ema-decay", type=float, default=0.999)
p.add_argument("--clip", type=float, default=0.1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--out", default=None)
p.add_argument("--resume", action="store_true")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
# Objetivo: decidir la carpeta de salida de esta corrida (la de la prueba rápida queda separada de la real).
OUT = ROOT / "runs" / (args.out or ("rtdetrv2_smoke" if args.smoke else f"rtdetrv2_l_s{args.seed}"))


class YoloSet(Dataset):
    # Objetivo: servir las imágenes y labels YOLO de un split directamente como tensores para entrenar o validar (sin convertir a COCO).
    def __init__(self, split, train, limit=None):
        # Objetivo: listar los archivos del split (con un límite opcional para la prueba rápida) y recordar si se aplican aumentos.
        self.files = sorted((SPLIT / "images" / split).iterdir())[: limit or None]
        self.split, self.train = split, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        # Objetivo: cargar la imagen, redimensionarla a SIZE x SIZE y pasarla a tensor con valores 0-1 (RT-DETR no usa media/desviación).
        img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        # Objetivo: leer las labels YOLO de la imagen como clases y cajas (centro y tamaño normalizados), que es lo que espera el modelo.
        rows = [l.split() for l in (SPLIT / "labels" / self.split / f"{path.stem}.txt").read_text().splitlines() if l.strip()]
        cls = torch.tensor([int(r[0]) for r in rows], dtype=torch.long)
        box = torch.tensor([[float(v) for v in r[1:]] for r in rows], dtype=torch.float32).reshape(-1, 4)  # cx cy w h, normalized
        # Objetivo: aumentar los datos solo en entrenamiento: volteo horizontal (con la caja reflejada) y cambios suaves de contraste y brillo.
        if self.train:
            if random.random() < 0.5:
                x = x.flip(-1)
                box[:, 0] = 1.0 - box[:, 0]
            x = ((x - 0.5) * random.uniform(0.8, 1.2) + 0.5) * random.uniform(0.8, 1.2)  # contrast, brightness
            x = x.clamp(0, 1)
        return x, {"class_labels": cls, "boxes": box}


def collate(batch):
    # Objetivo: juntar varias muestras en un batch (imágenes apiladas; las labels van en lista porque cada imagen tiene un número distinto de cajas).
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def xyxy(b):
    # Objetivo: convertir cajas de formato centro-tamaño (cx, cy, w, h) a esquinas (x0, y0, x1, y1).
    return torch.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)


def iou(a, b):
    # Objetivo: calcular el IoU entre cada caja de a y cada caja de b (matriz) para emparejar predicciones con labels.
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-12)


def ap101(s, t, ngt):
    # Objetivo: calcular el AP de una clase con el método de COCO: curva precision-recall con envolvente monótona, muestreada en 101 puntos de recall.
    if ngt == 0 or len(s) == 0:
        return 0.0
    # Objetivo: ordenar las predicciones por confianza y acumular aciertos y fallos para obtener precision y recall en cada punto.
    o = np.argsort(-s, kind="stable")
    tp = np.cumsum(t[o])
    fp = np.cumsum(~t[o])
    rc, pr = tp / ngt, tp / np.maximum(tp + fp, 1)
    # Objetivo: hacer la precision monótona decreciente y promediarla en los 101 niveles de recall.
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    idx = np.searchsorted(rc, np.linspace(0, 1, 101), side="left")
    return float(np.mean([pr[i] if i < len(pr) else 0.0 for i in idx]))


def coco_map(gts, preds):
    """gts: list of (boxes_xyxy, cls); preds: list of (boxes_xyxy, scores, cls). IoU is invariant to the
    per-axis normalisation, so everything stays in normalised coordinates. COCO-style: 101-point AP,
    max 100 detections per image and class, mean over IoU .5:.95 and over classes that have GT."""
    # Objetivo: preparar los acumuladores de puntajes y aciertos por clase y por umbral de IoU.
    S = [[[] for _ in IOUS] for _ in range(NC)]
    T = [[[] for _ in IOUS] for _ in range(NC)]
    ngt = np.zeros(NC, dtype=int)
    # Objetivo: recorrer cada imagen y clase, quedarse con las 100 mejores predicciones y contar las labels reales.
    for (gb, gc), (pb, ps, pc) in zip(gts, preds):
        for c in range(NC):
            g = gb[gc == c]
            ngt[c] += len(g)
            m = pc == c
            order = np.argsort(-ps[m], kind="stable")[:100]
            b, s = pb[m][order], ps[m][order]
            if len(s) == 0:
                continue
            # Objetivo: para cada umbral de IoU, emparejar las predicciones (de mayor a menor confianza) con la label libre de mayor IoU y marcar cuáles son aciertos.
            mat = iou(b, g) if len(g) else np.zeros((len(s), 0))
            for k, thr in enumerate(IOUS):
                used = np.zeros(len(g), dtype=bool)
                tp = np.zeros(len(s), dtype=bool)
                for i in range(len(s)):
                    if len(g):
                        row = np.where(used, -1.0, mat[i])
                        j = int(np.argmax(row))
                        if row[j] >= thr:
                            used[j] = True
                            tp[i] = True
                S[c][k].append(s)
                T[c][k].append(tp)
    # Objetivo: calcular el AP de cada clase y umbral de IoU juntando todas las imágenes.
    aps = np.zeros((NC, len(IOUS)))
    for c in range(NC):
        for k in range(len(IOUS)):
            if S[c][k]:
                aps[c, k] = ap101(np.concatenate(S[c][k]), np.concatenate(T[c][k]), ngt[c])
    # Objetivo: devolver mAP@50:95 y mAP@50, promediando solo las clases que tienen labels en el split.
    has = ngt > 0
    return float(aps[has].mean()), float(aps[has, 0].mean())


@torch.no_grad()
def evaluate(model, loader, dev, amp):
    # Objetivo: medir el modelo sobre un split: predecir todas las imágenes y calcular su mAP.
    model.eval()
    gts, preds = [], []
    for x, labels in loader:
        # Objetivo: obtener las predicciones crudas del modelo (clases y cajas de cada consulta del decoder).
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            out = model(pixel_values=x.to(dev))
        # Objetivo: convertir las salidas en detecciones: confianza por clase con sigmoide y quedarse con las 300 mejores combinaciones consulta-clase.
        prob = out.logits.float().sigmoid()
        B, Q, C = prob.shape
        s, i = prob.flatten(1).topk(min(300, Q * C), dim=1)
        q, c = i // C, i % C
        b = xyxy(out.pred_boxes.float().gather(1, q[..., None].expand(-1, -1, 4))).clamp(0, 1)
        # Objetivo: guardar, imagen por imagen, las predicciones y las labels reales como arrays para el cálculo del mAP.
        for k in range(B):
            preds.append((b[k].cpu().numpy(), s[k].cpu().numpy(), c[k].cpu().numpy()))
            gts.append((xyxy(labels[k]["boxes"]).numpy(), labels[k]["class_labels"].numpy()))
    return coco_map(gts, preds)


@torch.no_grad()
def ema_update(ema, model, step):
    # Objetivo: actualizar el modelo EMA (promedio móvil de los pesos), que suele rendir mejor y se usa para validar y guardar.
    # Objetivo: calcular el decay, que arranca bajo y sube hasta ema_decay para que el promedio no quede atado a los pesos iniciales.
    d = min(args.ema_decay, (1 + step) / (10 + step))
    msd = model.state_dict()
    # Objetivo: mezclar los pesos flotantes con el decay y copiar tal cual los demás tensores (contadores y similares).
    for k, v in ema.state_dict().items():
        if v.dtype.is_floating_point:
            v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
        else:
            v.copy_(msd[k])


def main():
    from transformers import RTDetrV2ForObjectDetection

    # Objetivo: fijar las semillas para que la corrida sea reproducible.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    # Objetivo: exigir GPU CUDA, activar bf16 si la GPU lo soporta y crear la carpeta de salida.
    if not torch.cuda.is_available():
        raise SystemExit(f"torch {torch.__version__} sees no CUDA GPU")
    dev = torch.device("cuda")
    amp = torch.cuda.is_bf16_supported()
    OUT.mkdir(parents=True, exist_ok=True)

    # Objetivo: crear los datasets y loaders de train y val; en modo --smoke se recortan a unos pocos batches y 1 época.
    lim_tr = args.batch_size * 8 if args.smoke else None
    lim_va = 16 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs
    tr = YoloSet("train", True, lim_tr)
    va = YoloSet("val", False, lim_va)
    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True,
                    generator=g, persistent_workers=args.workers > 0)
    vl = DataLoader(va, 8, shuffle=False, num_workers=args.workers, collate_fn=collate)

    # Objetivo: cargar RT-DETRv2-L preentrenado cambiando la cabeza de clasificación a nuestras 6 clases.
    model = RTDetrV2ForObjectDetection.from_pretrained(
        args.model, num_labels=NC, id2label=dict(enumerate(NAMES)), label2id={n: i for i, n in enumerate(NAMES)},
        ignore_mismatched_sizes=True).to(dev)
    # Objetivo: crear el modelo EMA como copia del modelo, sin gradientes.
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)

    # Objetivo: separar los parámetros del backbone de los del resto del modelo, porque usan distinto learning rate.
    bb, rest = [], []
    for n, q in model.named_parameters():
        if q.requires_grad:
            (bb if "backbone" in n else rest).append((n, q))
    # Objetivo: armar el optimizador AdamW con 4 grupos (backbone y resto, cada uno con y sin weight decay; los vectores/bias no se decaen).
    groups = []
    for params, lr in ((bb, args.lr_backbone), (rest, args.lr)):
        groups.append({"params": [q for _, q in params if q.ndim > 1], "lr": lr, "weight_decay": args.weight_decay})
        groups.append({"params": [q for _, q in params if q.ndim <= 1], "lr": lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    # Objetivo: definir el calendario del learning rate: calentamiento lineal inicial y caída x0.1 a partir de la época lr_drop_epoch.
    steps_per_epoch = math.ceil(len(tl) / args.grad_accum)
    drop_step = args.lr_drop_epoch * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup_steps) * (0.1 if s >= drop_step else 1.0))

    # Objetivo: inicializar el estado de la corrida y, con --resume, restaurar modelo, EMA, optimizador y contadores desde last.pth.
    start_epoch, best, step, elapsed_prev = 0, -1.0, 0, 0.0
    if args.resume and (OUT / "last.pth").exists():
        ck = torch.load(OUT / "last.pth", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best, step, elapsed_prev = ck["epoch"] + 1, ck["best"], ck["step"], ck["elapsed"]
        print(f"resumed from epoch {start_epoch}")

    # Objetivo: guardar la configuración completa de la corrida (hiperparámetros, GPU, tamaños) para poder reportarla después.
    cfg = {**vars(args), "model_hf": args.model, "epochs_effective": epochs, "effective_batch": args.batch_size * args.grad_accum,
           "optimizer": "AdamW", "lr_scheduler": f"warmup {args.warmup_steps} steps, x0.1 at epoch {args.lr_drop_epoch}",
           "input_size": SIZE, "multi_scale": "per batch 512-768 step 32", "bf16": amp,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "train_images": len(tr), "val_images": len(va)}
    (OUT / "training_config.json").write_text(json.dumps(cfg, indent=2))
    # Objetivo: crear el CSV de métricas por época (si se reanuda, se conserva el existente).
    csv_path = OUT / "metrics.csv"
    if not (args.resume and csv_path.exists()):
        csv_path.write_text("epoch,train_loss,val_mAP50_95,val_mAP50,lr,epoch_time_s,elapsed_s\n")

    # Objetivo: bucle de entrenamiento, una iteración por época.
    t_start = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        run, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, (x, labels) in enumerate(tl):
            # Objetivo: entrenamiento multi-escala: cada batch se reescala a un tamaño al azar entre 512 y 768 (las cajas normalizadas no cambian).
            s = random.choice(range(512, 769, 32))
            x = F.interpolate(x.to(dev), size=(s, s), mode="bilinear", align_corners=False)
            labels = [{k: v.to(dev) for k, v in l.items()} for l in labels]
            # Objetivo: calcular la pérdida del batch y acumular su gradiente (dividido por grad_accum para simular un batch más grande).
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = model(pixel_values=x, labels=labels).loss
            (loss / args.grad_accum).backward()
            run += loss.item(); n += 1
            # Objetivo: cada grad_accum batches (o al final de la época) dar el paso del optimizador con recorte de gradiente, avanzar el calendario de lr y actualizar el EMA.
            if (it + 1) % args.grad_accum == 0 or it + 1 == len(tl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                ema_update(ema, model, step)
        # Objetivo: validar el modelo EMA al terminar la época y medir el tiempo transcurrido (por época y acumulado, incluido lo de corridas previas).
        m5095, m50 = evaluate(ema, vl, dev, amp)
        dt = time.time() - t0
        elapsed = elapsed_prev + time.time() - t_start
        lr_now = opt.param_groups[2]["lr"]
        # Objetivo: registrar la pérdida, el mAP, el lr y los tiempos de la época en metrics.csv.
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{run / max(n, 1):.5f},{m5095:.5f},{m50:.5f},{lr_now:.3e},{dt:.1f},{elapsed:.1f}\n")
        # Objetivo: si el mAP@50:95 de validación mejora el mejor hasta ahora, guardar este modelo EMA como el mejor (best_model/).
        flag = ""
        if m5095 > best:
            best = m5095
            ema.save_pretrained(OUT / "best_model")
            (OUT / "best_model" / "best_epoch.json").write_text(json.dumps({"epoch": epoch, "val_mAP50_95": m5095, "val_mAP50": m50}))
            flag = " *best*"
        # Objetivo: guardar el estado completo de la corrida (last.pth) para poder reanudar si se interrumpe.
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "step": step, "elapsed": elapsed}, OUT / "last.pth")
        print(f"epoch {epoch + 1}/{epochs} loss {run / max(n, 1):.4f} val mAP50:95 {m5095:.4f} mAP50 {m50:.4f} {dt:.0f}s{flag}", flush=True)

    # Objetivo: al terminar, guardar en la configuración el tiempo total de entrenamiento y el mejor mAP de validación.
    cfg["total_training_time_s"] = elapsed_prev + time.time() - t_start
    cfg["best_val_mAP50_95"] = best
    (OUT / "training_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"done. best val mAP@50:95 = {best:.4f}; total {cfg['total_training_time_s'] / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()

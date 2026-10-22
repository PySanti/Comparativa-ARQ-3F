"""Shared three-panel collage drawing for the visual evaluation (section 5.3.4), independent of the model.

Only the bache class is drawn. Each collage shows the same test image three times: labels as dashed blue
boxes, predictions as solid boxes (green = true positive, red = false positive), with the predictions
filtered at FACTORS x the bache threshold. Used by the eval_* scripts:
    from collage import NAMES, make_collage
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAMES = ["bache", "semaforo", "alumbrado", "basura", "arbol", "contenedor"]
BACHE = 0  # class id drawn in the collages
IOU_THR = 0.5  # same matching threshold as the evaluation
FACTORS = (0.5, 1.0, 1.5)  # left / center / right panel, multiples of the bache threshold
BLUE, GREEN, RED = (30, 110, 255), (0, 200, 0), (255, 0, 0)
LINE_W = 1  # box line width in px, kept thin so small or adjacent potholes stay visible
SCORE_FONT = 11  # confidence text size


def dashed_rect(d, box, color, width, dash=5):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        L = max(1.0, np.hypot(bx - ax, by - ay))
        for k in range(0, int(L), dash * 2):
            a, b = k / L, min(k + dash, L) / L
            d.line([ax + (bx - ax) * a, ay + (by - ay) * a, ax + (bx - ax) * b, ay + (by - ay) * b], fill=color, width=width)


def iou(a, b):
    """a: (n,4), b: (m,4), both xyxy."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-12)


def true_positive_flags(pb, ps, gb):
    """Greedy matching by descending score at IoU >= IOU_THR, one label per prediction at most.
    Returns a bool array aligned with pb: True = true positive, False = false positive."""
    tp = np.zeros(len(ps), dtype=bool)
    if len(ps) == 0 or len(gb) == 0:
        return tp
    m = iou(pb, gb)
    used = np.zeros(len(gb), dtype=bool)
    for i in np.argsort(-ps, kind="stable"):
        row = np.where(used, -1.0, m[i])
        j = int(np.argmax(row))
        if row[j] >= IOU_THR:
            used[j] = True
            tp[i] = True
    return tp


def make_collage(path, gb, gc, pb, ps, pc, thr, out_path, panel_w=640):
    """path: image file; gb/gc: label boxes (xyxy, pixels) and classes; pb/ps/pc: predicted boxes (xyxy, pixels),
    scores and classes; thr: per-class thresholds (only thr[BACHE] is used); out_path: jpg to write."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    sc = panel_w / w
    ph = int(round(h * sc))
    img = img.resize((panel_w, ph), Image.Resampling.LANCZOS)
    font = ImageFont.load_default(size=14)
    big = ImageFont.load_default(size=16)
    score_font = ImageFont.load_default(size=SCORE_FONT)
    head, foot = 26, 26
    canvas = Image.new("RGB", (panel_w * 3 + 8, head + ph + foot), (25, 25, 25))
    cd = ImageDraw.Draw(canvas)
    labels = gb[gc == BACHE]
    for k, f in enumerate(FACTORS):
        x_off = k * (panel_w + 4)
        panel = img.copy()
        d = ImageDraw.Draw(panel)
        for b in labels:
            dashed_rect(d, (b * sc).tolist(), BLUE, LINE_W)
        t = min(1.0, f * float(thr[BACHE]))
        keep = (pc == BACHE) & (ps >= t)
        boxes, scores = pb[keep], ps[keep]
        for b, s, ok in zip(boxes, scores, true_positive_flags(boxes, scores, labels)):
            col = GREEN if ok else RED
            x0, y0, x1, y1 = (b * sc).tolist()
            d.rectangle([x0, y0, x1, y1], outline=col, width=LINE_W)
            d.text((x0, max(0, y0 - SCORE_FONT - 2)), f"{s:.2f}", fill=col, font=score_font)
        canvas.paste(panel, (x_off, head))
        cd.text((x_off + 6, 4), f"{f:g}x umbral bache ({t:.3f})", fill=(255, 255, 255), font=big)
    x = 6
    for col, txt in ((BLUE, "label (discontinuo)"), (GREEN, "verdadero positivo"), (RED, "falso positivo")):
        cd.rectangle([x, head + ph + 7, x + 12, head + ph + 19], fill=col)
        cd.text((x + 16, head + ph + 5), txt, fill=(230, 230, 230), font=font)
        x += 16 + int(cd.textlength(txt, font=font)) + 20
    canvas.save(out_path, quality=90)

"""Split dataset/ into train/val without scene leakage, balancing on class-0 (bache) boxes.

Images join the same group if their 128-bit dHash is within HASH_THR bits or they come
from the same source video (GoPro GX id, DVR channel+date). Whole groups go to one side.
Copies into dataset_split/; dataset/ is left untouched.
"""
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "dataset" / "images"
LABELS = ROOT / "dataset" / "labels"
OUT = ROOT / "dataset_split"
SEED = 42
VAL_FRAC = 0.20
HASH_THR = 12
TARGET_CLASS = 0
NAMES = ["bache", "semaforo", "alumbrado", "basura", "arbol", "contenedor"]


def dhash(path):
    a = np.asarray(Image.open(path).convert("L").resize((9, 9), Image.Resampling.LANCZOS), dtype=np.int16)
    horiz = a[:8, 1:] > a[:8, :8]
    vert = a[1:, :8] > a[:8, :8]
    return np.packbits(np.concatenate([horiz.ravel(), vert.ravel()]))


def video_id(stem):
    b = re.sub(r"^[0-9a-f]{8}-", "", stem)
    m = re.match(r"(GX\d+)_frame_", b)
    if m:
        return "GP:" + m.group(1)
    m = re.match(r"(BK0D\w+?_ch\d+_main)_(\d{8})", b)
    if m:
        return "BK:" + m.group(1) + m.group(2)
    m = re.match(r"ExternDisk-(\d+_ch\d+)_(\d{8})", b)
    if m:
        return "EX:" + m.group(1) + m.group(2)
    return None


def read_boxes(stem):
    return Counter(int(l.split()[0]) for l in (LABELS / f"{stem}.txt").read_text().splitlines() if l.strip())


def main():
    files = sorted(p for p in IMAGES.iterdir() if p.is_file())
    stems = [p.stem for p in files]
    n = len(files)
    assert all((LABELS / f"{s}.txt").exists() for s in stems), "image without label"

    H = np.stack([dhash(p) for p in files])
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    rows, cols = [], []
    for i in range(n):
        d = pop[H[i] ^ H].sum(1)
        j = np.where(d[i + 1:] <= HASH_THR)[0] + i + 1
        rows += [i] * len(j)
        cols += j.tolist()
    first = {}
    for i, s in enumerate(stems):
        v = video_id(s)
        if v is None:
            continue
        if v in first:
            rows.append(first[v])
            cols.append(i)
        else:
            first[v] = i
    k, group = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)

    boxes = [read_boxes(s) for s in stems]
    gbox = [Counter() for _ in range(k)]
    for i, g in enumerate(group):
        gbox[g].update(boxes[i])

    total = sum(b[TARGET_CLASS] for b in gbox)
    target = VAL_FRAC * total
    order = list(range(k))
    random.Random(SEED).shuffle(order)
    val_groups, acc = set(), 0
    for g in order:
        c = gbox[g][TARGET_CLASS]
        if c and acc + c <= target:
            val_groups.add(g)
            acc += c
    side = ["val" if g in val_groups else "train" for g in group]

    # verification: no group spans both splits
    span = {}
    for g, s in zip(group, side):
        span.setdefault(g, set()).add(s)
    assert all(len(v) == 1 for v in span.values()), "group leaked across splits"

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)
    for p, s, st in zip(files, side, stems):
        shutil.copy2(p, OUT / "images" / s / p.name)
        shutil.copy2(LABELS / f"{st}.txt", OUT / "labels" / s / f"{st}.txt")

    (OUT / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {nm}\n" for i, nm in enumerate(NAMES))
    )

    rep = [f"seed={SEED} hash_thr={HASH_THR} val_frac={VAL_FRAC} on class {TARGET_CLASS} boxes",
           f"groups={k} largest_group_imgs={np.bincount(group).max()}", ""]
    tot_box = {s: Counter() for s in ("train", "val")}
    n_img = Counter(side)
    for b, s in zip(boxes, side):
        tot_box[s].update(b)
    rep.append(f"images: train={n_img['train']} val={n_img['val']}")
    rep.append(f"{'class':12s} {'train':>7s} {'val':>7s} {'val%':>6s}")
    for c, nm in enumerate(NAMES):
        t, v = tot_box['train'][c], tot_box['val'][c]
        rep.append(f"{c} {nm:10s} {t:7d} {v:7d} {100 * v / max(1, t + v):6.1f}")
    (OUT / "split_report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()

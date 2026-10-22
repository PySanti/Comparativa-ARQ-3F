"""Split dataset_split/val into val + test, same policy as split_dataset.py.

Groups = dHash distance <= HASH_THR OR same source video, recomputed on val images only.
Whole groups go to test until test holds ~TEST_FRAC of val's class-0 (bache) boxes.
Test files are MOVED out of val/ into test/. Train is untouched.
"""
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from split_dataset import HASH_THR, NAMES, SEED, TARGET_CLASS, dhash, video_id

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "dataset_split"
TEST_FRAC = 0.50


def read_boxes(split, stem):
    return Counter(int(l.split()[0]) for l in (SPLIT / "labels" / split / f"{stem}.txt").read_text().splitlines() if l.strip())


def main():
    if (SPLIT / "images" / "test").exists():
        raise SystemExit("dataset_split/images/test already exists; rerun split_dataset.py first to regenerate train/val")
    files = sorted(p for p in (SPLIT / "images" / "val").iterdir() if p.is_file())
    stems = [p.stem for p in files]
    n = len(files)
    assert all((SPLIT / "labels" / "val" / f"{s}.txt").exists() for s in stems), "image without label"

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

    boxes = [read_boxes("val", s) for s in stems]
    gbox = [Counter() for _ in range(k)]
    for i, g in enumerate(group):
        gbox[g].update(boxes[i])
    total = sum(b[TARGET_CLASS] for b in gbox)
    target = TEST_FRAC * total
    order = list(range(k))
    random.Random(SEED).shuffle(order)
    test_groups, acc = set(), 0
    for g in order:
        c = gbox[g][TARGET_CLASS]
        if c and acc + c <= target:
            test_groups.add(g)
            acc += c
    side = ["test" if g in test_groups else "val" for g in group]

    span = {}
    for g, s in zip(group, side):
        span.setdefault(g, set()).add(s)
    assert all(len(v) == 1 for v in span.values()), "group leaked across val/test"

    for kind in ("images", "labels"):
        (SPLIT / kind / "test").mkdir(parents=True)
    for p, s, st in zip(files, side, stems):
        if s == "test":
            shutil.move(p, SPLIT / "images" / "test" / p.name)
            shutil.move(SPLIT / "labels" / "val" / f"{st}.txt", SPLIT / "labels" / "test" / f"{st}.txt")

    yaml = (SPLIT / "data.yaml").read_text()
    if "test:" not in yaml:
        yaml = yaml.replace("val: images/val\n", "val: images/val\ntest: images/test\n")
        (SPLIT / "data.yaml").write_text(yaml)

    # report over all three splits, read from disk
    rep = [f"val/test split: seed={SEED} hash_thr={HASH_THR} test_frac={TEST_FRAC} on class {TARGET_CLASS} boxes",
           f"val groups={k} largest_group_imgs={np.bincount(group).max()}", ""]
    cnt = {}
    imgs = {}
    for s in ("train", "val", "test"):
        c = Counter()
        for lf in (SPLIT / "labels" / s).glob("*.txt"):
            c.update(int(l.split()[0]) for l in lf.read_text().splitlines() if l.strip())
        cnt[s] = c
        imgs[s] = len(list((SPLIT / "images" / s).iterdir()))
        assert imgs[s] == len(list((SPLIT / "labels" / s).iterdir())), f"{s}: image/label count mismatch"
    rep.append("images: " + " ".join(f"{s}={imgs[s]}" for s in cnt))
    rep.append(f"{'class':12s} {'train':>7s} {'val':>7s} {'test':>7s}")
    for c, nm in enumerate(NAMES):
        rep.append(f"{c} {nm:10s} " + " ".join(f"{cnt[s][c]:7d}" for s in cnt))
    (SPLIT / "split_report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()

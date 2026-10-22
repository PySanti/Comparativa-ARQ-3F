# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Compare computer-vision architectures for pothole detection on dashcam-perspective images (camera on a moving car). Early stage: no build system or tests, only standalone scripts in `scripts/`. Splitting is done; next is training/comparing architectures.

## Dataset (`dataset/`)

YOLO detection format, flat layout (the split lives in `dataset_split/`, see below):

- `dataset/images/` — ~4704 files, almost all `.jpg` (7 `.png`).
- `dataset/labels/` — 4704 `.txt`, one per image, same basename. Each line: `class_id cx cy w h` (normalized 0–1).
- Exactly 1 label file is empty (background image). Keep empty labels paired with their image when splitting.
- No `classes.txt` / `data.yaml`. Class map given by user: 0 = bache, 1 = semaforo, 2 = alumbrado, 3 = basura, 4 = arbol, 5 = contenedor. Box counts are imbalanced (0: 8272, 1: 1809, 2: 2884, 3: 671, 4: 8755, 5: 83). Class 5 is very rare, so stratify or check it lands in both splits.
- `scripts/overlay_samples.py [N]` draws boxes on N random images (seed 42) into `overlays/`, one color per class id.
- Filenames come from mixed sources (`<hash>-<id>.jpg`, `<hash>-GX0xxxxx_frame_NNNNNN.jpg` GoPro, `<hash>-BK0D...ch2_main_<timestamp>_frame_N.jpg` and `ExternDisk-..._ch9_<timestamp>` dashcam DVR). Frames from the same video are near-duplicates, so a random per-image split leaks into val. Group by source video (name prefix after the hash, minus `_frame_NNNNNN`) when splitting.

## Conventions to preserve

- Splitting must copy/move image and label together (matched by basename) into e.g. `images/{train,val}` + `labels/{train,val}`, which is the layout Ultralytics-style YOLO trainers expect.
- Use a fixed random seed so the split is reproducible.
- Windows host; paths contain a space (`comparativa arq`), so quote them.

## Split and training

- `scripts/split_dataset.py` writes `dataset_split/{images,labels}/{train,val}` + `data.yaml` + `split_report.txt` (copies; `dataset/` untouched). Groups = dHash distance <= 12 OR same source video; whole groups go to one side; val holds ~20% of class-0 (bache) boxes, seed 42. Other classes are not balanced (semaforo ~6% and contenedor 10 boxes in val).
- `scripts/train_rfdetr.py --classes all|bache [--seed N]` converts the split to COCO under `runs/rfdetr_medium_<classes>_s<seed>/coco/` (bache mode keeps only class 0, other objects become background), trains RF-DETR Medium, and writes `metrics.json` (mAP50, mAP50:95, F1 at IoU 0.5 with best confidence threshold). Written without running it: `rfdetr` is not installed and torch here is a CPU build while the GPU is an RTX 5070 (needs CUDA >= 12.8 build). Lines marked FIXME-VERIFY need checking on first run.

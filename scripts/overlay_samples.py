"""Draw YOLO boxes on a random sample of dataset images, one color per class id."""
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "dataset" / "images"
LABELS = ROOT / "dataset" / "labels"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None  # keep only images containing this class id
OUT = ROOT / (f"overlays_class{ONLY}" if ONLY is not None else "overlays")
SEED = 42

COLORS = {
    0: (255, 0, 0),      # rojo
    1: (0, 200, 0),      # verde
    2: (0, 100, 255),    # azul
    3: (255, 255, 0),    # amarillo
    4: (255, 0, 255),    # magenta
    5: (0, 255, 255),    # cian
}


def main():
    OUT.mkdir(exist_ok=True)
    stems = sorted(p.stem for p in LABELS.glob("*.txt"))
    random.Random(SEED).shuffle(stems)
    font = ImageFont.load_default(size=16)
    done = 0
    for stem in stems:
        if done >= N:
            break
        lines = (LABELS / f"{stem}.txt").read_text().splitlines()
        if ONLY is not None and not any(l.split()[:1] == [str(ONLY)] for l in lines):
            continue
        img_path = next((p for p in IMAGES.glob(stem + ".*")), None)
        if img_path is None:
            continue
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        d = ImageDraw.Draw(img)
        th = max(2, w // 400)
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            cx, cy, bw, bh = (float(x) for x in parts[1:])
            x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
            col = COLORS.get(c, (255, 255, 255))
            d.rectangle([x0, y0, x1, y1], outline=col, width=th)
            d.text((x0 + 2, max(0, y0 - 18)), str(c), fill=col, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        img.save(OUT / f"{done:03d}_{stem[:40]}.jpg", quality=90)
        done += 1
    print(f"{done} overlays -> {OUT}")


if __name__ == "__main__":
    main()

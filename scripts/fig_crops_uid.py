"""fig_crops_uid.py — show what `full` cropped for one uid.

Top: source & edit images with all crop bboxes overlaid (color-coded).
Below: each crop's src/tgt pair (what actually gets DINOv3-encoded).

Usage: python scripts/fig_crops_uid.py uid_11
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

uid = sys.argv[1] if len(sys.argv) > 1 else "uid_11"
S = "full"
d = f"data/nano3d/{uid}"
cd = f"outputs/batch3/{S}/{uid}"

meta = json.load(open(f"{cd}/metadata.json"))["crop"]
bboxes = meta["bboxes_crop"]
N = len(bboxes)
COLORS = [(255, 60, 60), (40, 180, 60), (60, 120, 255), (240, 180, 40), (200, 60, 220)]

CELL = 300
src = Image.open(f"{d}/source.png").convert("RGB").resize((CELL, CELL))
tgt = Image.open(f"{d}/edit_512.png").convert("RGB").resize((CELL, CELL))
W0, H0 = meta.get("original_size", [512, 512])
sx, sy = CELL / W0, CELL / H0

def draw_boxes(img):
    im = img.copy(); dr = ImageDraw.Draw(im)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        c = COLORS[i % len(COLORS)]
        dr.rectangle([x1*sx, y1*sy, x2*sx, y2*sy], outline=c, width=3)
        dr.text((x1*sx+3, y1*sy+2), f"c{i}", fill=c)
    return im

top = [draw_boxes(src), draw_boxes(tgt)]

# per-crop src/tgt thumbnails
def load_crop(kind, i):
    p = f"{cd}/crop_{kind}_{i}.png"
    return Image.open(p).convert("RGB").resize((CELL, CELL)) if os.path.exists(p) else Image.new("RGB", (CELL, CELL), (230, 230, 230))

# layout: row0 = source|edit ; rows = per crop [crop_src | crop_tgt]
cols = 2
rows = 1 + N
canvas = Image.new("RGB", (cols * CELL + 220, rows * CELL), (255, 255, 255))
canvas.paste(top[0], (0, 0)); canvas.paste(top[1], (CELL, 0))
for i in range(N):
    y = (1 + i) * CELL
    canvas.paste(load_crop("src", i), (0, y))
    canvas.paste(load_crop("tgt", i), (CELL, y))

dr = ImageDraw.Draw(canvas)
dr.text((6, 4), "source + crop boxes", fill=(0, 0, 0))
dr.text((CELL + 6, 4), "edit + crop boxes", fill=(0, 0, 0))
for i in range(N):
    y = (1 + i) * CELL
    x1, y1, x2, y2 = bboxes[i]
    w, h = x2 - x1, y2 - y1
    frac = 100 * (w * h) / (W0 * H0)
    c = COLORS[i % len(COLORS)]
    dr.rectangle([0, y, CELL, y + CELL], outline=c, width=4)
    dr.rectangle([CELL, y, 2 * CELL, y + CELL], outline=c, width=4)
    tx = 2 * CELL + 8
    note = "≈ GLOBAL" if frac > 60 else "local"
    dr.text((tx, y + 8),  f"crop c{i}", fill=c)
    dr.text((tx, y + 26), f"bbox {x1},{y1}", fill=(0, 0, 0))
    dr.text((tx, y + 42), f"     {x2},{y2}", fill=(0, 0, 0))
    dr.text((tx, y + 60), f"{w}x{h}px", fill=(0, 0, 0))
    dr.text((tx, y + 78), f"{frac:.0f}% of image", fill=(0, 0, 0))
    dr.text((tx, y + 96), note, fill=c)
    dr.text((6, y + 6), "crop_src", fill=(255, 255, 255))
    dr.text((CELL + 6, y + 6), "crop_tgt", fill=(255, 255, 255))

out = f"outputs/crops_{uid}.png"
canvas.save(out)
print("saved", out, canvas.size, "n_crops", N)

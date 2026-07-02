"""
fig_pure_wins.py — for every uid where full_pure has the best DINO-I v2 of the
three settings, build one comparison row:

  [input source] [edit target] [no_crop] [full] [full_pure*] [params]

Renders are the textured canonical view of each setting's result_00.glb
(outputs/batch3/<setting>/<uid>/). The winning column (full_pure) gets a green
frame. Per-render DINO-I v2/v3 are printed on each cell; the params panel lists
CD->src / #crops / voxels for all three settings.
"""
import glob, json, os, sys
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.render_grid import render_cell, H, W  # reuse textured single-view renderer

B = "outputs/batch3"
CFG = ["full", "full_pure", "no_crop"]


def load_rows():
    rows = {}
    for p in glob.glob(f"{B}/uid_*.json"):
        r = json.load(open(p)); rows[r["uid"]] = r
    return rows


def d2(r, c):
    d = r.get(c) or {}
    return None if "error" in d else (d.get("dino") or {}).get("dinov2")


def pure_winners(rows):
    out = []
    for uid, r in rows.items():
        v = {c: d2(r, c) for c in CFG}
        if any(x is None for x in v.values()):
            continue
        if max(v, key=lambda c: v[c]) == "full_pure":
            others = max(v["full"], v["no_crop"])
            out.append((uid, v["full_pure"] - others))
    out.sort(key=lambda t: t[1], reverse=True)
    return [u for u, _ in out]


def dinos(r, c):
    d = (r.get(c) or {}).get("dino") or {}
    return d.get("dinov2"), d.get("dinov3")


def main():
    rows = load_rows()
    uids = pure_winners(rows)
    print(f"{len(uids)} pure-winning uids: {uids}")

    glctx = dr.RasterizeCudaContext()
    cols = ["input", "edit target", "no_crop", "full", "full_pure*"]
    PW = 470                       # params panel width
    rowh = H
    canvas = np.full((rowh * len(uids), W * len(cols) + PW, 3), 255, np.uint8)

    for i, uid in enumerate(uids):
        r = rows[uid]
        cells = []
        # input source image / edit target image
        src = Image.open(f"data/nano3d/{uid}/source.png").convert("RGB").resize((W, H))
        tgt = Image.open(f"data/nano3d/{uid}/edit_512.png").convert("RGB").resize((W, H))
        cells += [np.asarray(src), np.asarray(tgt)]
        for c in ["no_crop", "full", "full_pure"]:
            glb = f"{B}/{c}/{uid}/result_00.glb"
            cells.append(render_cell(glb, glctx) if os.path.exists(glb)
                         else np.full((H, W, 3), 235, np.uint8))
        strip = np.concatenate(cells, axis=1)
        canvas[i * rowh:(i + 1) * rowh, :W * len(cols)] = strip

    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    # column headers
    for j, c in enumerate(cols):
        col = (10, 140, 20) if c.endswith("*") else (200, 20, 20)
        d.text((j * W + 6, 4), c, fill=col)
    d.text((W * len(cols) + 8, 4), "params (CD->src / crops / vox)", fill=(30, 30, 30))

    for i, uid in enumerate(uids):
        r = rows[uid]
        y0 = i * rowh
        d.text((4, y0 + 16), uid, fill=(20, 20, 210))
        # per-render DINO scores
        for k, c in enumerate(["no_crop", "full", "full_pure"]):
            x = (2 + k) * W
            v2, v3 = dinos(r, c)
            if v2 is not None:
                d.text((x + 6, y0 + H - 30), f"D2 {v2:.3f}", fill=(0, 0, 0))
                d.text((x + 6, y0 + H - 16), f"D3 {v3:.3f}", fill=(0, 0, 0))
        # green frame on winning full_pure cell
        xp = 4 * W
        d.rectangle([xp + 1, y0 + 1, xp + W - 2, y0 + H - 2], outline=(10, 160, 20), width=4)
        # params panel
        px = W * len(cols) + 10
        lines = [f"{uid}   src-baseline D2={ (r.get('source_dino') or {}).get('dinov2', float('nan')):.3f}"]
        for c in CFG:
            dd = r.get(c) or {}
            if "error" in dd:
                lines.append(f"{c:<10} ERROR")
            else:
                v2, v3 = dinos(r, c)
                lines.append(f"{c:<10} D2={v2:.3f} D3={v3:.3f} | CD={dd.get('cd_src'):.3f} "
                             f"crops={dd.get('n_crops')} vox={dd.get('voxels')}")
        for li, ln in enumerate(lines):
            col = (10, 130, 20) if ln.strip().startswith("full_pure") else (30, 30, 30)
            d.text((px, y0 + 26 + li * 16), ln, fill=col)

    out = "outputs/pure_wins.png"
    im.save(out)
    print("saved", out, im.size)


if __name__ == "__main__":
    main()

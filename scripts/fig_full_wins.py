"""
fig_full_wins.py — comparison rows for the cases where `full` (multi-crop) wins
DECISIVELY (best DINO-I v2 of the three AND margin over 2nd-best > THRESH).

Same layout as fig_pure_wins.py:
  [input] [edit target] [no_crop] [full*] [full_pure] [params]
The winning column (full) gets the green frame.
"""
import glob, json, os, sys
import numpy as np
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.render_grid import render_cell, H, W

B = "outputs/batch3"
CFG = ["full", "full_pure", "no_crop"]
THRESH = 0.05


def d2(r, c):
    d = r.get(c) or {}
    return None if "error" in d else (d.get("dino") or {}).get("dinov2")


def dinos(r, c):
    d = (r.get(c) or {}).get("dino") or {}
    return d.get("dinov2"), d.get("dinov3")


def main():
    rows = {}
    for p in glob.glob(f"{B}/uid_*.json"):
        r = json.load(open(p)); rows[r["uid"]] = r

    picks = []
    for uid, r in rows.items():
        v = {c: d2(r, c) for c in CFG}
        if any(x is None for x in v.values()):
            continue
        if max(v, key=lambda c: v[c]) != "full":
            continue
        margin = v["full"] - sorted(v.values(), reverse=True)[1]
        if margin > THRESH:
            picks.append((uid, margin))
    picks.sort(key=lambda t: t[1], reverse=True)
    uids = [u for u, _ in picks]
    print(f"{len(uids)} decisive full wins (>{THRESH}): {uids}")

    glctx = dr.RasterizeCudaContext()
    cols = ["input", "edit target", "no_crop", "full*", "full_pure"]
    order = ["no_crop", "full", "full_pure"]      # render order in cells
    PW = 470
    canvas = np.full((H * len(uids), W * len(cols) + PW, 3), 255, np.uint8)

    for i, uid in enumerate(uids):
        cells = []
        src = Image.open(f"data/nano3d/{uid}/source.png").convert("RGB").resize((W, H))
        tgt = Image.open(f"data/nano3d/{uid}/edit_512.png").convert("RGB").resize((W, H))
        cells += [np.asarray(src), np.asarray(tgt)]
        for c in order:
            glb = f"{B}/{c}/{uid}/result_00.glb"
            cells.append(render_cell(glb, glctx) if os.path.exists(glb)
                         else np.full((H, W, 3), 235, np.uint8))
        canvas[i * H:(i + 1) * H, :W * len(cols)] = np.concatenate(cells, axis=1)

    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    for j, c in enumerate(cols):
        col = (10, 140, 20) if c.endswith("*") else (200, 20, 20)
        d.text((j * W + 6, 4), c, fill=col)
    d.text((W * len(cols) + 8, 4), "params (CD->src / crops / vox)", fill=(30, 30, 30))

    for i, uid in enumerate(uids):
        r = rows[uid]; y0 = i * H
        d.text((4, y0 + 16), uid, fill=(20, 20, 210))
        for k, c in enumerate(order):
            x = (2 + k) * W
            v2, v3 = dinos(r, c)
            if v2 is not None:
                d.text((x + 6, y0 + H - 30), f"D2 {v2:.3f}", fill=(0, 0, 0))
                d.text((x + 6, y0 + H - 16), f"D3 {v3:.3f}", fill=(0, 0, 0))
        # green frame on winning full cell (index 3 -> render col 'full' is order idx1 -> 2+1=3)
        xp = 3 * W
        d.rectangle([xp + 1, y0 + 1, xp + W - 2, y0 + H - 2], outline=(10, 160, 20), width=4)
        px = W * len(cols) + 10
        lines = [f"{uid}   src-baseline D2={(r.get('source_dino') or {}).get('dinov2', float('nan')):.3f}"]
        for c in CFG:
            dd = r.get(c) or {}
            if "error" in dd:
                lines.append(f"{c:<10} ERROR")
            else:
                v2, v3 = dinos(r, c)
                lines.append(f"{c:<10} D2={v2:.3f} D3={v3:.3f} | CD={dd.get('cd_src'):.3f} "
                             f"crops={dd.get('n_crops')} vox={dd.get('voxels')}")
        for li, ln in enumerate(lines):
            col = (10, 130, 20) if ln.strip().startswith("full ") else (30, 30, 30)
            d.text((px, y0 + 26 + li * 16), ln, fill=col)

    out = "outputs/full_wins.png"
    im.save(out)
    print("saved", out, im.size)


if __name__ == "__main__":
    main()

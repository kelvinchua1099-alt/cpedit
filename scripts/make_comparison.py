"""
make_comparison.py — side-by-side render comparison for the crop ablation.

For each uid it lays out one row:
  source.png | edit_512.png (2D target) | GT mesh | full pred | no_crop pred

Meshes are rendered headless (pyrender + EGL), normalised to a unit cube, from a
single 3/4 view so all columns are framed identically. PSNR/LPIPS/CLIP from the
metrics json (if present) are annotated under each prediction.

Usage:
  PYOPENGL_PLATFORM=egl python scripts/make_comparison.py \
      --results_root results_run --data_root data/nano3d \
      --modes full no_crop --out results_run/comparison_grid.png
"""
from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

VIEW = (20.0, 35.0)   # elevation, azimuth — single 3/4 view
TILE = 320            # px per cell


def _camera_pose(elev_deg, azim_deg, radius=2.2):
    el, az = math.radians(elev_deg), math.radians(azim_deg)
    eye = np.array([radius * math.cos(el) * math.sin(az),
                    radius * math.sin(el),
                    radius * math.cos(el) * math.cos(az)])
    fwd = -eye / np.linalg.norm(eye)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0] = right; pose[:3, 1] = up2; pose[:3, 2] = -fwd; pose[:3, 3] = eye
    return pose


def render_mesh(path, size=TILE):
    if not path or not os.path.exists(path):
        return np.full((size, size, 3), 235, np.uint8)
    tm = trimesh.load(path, force="scene")
    b = tm.bounds
    center = b.mean(0)
    scale = float((b[1] - b[0]).max()) or 1.0
    r = pyrender.OffscreenRenderer(size, size)
    scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.5, 0.5, 0.5])
    for _, g in tm.geometry.items():
        g = g.copy(); g.apply_translation(-center); g.apply_scale(1.0 / scale)
        scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))
    pose = _camera_pose(*VIEW)
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=pose)
    color, _ = r.render(scene)
    r.delete()
    return np.ascontiguousarray(color, np.uint8)


def load_img(path, size=TILE):
    if not path or not os.path.exists(path):
        return np.full((size, size, 3), 235, np.uint8)
    im = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(im, np.uint8)


def find_pred(mode_dir):
    for n in ("result_00.glb", "result_0.glb"):
        p = os.path.join(mode_dir, n)
        if os.path.exists(p):
            return p
    return None


def load_metrics(results_root, mode, suffix=""):
    p = os.path.join(results_root, f"metrics_{mode}{suffix}.json")
    if os.path.exists(p):
        return json.load(open(p)).get("per_item", {})
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results_run")
    ap.add_argument("--data_root", default="data/nano3d")
    ap.add_argument("--modes", nargs="+", default=["full", "no_crop"])
    ap.add_argument("--metrics_suffix", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.results_root, "comparison_grid.png")

    metr = {m: load_metrics(args.results_root, m, args.metrics_suffix) for m in args.modes}

    uids = sorted(d for d in os.listdir(args.data_root)
                  if os.path.isdir(os.path.join(args.data_root, d)))

    cols = ["source", "target (edit)", "GT (tar_mesh)"] + [f"pred: {m}" for m in args.modes]
    ncol = len(cols)
    pad, hdr = 8, 40
    W = ncol * (TILE + pad) + pad
    H = hdr + len(uids) * (TILE + hdr) + pad

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = small = ImageFont.load_default()

    for c, name in enumerate(cols):
        x = pad + c * (TILE + pad)
        draw.text((x + 6, 10), name, fill=(0, 0, 0), font=font)

    for r, uid in enumerate(uids):
        y0 = hdr + r * (TILE + hdr)
        draw.text((pad, y0 + TILE + 6), uid, fill=(0, 0, 0), font=font)
        obj = os.path.join(args.data_root, uid)
        cells = [
            load_img(os.path.join(obj, "source.png")),
            load_img(os.path.join(obj, "edit_512.png")),
            render_mesh(os.path.join(obj, "tar_mesh.glb")),
        ]
        labels = ["", "", ""]
        for m in args.modes:
            pred = find_pred(os.path.join(args.results_root, m, uid))
            cells.append(render_mesh(pred))
            mm = metr[m].get(uid, {})
            if mm:
                labels.append(
                    f"PSNR {mm.get('psnr',0):.1f}  LPIPS {mm.get('lpips',0):.2f}  "
                    f"CLIP {mm.get('clip_sim',0):.2f}  Dir {mm.get('clip_dir',0):.2f}"
                )
            else:
                labels.append("(no metrics)" if pred else "(no pred)")
        for c, (img, lab) in enumerate(zip(cells, labels)):
            x = pad + c * (TILE + pad)
            canvas.paste(Image.fromarray(img), (x, y0))
            if lab:
                draw.text((x + 4, y0 + TILE + 22), lab, fill=(40, 40, 40), font=small)

    canvas.save(out)
    print(f"wrote {out}  ({W}x{H})")


if __name__ == "__main__":
    main()

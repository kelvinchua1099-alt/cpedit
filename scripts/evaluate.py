"""
evaluate.py — render-based metrics for the VS3D crop-guidance ablation.

For every object that produced an edited mesh (`result_00.glb`) under
`results/<mode>/<uid>/`, render 8 fixed views of the prediction, the dataset
ground-truth edited mesh (`tar_mesh.glb`) and the source mesh (`src_mesh.glb`),
then compute:

  PSNR  ↑   pred vs GT, mean over 8 views          (skimage)
  SSIM  ↑   pred vs GT, mean over 8 views          (skimage, channel_axis)
  LPIPS ↓   pred vs GT, mean over 8 views          (lpips, net='alex')
  CLIP-Sim ↑ pred-render vs edited target image,   (open_clip ViT-B-32)
            MAX over 8 views (edit is most visible in one view)
  Identity-LPIPS ↓  src-render vs pred-render, mean over 8 views

Rendering is headless via pyrender + EGL. Each mesh is normalised to a unit
cube before rendering so prediction and ground truth are framed identically.

Outputs:
  results/metrics_<mode>.json     per-item + aggregate mean/std
  results/metrics_comparison.csv  modes side by side
Failed objects (missing result_00.glb) are counted and reported, never
silently dropped from the failure count.

Usage:
  PYOPENGL_PLATFORM=egl python scripts/evaluate.py \
      --results_root results --data_root data/nano3d --modes full no_crop
"""
from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 8 fixed views (elevation_deg, azimuth_deg)
VIEWS = [
    (0,   0), (0,  90), (0, 180), (0, 270),
    (30, 45), (30, 135), (30, 225), (30, 315),
]
RENDER_SIZE = 512


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _camera_pose(elev_deg: float, azim_deg: float, radius: float = 2.2) -> np.ndarray:
    el = math.radians(elev_deg)
    az = math.radians(azim_deg)
    eye = np.array([radius * math.cos(el) * math.sin(az),
                    radius * math.sin(el),
                    radius * math.cos(el) * math.cos(az)])
    fwd = -eye / np.linalg.norm(eye)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up2
    pose[:3, 2] = -fwd
    pose[:3, 3] = eye
    return pose


def render_views(mesh_path: str, size: int = RENDER_SIZE):
    """Render all VIEWS of a GLB, normalised to a unit cube. Returns list of HxWx3 uint8."""
    tm = trimesh.load(mesh_path, force="scene")
    bounds = tm.bounds
    center = bounds.mean(0)
    scale = float((bounds[1] - bounds[0]).max())
    if scale <= 0:
        scale = 1.0

    imgs = []
    renderer = pyrender.OffscreenRenderer(size, size)
    for elev, azim in VIEWS:
        scene = pyrender.Scene(bg_color=[255, 255, 255, 0],
                               ambient_light=[0.5, 0.5, 0.5])
        for _, g in tm.geometry.items():
            g = g.copy()
            g.apply_translation(-center)
            g.apply_scale(1.0 / scale)
            scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))
        pose = _camera_pose(elev, azim)
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=pose)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=pose)
        color, _ = renderer.render(scene)
        # pyrender may return a flipped view (negative strides); torch.from_numpy
        # and several ops need a contiguous copy.
        imgs.append(np.ascontiguousarray(color, dtype=np.uint8))
    renderer.delete()
    return imgs


# ---------------------------------------------------------------------------
# Metric models (loaded once)
# ---------------------------------------------------------------------------

class Metrics:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        import lpips
        self.lpips = lpips.LPIPS(net="alex").to(self.device).eval()
        import open_clip
        # OpenAI CLIP ViT-B/32 uses QuickGELU; the plain "ViT-B-32" arch defaults
        # to nn.GELU and warns/degrades with the openai weights.
        self.clip, _, self.clip_pre = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai"
        )
        self.clip = self.clip.to(self.device).eval()
        self.clip_tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    @staticmethod
    def _to_lpips(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return (t * 2.0 - 1.0).unsqueeze(0)

    @torch.no_grad()
    def lpips_pair(self, a: np.ndarray, b: np.ndarray) -> float:
        ta = self._to_lpips(a).to(self.device)
        tb = self._to_lpips(b).to(self.device)
        return float(self.lpips(ta, tb).item())

    @torch.no_grad()
    def clip_feat_img(self, img: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(img)
        x = self.clip_pre(pil).unsqueeze(0).to(self.device)
        f = self.clip.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Per-object evaluation
# ---------------------------------------------------------------------------

def eval_object(metrics: Metrics, pred_glb: str, gt_glb: str, src_glb: str,
                tgt_img_path: str, src_img_path: str | None = None) -> dict:
    pred = render_views(pred_glb)
    gt = render_views(gt_glb)
    src = render_views(src_glb)

    psnrs, ssims, lpips_pg, ident = [], [], [], []
    for p, g, s in zip(pred, gt, src):
        psnrs.append(peak_signal_noise_ratio(g, p, data_range=255))
        ssims.append(structural_similarity(g, p, channel_axis=2, data_range=255))
        lpips_pg.append(metrics.lpips_pair(p, g))
        ident.append(metrics.lpips_pair(s, p))

    # CLIP: max cosine sim over views between pred render and the 2D edited target
    tgt = np.asarray(Image.open(tgt_img_path).convert("RGB"))
    tgt_f = metrics.clip_feat_img(tgt)
    clip_sims = [float((metrics.clip_feat_img(p) @ tgt_f.T).item()) for p in pred]

    out = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpips_pg)),
        "clip_sim": float(np.max(clip_sims)),
        "identity_lpips": float(np.mean(ident)),
    }

    # Directional CLIP ↑: does the 3D edit move in the same CLIP direction as the
    # 2D edit?  Δ_img = CLIP(edit) − CLIP(source);  Δ_pred = CLIP(pred) − CLIP(src).
    # cos(Δ_pred, Δ_img), mean over views. Unlike absolute CLIP this is NOT
    # dominated by the unchanged bulk, so it can actually resolve the ablation.
    if src_img_path and os.path.exists(src_img_path):
        src_img = np.asarray(Image.open(src_img_path).convert("RGB"))
        d_img = (tgt_f - metrics.clip_feat_img(src_img))
        d_img = d_img / (d_img.norm(dim=-1, keepdim=True) + 1e-8)
        dirs = []
        for p, s in zip(pred, src):
            d_pred = metrics.clip_feat_img(p) - metrics.clip_feat_img(s)
            d_pred = d_pred / (d_pred.norm(dim=-1, keepdim=True) + 1e-8)
            dirs.append(float((d_pred @ d_img.T).item()))
        out["clip_dir"] = float(np.mean(dirs))

    return out


def aggregate(per_item: dict) -> dict:
    keys = ["psnr", "ssim", "lpips", "clip_sim", "clip_dir", "identity_lpips"]
    agg = {}
    for k in keys:
        vals = [v[k] for v in per_item.values() if k in v]
        agg[k] = {"mean": float(np.mean(vals)) if vals else float("nan"),
                  "std": float(np.std(vals)) if vals else float("nan")}
    return agg


def find_pred_glb(mode_dir: str) -> str | None:
    for name in ("result_00.glb", "result_0.glb"):
        p = os.path.join(mode_dir, name)
        if os.path.exists(p):
            return p
    return None


def run_mode(metrics: Metrics, results_root: str, data_root: str, mode: str,
             only_uids=None):
    mode_root = os.path.join(results_root, mode)
    per_item, failures = {}, []
    if not os.path.isdir(mode_root):
        return {"per_item": {}, "aggregate": aggregate({}), "failures": []}, []

    for uid in sorted(os.listdir(mode_root)):
        if only_uids and uid not in only_uids:
            continue
        mode_dir = os.path.join(mode_root, uid)
        if not os.path.isdir(mode_dir):
            continue
        pred = find_pred_glb(mode_dir)
        gt = os.path.join(data_root, uid, "tar_mesh.glb")
        src = os.path.join(data_root, uid, "src_mesh.glb")
        tgt_img = os.path.join(data_root, uid, "edit_512.png")
        src_img = os.path.join(data_root, uid, "source.png")
        if pred is None or not (os.path.exists(gt) and os.path.exists(src) and os.path.exists(tgt_img)):
            failures.append({"uid": uid, "reason": "missing pred glb or GT files"})
            continue
        try:
            per_item[uid] = eval_object(metrics, pred, gt, src, tgt_img, src_img)
            print(f"[{mode}] {uid}: " + ", ".join(f"{k}={v:.3f}" for k, v in per_item[uid].items()))
        except Exception as e:
            failures.append({"uid": uid, "reason": f"{type(e).__name__}: {e}"})
            print(f"[{mode}] {uid}: EVAL ERROR {e}")

    return {"per_item": per_item, "aggregate": aggregate(per_item),
            "failures": failures}, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--data_root", default="data/nano3d")
    ap.add_argument("--modes", nargs="+", default=["full", "no_crop"])
    ap.add_argument("--device", default="cuda", help="cuda or cpu (cpu avoids "
                    "GPU contention with a running experiment)")
    ap.add_argument("--uids", nargs="*", default=None,
                    help="restrict to these uids (default: all)")
    ap.add_argument("--out_suffix", default="",
                    help="suffix for metrics_*.json (e.g. _partial)")
    args = ap.parse_args()

    metrics = Metrics(device=args.device)
    all_results = {}
    for mode in args.modes:
        res, fails = run_mode(metrics, args.results_root, args.data_root, mode,
                              only_uids=args.uids)
        all_results[mode] = res
        out = os.path.join(args.results_root, f"metrics_{mode}{args.out_suffix}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {out}  ({len(res['per_item'])} ok, {len(fails)} failed)")

    # comparison CSV
    keys = ["psnr", "ssim", "lpips", "clip_sim", "clip_dir", "identity_lpips"]
    csv_path = os.path.join(args.results_root, f"metrics_comparison{args.out_suffix}.csv")
    with open(csv_path, "w") as f:
        f.write("metric," + ",".join(f"{m}_mean,{m}_std" for m in args.modes) + "\n")
        for k in keys:
            row = [k]
            for m in args.modes:
                a = all_results[m]["aggregate"][k]
                row += [f"{a['mean']:.4f}", f"{a['std']:.4f}"]
            f.write(",".join(row) + "\n")
    print(f"wrote {csv_path}")

    # terminal summary
    n = {m: len(all_results[m]["per_item"]) for m in args.modes}
    labels = {"psnr": "PSNR ↑", "ssim": "SSIM ↑", "lpips": "LPIPS ↓",
              "clip_sim": "CLIP-Sim ↑", "clip_dir": "CLIP-Dir ↑",
              "identity_lpips": "Identity LPIPS ↓"}
    print("\n" + "=" * 60)
    print(f"{'metric':<18}" + "".join(f"{m:>20}" for m in args.modes))
    print("-" * 60)
    for k in keys:
        cells = ""
        for m in args.modes:
            a = all_results[m]["aggregate"][k]
            cells += f"{a['mean']:>10.3f} ± {a['std']:<7.3f}"
        print(f"{labels[k]:<18}{cells}")
    print("=" * 60)
    print("Objects: " + ", ".join(f"{m}={n[m]}" for m in args.modes)
          + " from yejunliang23/Nano3D-Edit-100k")
    for m in args.modes:
        fails = all_results[m]["failures"]
        if fails:
            print(f"  [{m}] FAILED {len(fails)}: " + ", ".join(x['uid'] for x in fails))


if __name__ == "__main__":
    main()

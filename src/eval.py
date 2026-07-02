"""
eval.py — lightweight evaluation + logging for VS3D edits.

Everything here degrades gracefully: a missing optional dependency (lpips,
open_clip) or a missing input file yields ``None`` for that metric instead of
crashing the run.  This matches CODING_AGENT.md's "implement only lightweight
metrics first" guideline.

Metrics
-------
Image space (needs a rendered/target image pair):
    psnr, ssim         — scikit-image
    lpips              — lpips (optional)
    clip_sim           — open_clip cosine similarity of two images (optional)

Mesh space (needs .glb files; uses trimesh + scipy):
    chamfer_to_gt      — Chamfer distance between the edited result and tar_mesh
                         (the ground-truth edit)  → lower is a better edit
    chamfer_to_src     — Chamfer distance between the result and src_mesh
                         (identity reference)      → context for how much moved
    n_vertices/n_faces — geometry size of the result

Public API
----------
    save_run_summary(output_dir, metrics)          -> path to metrics.json
    evaluate_result(output_dir, meta, ...)          -> metrics dict (also saved)
    chamfer_distance(mesh_a, mesh_b, ...)           -> float
    image_metrics(img_pred, img_ref)                -> dict
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_run_summary(output_dir: str, metrics: Dict[str, Any]) -> str:
    """Write ``metrics`` to ``<output_dir>/metrics.json`` and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(_jsonable(metrics), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Mesh loading + Chamfer distance
# ---------------------------------------------------------------------------

def _load_mesh(path: str):
    """Load a .glb/.obj/.ply into a single trimesh.Trimesh (concatenate scenes)."""
    import trimesh

    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(
            [g for g in m.geometry.values() if hasattr(g, "vertices")]
        )
    return m


def _normalize_unit(v: np.ndarray) -> np.ndarray:
    """Center a point cloud and scale its bbox diagonal to 1 (scale-invariant)."""
    v = v - v.mean(axis=0, keepdims=True)
    diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    if diag > 1e-9:
        v = v / diag
    return v


def chamfer_distance(
    mesh_a_path: str,
    mesh_b_path: str,
    n_samples: int = 20000,
    normalize: bool = True,
    seed: int = 0,
) -> Optional[float]:
    """
    Symmetric Chamfer distance between two meshes' sampled surface points.
    Returns None if either mesh is unreadable/empty.  Meshes are each normalised
    to a unit bbox diagonal by default so the metric is scale/translation robust.
    """
    try:
        from scipy.spatial import cKDTree

        ma, mb = _load_mesh(mesh_a_path), _load_mesh(mesh_b_path)
        if ma is None or mb is None or len(ma.vertices) == 0 or len(mb.vertices) == 0:
            return None

        rng = np.random.RandomState(seed)
        pa = ma.sample(n_samples).astype(np.float64)
        pb = mb.sample(n_samples).astype(np.float64)
        if normalize:
            pa, pb = _normalize_unit(pa), _normalize_unit(pb)

        d_ab = cKDTree(pb).query(pa, k=1)[0]
        d_ba = cKDTree(pa).query(pb, k=1)[0]
        return float(d_ab.mean() + d_ba.mean())
    except Exception:
        return None


def mesh_stats(mesh_path: str) -> Dict[str, Any]:
    try:
        m = _load_mesh(mesh_path)
        return {
            "n_vertices": int(len(m.vertices)),
            "n_faces": int(len(m.faces)),
            "watertight": bool(m.is_watertight),
        }
    except Exception:
        return {"n_vertices": None, "n_faces": None, "watertight": None}


# ---------------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------------

def _to_rgb_array(img, size=None) -> np.ndarray:
    from PIL import Image

    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img)


def image_metrics(img_pred, img_ref) -> Dict[str, Optional[float]]:
    """PSNR/SSIM (+ optional LPIPS, CLIP) between two RGB images (paths or PIL)."""
    out: Dict[str, Optional[float]] = {"psnr": None, "ssim": None,
                                       "lpips": None, "clip_sim": None}
    try:
        from PIL import Image

        ref = img_ref if not isinstance(img_ref, str) else Image.open(img_ref)
        ref = ref.convert("RGB")
        a = _to_rgb_array(img_pred, size=ref.size)
        b = np.asarray(ref)
    except Exception:
        return out

    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim

        out["psnr"] = float(psnr(b, a, data_range=255))
        out["ssim"] = float(ssim(b, a, channel_axis=-1, data_range=255))
    except Exception:
        pass

    try:  # optional LPIPS
        import torch
        import lpips as lpips_lib

        net = lpips_lib.LPIPS(net="alex", verbose=False)
        def _t(x):
            t = torch.from_numpy(x).float().permute(2, 0, 1)[None] / 127.5 - 1.0
            return t
        with torch.no_grad():
            out["lpips"] = float(net(_t(a), _t(b)).item())
    except Exception:
        pass

    try:  # optional CLIP cosine similarity
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        from PIL import Image
        with torch.no_grad():
            fa = model.encode_image(preprocess(Image.fromarray(a)).unsqueeze(0))
            fb = model.encode_image(preprocess(Image.fromarray(b)).unsqueeze(0))
            fa = fa / fa.norm(dim=-1, keepdim=True)
            fb = fb / fb.norm(dim=-1, keepdim=True)
            out["clip_sim"] = float((fa * fb).sum().item())
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def evaluate_result(
    output_dir: str,
    meta: Dict[str, Any],
    gt_mesh: Optional[str] = None,
    src_mesh: Optional[str] = None,
    tgt_image: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Compute whatever metrics the available inputs allow and (optionally) save
    them to ``<output_dir>/metrics.json``.

    Args
    ----
    output_dir : run output dir (must contain result_00.glb from the pipeline)
    meta       : the pipeline's returned meta dict (for n_voxels etc.)
    gt_mesh    : path to tar_mesh.glb (ground-truth edit)         -> chamfer_to_gt
    src_mesh   : path to src_mesh.glb (identity reference)        -> chamfer_to_src
    tgt_image  : path to the 2D edit target image                 -> image metrics
    """
    metrics: Dict[str, Any] = {
        "exp_name": meta.get("exp_name"),
        "n_voxels_tgt": meta.get("n_voxels_tgt"),
        "crop_guidance_active": meta.get("crop_guidance_active"),
    }

    glb_paths = meta.get("glb_paths") or []
    result_glb = glb_paths[0] if glb_paths else os.path.join(output_dir, "result_00.glb")

    if os.path.exists(result_glb):
        metrics["result_mesh"] = mesh_stats(result_glb)
        if gt_mesh and os.path.exists(gt_mesh):
            metrics["chamfer_to_gt"] = chamfer_distance(result_glb, gt_mesh)
        if src_mesh and os.path.exists(src_mesh):
            metrics["chamfer_to_src"] = chamfer_distance(result_glb, src_mesh)
    else:
        metrics["result_mesh"] = None
        metrics["note"] = f"result glb not found at {result_glb}"

    # Sanity reference: how far the GT edit moved from the source.
    if gt_mesh and src_mesh and os.path.exists(gt_mesh) and os.path.exists(src_mesh):
        metrics["chamfer_gt_vs_src"] = chamfer_distance(gt_mesh, src_mesh)

    # Image-space (compares the rendered result view to the edit target, if a
    # render exists; otherwise skipped).
    render = os.path.join(output_dir, "renders_00", "view_000.png")
    if tgt_image and os.path.exists(tgt_image) and os.path.exists(render):
        metrics["image"] = image_metrics(render, tgt_image)

    if save:
        save_run_summary(output_dir, metrics)
    return metrics

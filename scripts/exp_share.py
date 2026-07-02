"""
exp_share.py — is the crop computation reusable from the global one?

Sets up one uid through Stage-1 inputs (x_src, global conds, crop conds) and, at
a few real denoising steps, measures how similar the crop signal is to the
global signal at two levels:

  (1) COND level   : cosine(crop_cond, global_cond)  for tgt and src.
                     Tests the literal "crop is a sub-region of global" idea.
  (2) VELOCITY level: for the edit-relevant velocity difference
                        v_Δ = v_tgt - v_src,
                     compare v_Δ_crop_i against v_Δ_global via
                       • relative L2  ||v_c - v_g|| / ||v_g||
                       • mean per-voxel cosine
                     If v_Δ_crop ≈ v_Δ_global, the crop adds little / is reusable.
                     If very different, it carries independent signal.
  (3) PHI identity : model(z,phi) recomputed in the crop branch is bit-identical
                     to the global one  ->  confirms Tier-1 dedup is EXACT.

Usage: python scripts/exp_share.py [uid]   (default uid_1)
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
from PIL import Image

from scripts.run import load_config
from src.trellis_wrapper import TrellisWrapper
from src.vs3d import VS3DEditor
from src.pipeline import EditPipeline


def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def voxel_cos(vc, vg):
    # vc,vg: [B,C,R,R,R] -> per-voxel cosine over channel dim, averaged on active voxels
    num = (vc * vg).sum(1)
    den = vc.norm(dim=1) * vg.norm(dim=1) + 1e-8
    c = num / den
    active = vg.norm(dim=1) > vg.norm(dim=1).mean() * 0.1
    return float(c[active].mean()), float(active.float().mean())


def main():
    uid = sys.argv[1] if len(sys.argv) > 1 else "uid_1"
    d = f"data/nano3d/{uid}"
    img_src = Image.open(f"{d}/source.png")
    img_tgt = Image.open(f"{d}/edit_512.png")

    cfg = load_config("full", ["model.trellis.resolution=512"])
    print(f"[exp] loading backbone (uid={uid}) ...", flush=True)
    wrapper = TrellisWrapper.from_config(cfg)
    editor = VS3DEditor.from_config(wrapper, cfg)
    pipeline = EditPipeline(wrapper, editor, cfg)

    res = cfg.model.trellis.resolution
    crops, crop_meta = pipeline._preprocess(img_src, img_tgt)
    shape_slat_src, tex_slat_src, slat_res = wrapper.run_source(
        img_src, seed=int(cfg.get("seed", 42)), preprocess=True, pipeline_type="512")
    src_cond, tgt_cond, crop_conds = pipeline._build_conditions(
        img_src, img_tgt, crops, crop_meta, res)
    x_src = pipeline._build_stage1_src_latent(shape_slat_src)
    wrapper.offload_for_stage1()

    torch.set_grad_enabled(False)   # measurement only — no autograd graphs (avoids OOM)
    N = len(crop_conds or [])
    print(f"\n### n_crops = {N}")
    print(f"### global tgt_cond shape = {tuple(tgt_cond.cond.shape)}")

    # ---------- (1) COND level ----------
    print("\n=== (1) COND cosine vs global ===")
    for i, (csc, ctc, _s) in enumerate(crop_conds or []):
        print(f"  crop[{i}]  cos(tgt)={cos(ctc.cond, tgt_cond.cond):+.3f}   "
              f"cos(src)={cos(csc.cond, src_cond.cond):+.3f}")

    # ---------- (2)+(3) VELOCITY level at a few steps ----------
    hp = editor.hp
    active = editor._active_schedule()
    # pick 3 steps: early (high t, CFG on), mid, late
    idxs = sorted(set([0, len(active) // 2, len(active) - 1]))
    z_edit = x_src.clone()
    torch.manual_seed(0)

    for step_idx in idxs:
        t_curr, t_next = active[step_idx]
        omega_tgt = hp.omega_tgt if t_curr >= hp.cfg_t_min else 0.0
        omega_src = hp.omega_src if t_curr >= hp.cfg_t_min else 0.0
        eps = torch.randn_like(x_src)
        z_src_t, z_tgt_t = editor._coupling(x_src, z_edit, t_curr, eps)
        phi = src_cond.neg_cond

        v_tgt_g = editor._dense_cfg_velocity(z_tgt_t, t_curr, tgt_cond.cond, phi, omega_tgt)
        v_src_g = editor._dense_cfg_velocity(z_src_t, t_curr, src_cond.cond, phi, omega_src)
        v_g = v_tgt_g - v_src_g

        print(f"\n=== step idx={step_idx}  t={t_curr:.3f}  CFG={'on' if omega_tgt>0 else 'off'} ===")
        print(f"  {'crop':>6} {'relL2(vΔc-vΔg)':>16} {'meanVoxCos':>11} {'|vΔc|/|vΔg|':>12}")
        for i, (csc, ctc, _s) in enumerate(crop_conds or []):
            v_tgt_c = editor._dense_cfg_velocity(z_tgt_t, t_curr, ctc.cond, phi, omega_tgt)
            v_src_c = editor._dense_cfg_velocity(z_src_t, t_curr, csc.cond, phi, omega_src)
            v_c = v_tgt_c - v_src_c
            rel = float((v_c - v_g).norm() / (v_g.norm() + 1e-8))
            vcos, act = voxel_cos(v_c, v_g)
            mag = float(v_c.norm() / (v_g.norm() + 1e-8))
            print(f"  {i:>6} {rel:>16.3f} {vcos:>11.3f} {mag:>12.3f}")

        # (3) phi identity: recompute model(z_tgt_t, phi) twice
        flow = wrapper.pipeline.models["sparse_structure_flow_model"]
        tt = torch.tensor([1000.0 * t_curr], device=z_tgt_t.device)
        a = flow(z_tgt_t, tt, phi); b = flow(z_tgt_t, tt, phi)
        print(f"  [phi identity] max|a-b| = {float((a-b).abs().max()):.2e}  "
              f"(0 => phi-forward is exactly cacheable)")
        del v_tgt_g, v_src_g, v_g, a, b
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

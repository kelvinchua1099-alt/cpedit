"""
pipeline.py — end-to-end VS3D editing orchestration.

Config keys consumed
--------------------
seed                             : global random seed
device                           : "cuda" / "cpu"
model.trellis.resolution         : 512 or 1024
solver.num_steps                 : Euler denoising steps
editing.crop.enabled             : detect edit region via diff and crop
editing.crop.padding_scale       : bbox expansion factor (default 1.5)
editing.crop.diff_threshold      : pixel diff threshold 0-255 (default 15)
editing.crop.blur_radius         : GaussianBlur radius (default 3)
editing.crop.guidance_scale      : weight for crop velocity signal (default 0.5)
editing.rasi.enabled             : RASI on/off
editing.pmg.enabled              : PMG on/off
editing.tar.enabled              : TAR on/off
output.root                      : root directory for all outputs
output.save_renders              : save rendered views to disk

Signal design
-------------
Stage-1 velocity update at every active step:
    v_Δ^(s) = [v(z_tgt_t | c_tgt_full) − v(z_src_t | c_src_full)]   ← global edit
             + guidance_scale × [v(z_tgt_t | c_tgt_crop) − v(z_src_t | c_src_crop)]
                                                                        ← local zoom-in
    (second term only when crop.enabled = true and a change was found)

PMG extrapolation is applied to the *combined* v_Δ^(s), so it amplifies
whatever is consistent across noise draws in both signals together.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

from .asset_loader import render_asset
from .guidance import build_guidance
from .preprocess import diff_crop_resize, diff_multi_crop
from .trellis_wrapper import Trellis2Condition, TrellisWrapper
from .vs3d import VS3DEditor

_DENSE_SS_RES = 64   # Stage-1 dense occupancy resolution  (R=64 for 1024³ output)


class EditPipeline:
    """
    Full VS3D editing pipeline.

    Step 1  — diff_crop_resize  (when crop.enabled)
    Step 2  — wrapper.run_source  →  source 3D SLATs
    Step 3  — DINOv3 conditioning (global full-image; +crop when enabled)
    Step 4  — Stage-1: editor.run_stage1  with optional crop guidance signal
    Step 5  — Stage-2/3: editor.edit_sparse_stages  (TAR)
    Step 6  — wrapper.decode_latent  →  MeshWithVoxel
    Step 7  — save GLB / renders / metadata.json
    """

    def __init__(
        self,
        wrapper: TrellisWrapper,
        editor:  VS3DEditor,
        cfg:     DictConfig,
    ) -> None:
        self.wrapper = wrapper
        self.editor  = editor
        self.cfg     = cfg
        self.device  = wrapper.device
        _set_seed(int(cfg.get("seed", 42)))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "EditPipeline":
        wrapper = TrellisWrapper.from_config(cfg)
        editor  = VS3DEditor.from_config(wrapper, cfg)
        return cls(wrapper, editor, cfg)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        img_src:    Image.Image,
        img_tgt:    Image.Image,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full editing pipeline.

        Args
        ----
        img_src    : source (reference) PIL image
        img_tgt    : 2D-edited target PIL image  (same size as img_src)
        output_dir : output folder; falls back to cfg.output.root/<exp>/<timestamp>

        Returns
        -------
        dict with keys  meshes, output_dir, meta
        """
        out_dir = output_dir or self._make_output_dir()
        os.makedirs(out_dir, exist_ok=True)

        meta: Dict[str, Any] = {
            "exp_name":   self.cfg.get("exp_name", "edit"),
            "output_dir": out_dir,
            "seed":       int(self.cfg.get("seed", 42)),
        }

        # ── Step 1: preprocessing (multi-crop) ───────────────────────────
        crops, crop_meta = self._preprocess(img_src, img_tgt)
        meta["crop"] = _jsonable(crop_meta)

        for i, (cs, ct) in enumerate(crops):
            cs.save(os.path.join(out_dir, f"crop_src_{i}.png"))
            ct.save(os.path.join(out_dir, f"crop_tgt_{i}.png"))

        # ── Step 2: generate source 3D asset ─────────────────────────────
        res          = self.cfg.model.trellis.resolution
        pipeline_type = "1024" if res >= 1024 else "512"

        shape_slat_src, tex_slat_src, slat_res = self.wrapper.run_source(
            img_src,
            seed=int(self.cfg.get("seed", 42)),
            preprocess=True,
            pipeline_type=pipeline_type,
        )
        meta["slat_resolution"] = slat_res

        # ── Step 3: conditioning ─────────────────────────────────────────
        #   • src_cond / tgt_cond  — always from FULL images (global signal)
        #   • crop_conds           — from cropped images + scale (extra signal)
        src_cond, tgt_cond, crop_conds = self._build_conditions(
            img_src, img_tgt, crops, crop_meta, res
        )
        meta["crop_guidance_active"] = bool(crop_conds)
        meta["n_crop_signals"] = len(crop_conds) if crop_conds else 0

        # ── Step 4: Stage-1 RASI + PMG ───────────────────────────────────
        x_src      = self._build_stage1_src_latent(shape_slat_src)
        # Free VRAM for RASI's backward pass: only the SS models are needed here.
        self.wrapper.offload_for_stage1()
        z_edit_0   = self.editor.run_stage1(
            x_src, src_cond, tgt_cond, crop_conds=crop_conds
        )
        coords_tgt = self._decode_stage1_to_coords(z_edit_0)
        # Bring the SLAT flow models + decoders back for Stage-2/3.
        self.wrapper.restore_all()
        if coords_tgt.shape[0] == 0:
            # Stage-1 produced an empty structure; fall back to the source
            # lattice so Stage-2/3 still have coordinates to edit on.
            coords_tgt = shape_slat_src.coords.to(coords_tgt.device).int()
            meta["stage1_empty_fallback"] = True
        meta["n_voxels_tgt"] = int(coords_tgt.shape[0])

        # ── Step 5: Stage-2/3 TAR ────────────────────────────────────────
        z_src_shape = self.wrapper.normalize_shape_slat(shape_slat_src)
        z_src_tex   = self.wrapper.normalize_tex_slat(tex_slat_src)

        num_steps = int(self.cfg.solver.get("num_steps", 25))
        guidance  = 1.0 + self.editor.hp.omega_tgt

        z_shape_edit, z_tex_edit = self.editor.edit_sparse_stages(
            z_src_shape_enc=z_src_shape,
            z_src_tex_enc=z_src_tex,
            src_cond=src_cond,
            tgt_cond=tgt_cond,
            coords_tgt=coords_tgt,
            num_steps=num_steps,
            guidance=guidance,
        )

        # ── Step 6: decode to 3D ─────────────────────────────────────────
        shape_slat_edit = self.wrapper.denormalize_shape_slat(z_shape_edit)
        tex_slat_edit   = self.wrapper.denormalize_tex_slat(z_tex_edit)
        meshes = self.wrapper.decode_latent(shape_slat_edit, tex_slat_edit, slat_res)

        # ── Step 7: save ─────────────────────────────────────────────────
        self._save_outputs(meshes, out_dir, meta)

        return {"meshes": meshes, "output_dir": out_dir, "meta": meta}

    # ------------------------------------------------------------------
    # Asset entry point  (3D file → render → run)
    # ------------------------------------------------------------------

    def run_from_asset(
        self,
        asset_path: str,
        img_tgt:    Image.Image,
        output_dir: Optional[str] = None,
        render_size: int  = 512,
        elevation:   float = 15.0,
        azimuth:     float = 30.0,
    ) -> Dict[str, Any]:
        """
        Edit a 3D asset file (.glb / .obj / .ply) given a 2D edited target image.

        Because TRELLIS 2.0 has no mesh-to-SLAT encoder, the asset is rendered
        from a canonical viewpoint to produce img_src, which is then passed
        through the normal image-to-3D pipeline.

        Args
        ----
        asset_path  : path to .glb / .obj / .ply / .stl
        img_tgt     : 2D-edited PIL image showing the desired changes
        output_dir  : output folder (auto if omitted)
        render_size : rendered image resolution (default 512, should match img_tgt)
        elevation   : camera elevation in degrees (default 15°)
        azimuth     : camera azimuth in degrees (default 30° — slight 3/4 view)

        Returns
        -------
        dict with keys  meshes, output_dir, meta  (same as run())
        """
        out_dir = output_dir or self._make_output_dir()
        os.makedirs(out_dir, exist_ok=True)

        # Render the asset → img_src
        img_src = render_asset(
            asset_path,
            size=render_size,
            elevation_deg=elevation,
            azimuth_deg=azimuth,
        )
        # Resize img_tgt to match rendered img_src
        if img_tgt.size != img_src.size:
            img_tgt = img_tgt.resize(img_src.size, Image.BICUBIC)

        # Save the rendered source so the user can inspect the viewpoint
        img_src.save(os.path.join(out_dir, "rendered_src.png"))

        result = self.run(img_src, img_tgt, output_dir=out_dir)
        result["meta"]["asset_path"]    = asset_path
        result["meta"]["render_elev"]   = elevation
        result["meta"]["render_azimuth"] = azimuth
        return result

    # ------------------------------------------------------------------
    # Step 1: preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        img_src: Image.Image,
        img_tgt: Image.Image,
    ) -> Tuple[List[Tuple[Image.Image, Image.Image]], Dict]:
        """
        When crop.enabled: diff_multi_crop → a LIST of (crop_src, crop_tgt) pairs,
        one per significant connected edit component (top-k).  Otherwise: return
        an empty list with found_change=False.
        """
        crop_cfg = self.cfg.editing.get("crop", {})
        if not crop_cfg.get("enabled", False):
            W, H = img_src.size
            return [], {"found_change": False, "n_crops": 0,
                        "bboxes_crop": [], "original_size": (W, H)}

        mode = str(crop_cfg.get("mode", "multi")).lower()

        if mode in ("pure", "single"):
            # Pure RGB-diff + threshold + bbox (single crop), optional percentile.
            cs, ct, m = diff_crop_resize(
                img_src, img_tgt,
                padding_scale   = float(crop_cfg.get("padding_scale",   1.05)),
                diff_threshold  = int(crop_cfg.get("diff_threshold",    70)),
                blur_radius     = int(crop_cfg.get("blur_radius",        3)),
                min_side_frac   = float(crop_cfg.get("min_side_frac",   0.0)),
                bbox_percentile = float(crop_cfg.get("bbox_percentile", 1.0)),
            )
            found = m.get("found_change", False)
            meta = {"found_change": found, "n_crops": 1 if found else 0,
                    "bboxes_crop": [m.get("bbox_crop")] if found else [],
                    "bbox_diff": m.get("bbox_diff"), "original_size": m.get("original_size")}
            return ([(cs, ct)] if found else []), meta

        # Multi-crop: ALL significant connected edit components (no top-k cap),
        # sorted largest-first so crop[0] is the main edit region.  Each becomes
        # a velocity signal, aggregated SplitFlow-style in Stage-1.
        return diff_multi_crop(
            img_src,
            img_tgt,
            padding_scale    = float(crop_cfg.get("padding_scale",   1.2)),
            diff_threshold   = int(crop_cfg.get("diff_threshold",    35)),
            blur_radius      = int(crop_cfg.get("blur_radius",        3)),
            top_k            = int(crop_cfg.get("top_k",              0)),   # 0 = unlimited
            min_component    = int(crop_cfg.get("min_component",    150)),
            morph_open_iters = int(crop_cfg.get("morph_open_iters",   2)),
            min_side_frac    = float(crop_cfg.get("min_side_frac",  0.25)),
        )

    # ------------------------------------------------------------------
    # Step 3: conditioning
    # ------------------------------------------------------------------

    def _build_conditions(
        self,
        img_src:    Image.Image,
        img_tgt:    Image.Image,
        crops:      List[Tuple[Image.Image, Image.Image]],
        crop_meta:  Dict,
        resolution: int,
    ) -> Tuple[
        Trellis2Condition,
        Trellis2Condition,
        Optional[List[Tuple[Trellis2Condition, Trellis2Condition, float]]],
    ]:
        """
        Returns (src_cond, tgt_cond, crop_conds).

        src_cond / tgt_cond  — DINOv3 encodings of the FULL images (global signal).
        crop_conds           — LIST of (crop_src_cond, crop_tgt_cond, scale), one
                               per multi-crop component, when crop is enabled AND
                               changes were found; else None.  Each provides a
                               localized velocity-difference signal (mean-aggregated
                               in Stage-1).
        """
        src_cond = self.wrapper.get_cond([img_src], resolution=resolution)
        tgt_cond = self.wrapper.get_cond([img_tgt], resolution=resolution)

        # TwoViewGuidance decides whether the local (crop) views participate and
        # with what weight; IdentityGuidance (no_two_view ablation) disables it.
        guidance = build_guidance(self.cfg.editing)
        if not guidance.enabled or not crops:
            return src_cond, tgt_cond, None

        scale = guidance.scale
        crop_conds = []
        for (cs, ct) in crops:
            csc = self.wrapper.get_cond([cs], resolution=resolution)
            ctc = self.wrapper.get_cond([ct], resolution=resolution)
            crop_conds.append((csc, ctc, scale))
        return src_cond, tgt_cond, crop_conds

    # ------------------------------------------------------------------
    # Step 4 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_grid_res(shape_slat_src: Any) -> int:
        """Detect the source SLAT lattice resolution from its coords."""
        m = int(shape_slat_src.coords[:, 1:].max().item()) + 1
        for r in (16, 32, 64, 128, 256):
            if m <= r:
                return r
        return 256

    @torch.no_grad()
    def _build_stage1_src_latent(self, shape_slat_src: Any) -> torch.Tensor:
        """
        Build x_src for Stage-1 FlowEdit *in the SS-VAE latent space*.

        TRELLIS.2's sparse_structure_flow_model works on the latent
        z_s ∈ [B, 8, 16, 16, 16] — NOT on raw occupancy.  We scatter the source
        SLAT coords onto the SS VAE's native 64³ occupancy grid (rescaled from
        the runtime-inferred source lattice) and encode:  occupancy → z_s.
        """
        R      = self.wrapper.ss_occ_res            # 64
        device = self.device
        coords = shape_slat_src.coords              # [N, 4]  (batch, x, y, z)
        B      = int(coords[:, 0].max().item()) + 1

        self._src_grid = self._infer_grid_res(shape_slat_src)
        scale = R / float(self._src_grid)

        occ = torch.zeros(B, 1, R, R, R, device=device, dtype=torch.float32)
        b = coords[:, 0].long()
        x = (coords[:, 1].float() * scale).long().clamp(0, R - 1)
        y = (coords[:, 2].float() * scale).long().clamp(0, R - 1)
        z = (coords[:, 3].float() * scale).long().clamp(0, R - 1)
        occ[b, 0, x, y, z] = 1.0

        return self.wrapper.ss_encode(occ)          # [B, 8, 16, 16, 16] float32

    @torch.no_grad()
    def _decode_stage1_to_coords(self, z_edit_0: torch.Tensor) -> torch.Tensor:
        """
        Decode Stage-1 latent at t=0 → C_tgt coords [N, 4] via the SS decoder.

        The decoder yields 64³ occupancy logits; we threshold and max-pool back
        onto the source lattice (``self._src_grid``) so C_tgt lives on the same
        grid as the source SLAT — matching TAR intersection and the SLAT decoder.
        """
        occ = self.wrapper.ss_decode(z_edit_0) > 0          # [B,1,64,64,64] bool
        tgt = getattr(self, "_src_grid", _DENSE_SS_RES)
        if tgt != occ.shape[-1]:
            ratio = occ.shape[-1] // tgt
            occ = F.max_pool3d(occ.float(), ratio, ratio) > 0.5
        coords = torch.argwhere(occ)[:, [0, 2, 3, 4]].int()  # drop channel dim
        return coords.to(z_edit_0.device)

    # ------------------------------------------------------------------
    # Step 7: save outputs
    # ------------------------------------------------------------------

    def _save_outputs(
        self,
        meshes:  List[Any],
        out_dir: str,
        meta:    Dict[str, Any],
    ) -> None:
        save_renders = self.cfg.output.get("save_renders", False)

        for i, mesh in enumerate(meshes):
            glb_path = os.path.join(out_dir, f"result_{i:02d}.glb")
            self._export_glb(mesh, glb_path)
            meta.setdefault("glb_paths", []).append(glb_path)

            if save_renders:
                render_dir = os.path.join(out_dir, f"renders_{i:02d}")
                os.makedirs(render_dir, exist_ok=True)
                for j, img in enumerate(self._render_views(mesh)):
                    img.save(os.path.join(render_dir, f"view_{j:03d}.png"))
                meta.setdefault("render_dirs", []).append(render_dir)

        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(_jsonable(meta), f, indent=2)

    def _export_glb(self, mesh: Any, glb_path: str) -> None:
        """
        Extract a textured GLB from a MeshWithVoxel.

        MeshWithVoxel has no .export()/.render(); GLB extraction goes through
        o_voxel.postprocess.to_glb (remesh + decimate + UV + PBR bake), exactly
        as in TRELLIS.2/example.py.  Quality knobs are read from cfg.output.glb.
        """
        import o_voxel

        gcfg = self.cfg.output.get("glb", {}) if hasattr(self.cfg.output, "get") else {}
        decimation = int(gcfg.get("decimation_target", 300000))
        tex_size   = int(gcfg.get("texture_size", 2048))

        glb = o_voxel.postprocess.to_glb(
            vertices          = mesh.vertices,
            faces             = mesh.faces,
            attr_volume       = mesh.attrs,
            coords            = mesh.coords,
            attr_layout       = mesh.layout,
            voxel_size        = mesh.voxel_size,
            aabb              = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target = decimation,
            texture_size      = tex_size,
            remesh            = True,
            remesh_band       = 1,
            remesh_project    = 0,
            verbose           = False,
        )
        glb.export(glb_path, extension_webp=True)

    def _render_views(self, mesh: Any) -> List[Image.Image]:
        num_views  = int(self.cfg.render.get("num_views", 8))
        image_size = int(self.cfg.render.get("image_size", 512))
        try:
            return mesh.render(num_views=num_views, image_size=image_size)
        except Exception:
            return []

    def _make_output_dir(self) -> str:
        root      = self.cfg.output.get("root", "./outputs")
        exp_name  = self.cfg.get("exp_name", "edit")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(root, exp_name, timestamp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj

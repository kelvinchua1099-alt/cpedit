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
from omegaconf import DictConfig
from PIL import Image

from .asset_loader import render_asset
from .preprocess import diff_crop_resize
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

        # ── Step 1: preprocessing ────────────────────────────────────────
        crop_src, crop_tgt, crop_meta = self._preprocess(img_src, img_tgt)
        meta["crop"] = _jsonable(crop_meta)

        if crop_meta.get("found_change"):
            crop_src.save(os.path.join(out_dir, "crop_src.png"))
            crop_tgt.save(os.path.join(out_dir, "crop_tgt.png"))

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
            img_src, img_tgt, crop_src, crop_tgt, crop_meta, res
        )
        meta["crop_guidance_active"] = crop_conds is not None

        # ── Step 4: Stage-1 RASI + PMG ───────────────────────────────────
        x_src      = self._build_stage1_src_latent(shape_slat_src)
        z_edit_0   = self.editor.run_stage1(
            x_src, src_cond, tgt_cond, crop_conds=crop_conds
        )
        coords_tgt = self._decode_stage1_to_coords(z_edit_0)
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
    ) -> Tuple[Image.Image, Image.Image, Dict]:
        """
        When crop.enabled: diff_crop_resize → zoomed crops + meta.
        Otherwise: return original images with found_change=False.
        """
        crop_cfg = self.cfg.editing.get("crop", {})
        if not crop_cfg.get("enabled", False):
            W, H = img_src.size
            return img_src, img_tgt, {
                "found_change":  False,
                "bbox_crop":     (0, 0, W, H),
                "original_size": (W, H),
            }

        return diff_crop_resize(
            img_src,
            img_tgt,
            padding_scale  = float(crop_cfg.get("padding_scale",  1.5)),
            diff_threshold = int(crop_cfg.get("diff_threshold",   15)),
            blur_radius    = int(crop_cfg.get("blur_radius",       3)),
        )

    # ------------------------------------------------------------------
    # Step 3: conditioning
    # ------------------------------------------------------------------

    def _build_conditions(
        self,
        img_src:    Image.Image,
        img_tgt:    Image.Image,
        crop_src:   Image.Image,
        crop_tgt:   Image.Image,
        crop_meta:  Dict,
        resolution: int,
    ) -> Tuple[
        Trellis2Condition,
        Trellis2Condition,
        Optional[Tuple[Trellis2Condition, Trellis2Condition, float]],
    ]:
        """
        Returns (src_cond, tgt_cond, crop_conds).

        src_cond / tgt_cond  — DINOv3 encodings of the FULL images.
                               Always computed; used as the global edit signal.
        crop_conds           — (crop_src_cond, crop_tgt_cond, scale) when crop is
                               enabled AND a real change was found; else None.
                               Provides the localized velocity difference signal.
        """
        src_cond = self.wrapper.get_cond([img_src], resolution=resolution)
        tgt_cond = self.wrapper.get_cond([img_tgt], resolution=resolution)

        crop_cfg  = self.cfg.editing.get("crop", {})
        use_crop  = crop_cfg.get("enabled", False) and crop_meta.get("found_change", False)

        if not use_crop:
            return src_cond, tgt_cond, None

        crop_scale     = float(crop_cfg.get("guidance_scale", 0.5))
        crop_src_cond  = self.wrapper.get_cond([crop_src], resolution=resolution)
        crop_tgt_cond  = self.wrapper.get_cond([crop_tgt], resolution=resolution)
        return src_cond, tgt_cond, (crop_src_cond, crop_tgt_cond, crop_scale)

    # ------------------------------------------------------------------
    # Step 4 helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_stage1_src_latent(self, shape_slat_src: Any) -> torch.Tensor:
        """
        Build x_src for Stage-1 FlowEdit.

        TRELLIS 2.0's sparse_structure_flow_model operates directly on binary
        occupancy tensors [B, 1, R, R, R] — there is no separate VAE encode step.
        x_src is the "clean data" at t=0: 1.0 for occupied voxels, 0.0 otherwise.
        """
        R      = _DENSE_SS_RES
        device = self.device
        coords = shape_slat_src.coords          # [N, 4]  (batch, x, y, z)
        B      = int(coords[:, 0].max().item()) + 1

        occ = torch.zeros(B, 1, R, R, R, device=device, dtype=torch.float32)
        b   = coords[:, 0].long()
        x   = coords[:, 1].long().clamp(0, R - 1)
        y   = coords[:, 2].long().clamp(0, R - 1)
        z   = coords[:, 3].long().clamp(0, R - 1)
        occ[b, 0, x, y, z] = 1.0
        return occ   # [B, 1, R, R, R]

    @torch.no_grad()
    def _decode_stage1_to_coords(
        self,
        z_edit_0:  torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Decode Stage-1 flow output at t=0 → C_tgt coordinate set [N, 4].

        Tries sparse_structure_decoder first (TRELLIS 2.0 official model key);
        falls back to direct channel-0 thresholding.
        """
        decoder = self.wrapper.pipeline.models.get("sparse_structure_decoder")
        if decoder is not None:
            try:
                out = decoder(z_edit_0)
                if hasattr(out, "coords"):
                    return out.coords.to(z_edit_0.device)
            except Exception:
                pass
        # Fallback: threshold logit channel 0
        active = (z_edit_0[:, 0] > threshold).nonzero()   # [N, 4]
        return active.to(z_edit_0.device)

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
            mesh.export(glb_path)
            meta.setdefault("glb_paths", []).append(glb_path)

            if save_renders:
                render_dir = os.path.join(out_dir, f"renders_{i:02d}")
                os.makedirs(render_dir, exist_ok=True)
                for j, img in enumerate(self._render_views(mesh)):
                    img.save(os.path.join(render_dir, f"view_{j:03d}.png"))
                meta.setdefault("render_dirs", []).append(render_dir)

        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(_jsonable(meta), f, indent=2)

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

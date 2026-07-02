"""
guidance.py — SplitFlow-style two-view guidance for VS3D editing.

The "two views" here are the two conditioning signals that drive the Stage-1
FlowEdit velocity difference:

    • global view  — DINOv3 features of the FULL source / target images
    • local  view  — DINOv3 features of the diff-CROP (zoomed edit region)

At every denoising step the pipeline forms a global velocity difference
``v_Δ_global = v_tgt(full) − v_src(full)`` and (optionally) a local one
``v_Δ_crop = v_tgt(crop) − v_src(crop)``.  TwoViewGuidance fuses them:

    v_Δ = v_Δ_global + scale · v_Δ_crop            (SplitFlow-style blend)

Design notes
------------
* This module owns the *decision* of whether the local (crop) view participates
  and with what weight.  ``EditPipeline._build_conditions`` consults it to decide
  whether to build ``crop_conds`` and which ``scale`` to hand to the Stage-1 loop
  (``VS3DEditor._stage1_pmg_phase`` performs the actual per-step fusion using the
  identical formula in ``combine`` below).
* ``IdentityGuidance`` is the ablation control (``no_two_view``): it disables the
  local view entirely, so only the global full-image signal edits the asset.
  Crop *preprocessing* still runs (debug crops are saved), which is exactly what
  distinguishes ``no_two_view`` (crop images saved, no local guidance) from
  ``no_crop`` (no cropping at all).
"""
from __future__ import annotations


class TwoViewGuidance:
    """Fuse the global and local velocity-difference views (SplitFlow style)."""

    def __init__(self, scale: float = 0.5) -> None:
        self.scale = float(scale)

    @property
    def enabled(self) -> bool:
        return True

    def combine(self, v_global, v_crop=None):
        """
        v_Δ = v_global + scale · v_crop

        Reference implementation of the fusion that
        ``VS3DEditor._stage1_pmg_phase`` performs inline.  Kept here so the
        two-view math lives in one documented place and can be unit-tested.
        """
        if v_crop is None or self.scale == 0.0:
            return v_global
        return v_global + self.scale * v_crop

    def __repr__(self) -> str:
        return f"TwoViewGuidance(scale={self.scale})"


class IdentityGuidance:
    """Ablation control: no local view — return the global signal unchanged."""

    scale = 0.0

    @property
    def enabled(self) -> bool:
        return False

    def combine(self, v_global, v_crop=None):
        return v_global

    def __repr__(self) -> str:
        return "IdentityGuidance()"


def build_guidance(editing_cfg) -> "TwoViewGuidance | IdentityGuidance":
    """
    Build the guidance object from the ``editing`` config block.

    Enabled when BOTH hold (defaults keep legacy full/no_crop configs working):
        editing.crop.enabled      != false   (a crop signal is available)
        editing.two_view.enabled  != false   (two-view fusion not ablated)

    The scale is taken from ``editing.crop.guidance_scale`` (default 0.5).
    """
    def _get(block, key, default):
        if block is None:
            return default
        try:
            if hasattr(block, "get"):
                return block.get(key, default)
            return getattr(block, key, default)
        except Exception:
            return default

    crop_cfg = _get(editing_cfg, "crop", None)
    two_cfg = _get(editing_cfg, "two_view", None)

    crop_on = bool(_get(crop_cfg, "enabled", False))
    two_on = bool(_get(two_cfg, "enabled", True))     # default on if unspecified
    scale = float(_get(crop_cfg, "guidance_scale", 0.5))

    if crop_on and two_on and scale != 0.0:
        return TwoViewGuidance(scale=scale)
    return IdentityGuidance()

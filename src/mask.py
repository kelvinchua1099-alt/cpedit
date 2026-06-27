"""
mask.py — automatic edit-region mask in the Stage-1 SS latent space.

The B1 ("latent-delta") strategy: run one *global* (un-masked) Stage-1 FlowEdit
pass, then read off where the model actually wants to move the occupancy latent.
The per-voxel change magnitude of z_edit − x_src is a camera-free localization
prior for "where the edit lives", which we threshold + dilate + feather into a
soft mask M ∈ [0,1] on the 16³ SS latent grid.

That mask is then used to (C1) gate the Stage-1 velocity so drift outside the
edit region is suppressed (identity preserved) while the window inside can be
opened to earlier/higher-noise steps, and (C2) switch OFF TAR's source-residual
re-injection inside the region so the edit is not pulled back to the source.

Honest limitation: a low-noise global probe localizes appearance / where the
model already nudges; it cannot localize purely-additive structure that only
forms at high noise (chicken-and-egg). For that, an image-grounded mask
(project the 2D diff mask into the grid) is the follow-up — see README notes.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _gaussian_feather(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """Cheap separable feather via repeated 3×3×3 average pooling (≈ Gaussian)."""
    if sigma <= 0:
        return m
    n = max(1, int(round(sigma)))
    for _ in range(n):
        m = F.avg_pool3d(m, kernel_size=3, stride=1, padding=1)
    return m


@torch.no_grad()
def build_latent_delta_mask(
    z_edit:    torch.Tensor,   # [B, C, R, R, R] global-pass edited SS latent
    x_src:     torch.Tensor,   # [B, C, R, R, R] source SS latent
    threshold: float = 0.25,
    dilate:    int   = 1,
    feather:   float = 1.0,
    min_frac:  float = 0.02,
) -> torch.Tensor:
    """
    Returns a soft mask M ∈ [0,1] of shape [B, 1, R, R, R].

    Per-sample min-max normalised change magnitude, thresholded, dilated and
    feathered. If the thresholded mask is empty (or smaller than min_frac of the
    grid) it falls back to all-ones so the run degrades gracefully to global
    editing rather than producing a no-op.
    """
    B = z_edit.shape[0]
    d = (z_edit - x_src).abs().mean(dim=1, keepdim=True)        # [B,1,R,R,R]

    flat = d.view(B, -1)
    lo = flat.amin(dim=1).view(B, 1, 1, 1, 1)
    hi = flat.amax(dim=1).view(B, 1, 1, 1, 1)
    m_norm = (d - lo) / (hi - lo + 1e-8)

    M = (m_norm > threshold).float()

    if dilate > 0:
        k = 2 * dilate + 1
        M = F.max_pool3d(M, kernel_size=k, stride=1, padding=dilate)

    # Graceful fallback: degenerate (empty) mask → edit globally.
    n_vox = M[0, 0].numel()
    for b in range(B):
        if M[b].sum() < min_frac * n_vox:
            M[b] = 1.0

    M = _gaussian_feather(M, feather).clamp(0.0, 1.0)
    return M


@torch.no_grad()
def sample_mask_at_coords(
    M:        torch.Tensor,    # [B, 1, R, R, R] latent-grid mask
    coords:   torch.Tensor,    # [N, 4] (b, x, y, z) on a `grid_res` lattice
    grid_res: int,
) -> torch.Tensor:
    """
    Nearest-neighbour sample the latent-grid mask at sparse voxel coordinates
    that live on a (usually finer) `grid_res` lattice. Returns [N] float in [0,1].
    """
    R = M.shape[-1]
    scale = R / float(grid_res)
    b  = coords[:, 0].long()
    ix = (coords[:, 1].float() * scale).long().clamp(0, R - 1)
    iy = (coords[:, 2].float() * scale).long().clamp(0, R - 1)
    iz = (coords[:, 3].float() * scale).long().clamp(0, R - 1)
    return M[b, 0, ix, iy, iz]

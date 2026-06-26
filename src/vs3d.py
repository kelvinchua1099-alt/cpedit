"""
VS3D: Velocity-Space 3D Asset Editing.
Implements Algorithm 1 from arXiv:2605.07385.

Stage 1  — RASI (Reconstruction-Anchored Source Injection) +
           PMG  (Partial-Mean Guidance) on the dense occupancy latent z_ss.
Stage 2/3 — TAR (Twin-Agreement Residual injection) on sparse geometry
             and material SLATs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from omegaconf import DictConfig

try:
    from trellis2.modules.sparse import SparseTensor
except ImportError:
    SparseTensor = None  # type: ignore[assignment,misc]

from .trellis_wrapper import Trellis2Condition, TrellisWrapper


# ---------------------------------------------------------------------------
# Hyper-parameter bundle  (mirrors Appendix A.1 of the paper)
# ---------------------------------------------------------------------------

@dataclass
class VS3DHyperparams:
    # Stage-1 schedule
    num_steps: int   = 25
    n_max:     int   = 12    # (legacy, unused) last active step index
    n_min:     int   = 0     # (legacy, unused) first active step index
    # FlowEdit edits the LATE (low-noise) steps only: skip the first st_step
    # high-noise steps and edit the rest (matches Nano3D inference/sampling.py:
    # `if num_st < st_step: continue`). Editing the early high-noise steps
    # instead rewrites the global structure and scrambles the object.
    st_step:   int   = 12

    # CFG weights. Nano3D uses cfg_strength directly; cpedit's CFG is
    # (1+ω)·v_cond − ω·v_phi, so cfg_strength = 1+ω. Nano3D defaults
    # tar_cfg=5.5, src_cfg=1.5  →  ω_tgt=4.5, ω_src=0.5. CFG is applied over the
    # full t-range (Nano3D cfg_interval=(0,1)), so cfg_t_min=0.
    omega_src:  float = 0.5
    omega_tgt:  float = 4.5
    cfg_t_min:  float = 0.0  # CFG on for all t ∈ [cfg_t_min, 1.0]

    # Monte-Carlo noise budget
    S: int = 5               # total noise samples per step

    # PMG  (Sec. 3.3)
    pmg_w: float = 1.2       # extrapolation weight w
    pmg_L: int   = 2         # partial-mean sample count L  (L < S)

    # RASI inner optimisation  (Sec. 3.2)
    rasi_K:      int   = 3
    rasi_lr:     float = 1e-5
    rasi_tau_es: float = 1e-5   # early-stop loss threshold

    # TAR  (Sec. 3.4)
    tar_lambda: float = 0.5
    tar_tau:    float = 10.0    # norm-clip threshold τ
    tar_theta:  float = 0.7     # agreement threshold ϑ
    tar_alpha:  float = 0.05    # quantile lower bound α
    tar_beta:   float = 0.95    # quantile upper bound β


# ---------------------------------------------------------------------------
# Main editor class
# ---------------------------------------------------------------------------

class VS3DEditor:
    """
    Implements the full VS3D pipeline on a frozen TRELLIS 2.0 backbone.

    Usage (from pipeline.py)
    ------------------------
    editor = VS3DEditor.from_config(wrapper, cfg)

    # Stage 1 (RASI + PMG → edited occupancy latent)
    z_edit_0 = editor.run_stage1(x_src_feats, src_cond, tgt_cond)
    # decode z_edit_0 via pipeline's sparse_structure_vae to get C_tgt

    # Stage 2/3 (TAR → geometry / material SLATs)
    z_shape = editor.run_tar(z_src_shape_enc, src_cond, tgt_cond, model='shape', ...)
    z_tex   = editor.run_tar(z_src_tex_enc,   src_cond, tgt_cond, model='tex',   ...)
    """

    def __init__(
        self,
        wrapper:      TrellisWrapper,
        hp:           VS3DHyperparams,
        enabled_rasi: bool = True,
        enabled_pmg:  bool = True,
        enabled_tar:  bool = True,
    ) -> None:
        self.wrapper      = wrapper
        self.hp           = hp
        self.enabled_rasi = enabled_rasi
        self.enabled_pmg  = enabled_pmg
        self.enabled_tar  = enabled_tar
        self.device       = wrapper.device

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, wrapper: TrellisWrapper, cfg: DictConfig) -> "VS3DEditor":
        ec = cfg.editing  # shorthand

        hp = VS3DHyperparams(
            num_steps   = cfg.solver.get("num_steps", 25),
            n_max       = ec.get("n_max", 12),
            n_min       = ec.get("n_min", 0),
            st_step     = ec.get("st_step", 12),
            omega_src   = ec.get("omega_src", 0.5),
            omega_tgt   = ec.get("omega_tgt", 4.5),
            cfg_t_min   = ec.get("cfg_t_min", 0.0),
            S           = ec.get("noise_samples", 5),
            pmg_w       = ec.pmg.get("w", 1.2)         if hasattr(ec, "pmg")  else 1.2,
            pmg_L       = ec.pmg.get("L", 2)           if hasattr(ec, "pmg")  else 2,
            rasi_K      = ec.rasi.get("K", 3)          if hasattr(ec, "rasi") else 3,
            rasi_lr     = ec.rasi.get("lr", 1e-5)      if hasattr(ec, "rasi") else 1e-5,
            rasi_tau_es = ec.rasi.get("tau_es", 1e-5)  if hasattr(ec, "rasi") else 1e-5,
            tar_lambda  = ec.tar.get("lam", 0.5)       if hasattr(ec, "tar")  else 0.5,
            tar_tau     = ec.tar.get("tau", 10.0)      if hasattr(ec, "tar")  else 10.0,
            tar_theta   = ec.tar.get("theta", 0.7)     if hasattr(ec, "tar")  else 0.7,
        )

        return cls(
            wrapper,
            hp,
            enabled_rasi = ec.rasi.enabled if hasattr(ec, "rasi") else True,
            enabled_pmg  = ec.pmg.enabled  if hasattr(ec, "pmg")  else True,
            enabled_tar  = ec.tar.enabled  if hasattr(ec, "tar")  else True,
        )

    # ------------------------------------------------------------------
    # Timestep schedule helpers
    # ------------------------------------------------------------------

    def _full_schedule(self) -> List[Tuple[float, float]]:
        """Uniform (t_curr, t_next) pairs from t=1 down to t=0."""
        T = self.hp.num_steps
        return [(1.0 - k / T, 1.0 - (k + 1) / T) for k in range(T)]

    def _active_schedule(self) -> List[Tuple[float, float]]:
        """
        Active (t_curr, t_next) pairs — the LATE, low-noise steps only.

        Nano3D skips the first `st_step` high-noise steps and edits the rest
        (inference/sampling.py: `if num_st < st_step: continue`). st_step is
        clamped below num_steps so at least one step is always edited.
        """
        sched = self._full_schedule()
        st = max(0, min(self.hp.st_step, len(sched) - 1))
        return sched[st:]

    # ------------------------------------------------------------------
    # FlowEdit coupling  (eq. 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _coupling(
        x_src: torch.Tensor,   # clean source latent (any shape)
        z_edit: torch.Tensor,  # current edit state (same shape)
        t: float,
        eps: torch.Tensor,     # noise (same shape)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_src_t  = (1-t)*x_src + t*eps
        z_tgt_t  = z_edit + (z_src_t - x_src)   [eq. 3]
        """
        z_src_t = (1.0 - t) * x_src + t * eps
        z_tgt_t = z_edit + z_src_t - x_src
        return z_src_t, z_tgt_t

    # ------------------------------------------------------------------
    # Stage-1 dense CFG velocity with custom unconditional embedding ϕ
    # ------------------------------------------------------------------

    def _dense_cfg_velocity(
        self,
        x_t:   torch.Tensor,   # [B, C, R, R, R]
        t:     float,
        cond:  torch.Tensor,   # [B, D] positive condition
        phi:   torch.Tensor,   # [B, D] unconditional / negative embedding ϕ
        omega: float,          # CFG weight in paper notation
    ) -> torch.Tensor:
        """
        ṽ = (1+ω)*v_θ(x_t, t, cond) − ω*v_θ(x_t, t, ϕ)   [eq. 4]

        Accesses pipeline.models["sparse_structure_flow_model"] directly
        because the TrellisWrapper exposes only the sparse SLAT models.
        """
        flow_model = self.wrapper.pipeline.models["sparse_structure_flow_model"]
        B = x_t.shape[0]
        t_model = torch.tensor([1000.0 * t] * B, dtype=torch.float32, device=x_t.device)

        v_cond = flow_model(x_t, t_model, cond)
        if omega == 0.0:
            return v_cond
        v_phi = flow_model(x_t, t_model, phi)
        return (1.0 + omega) * v_cond - omega * v_phi

    # ------------------------------------------------------------------
    # RASI: per-step ϕ optimisation  (eq. 7, Algorithm 1 lines 4-9)
    # ------------------------------------------------------------------

    def _rasi_step(
        self,
        z_edit:   torch.Tensor,  # current edit state [B, C, R, R, R]
        x_src:    torch.Tensor,  # clean source latent
        t_curr:   float,
        t_next:   float,
        src_cond: torch.Tensor,  # c_src  [B, D]
        phi_init: torch.Tensor,  # ϕ_0 as warm-start [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Minimise L_rec (eq. 7) w.r.t. ϕ and advance z_edit one Euler step
        on the source-reconstruction ODE (both branches use c_src).

        Returns (ϕ_t, z_edit_next).
        """
        hp = self.hp
        dt = t_next - t_curr  # negative (denoising direction)

        phi = phi_init.detach().clone().requires_grad_(True)
        opt = optim.Adam([phi], lr=hp.rasi_lr)

        for _ in range(hp.rasi_K):
            opt.zero_grad()

            eps = torch.randn_like(x_src)
            z_src_t, z_tgt_t = self._coupling(x_src, z_edit, t_curr, eps)

            # Probe: both branches use c_src, but keep their real CFG weights
            v_tgt = self._dense_cfg_velocity(z_tgt_t, t_curr, src_cond, phi, hp.omega_tgt)
            v_src = self._dense_cfg_velocity(z_src_t, t_curr, src_cond, phi, hp.omega_src)

            z_reconstructed = z_edit + dt * (v_tgt - v_src)
            loss = (z_reconstructed - x_src).pow(2).mean()
            loss.backward()
            opt.step()

            if loss.item() < hp.rasi_tau_es:
                break

        phi_cached = phi.detach()

        # Advance z_edit one step along the source-reconstruction ODE
        with torch.no_grad():
            eps = torch.randn_like(x_src)
            z_src_t, z_tgt_t = self._coupling(x_src, z_edit, t_curr, eps)
            v_tgt = self._dense_cfg_velocity(z_tgt_t, t_curr, src_cond, phi_cached, hp.omega_tgt)
            v_src = self._dense_cfg_velocity(z_src_t, t_curr, src_cond, phi_cached, hp.omega_src)
            z_edit_next = z_edit + dt * (v_tgt - v_src)

        return phi_cached, z_edit_next

    # ------------------------------------------------------------------
    # Stage-1 Phase 1: RASI calibration  (Algorithm 1 lines 2-10)
    # ------------------------------------------------------------------

    def _stage1_rasi_phase(
        self,
        x_src:    torch.Tensor,
        src_cond: Trellis2Condition,
    ) -> Dict[int, torch.Tensor]:
        """
        Optimise and cache ϕ_t for each active timestep.
        Returns phi_cache: step_index → ϕ_t tensor.
        """
        active   = self._active_schedule()
        phi_prev = src_cond.neg_cond.clone()
        z_edit   = x_src.clone()
        cache: Dict[int, torch.Tensor] = {}

        for idx, (t_curr, t_next) in enumerate(active):
            with torch.enable_grad():
                phi_t, z_edit = self._rasi_step(
                    z_edit, x_src, t_curr, t_next,
                    src_cond.cond, phi_prev,
                )
            cache[idx] = phi_t
            phi_prev   = phi_t   # warm-start next step

        return cache

    # ------------------------------------------------------------------
    # Stage-1 Phase 2: PMG FlowEdit  (Algorithm 1 lines 11-20)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _stage1_pmg_phase(
        self,
        x_src:      torch.Tensor,
        src_cond:   Trellis2Condition,
        tgt_cond:   Trellis2Condition,
        phi_cache:  Dict[int, torch.Tensor],
        crop_conds: Optional[Tuple[Trellis2Condition, Trellis2Condition, float]] = None,
    ) -> torch.Tensor:
        """
        FlowEdit with RASI-injected ϕ_t, PMG amplification, and optional
        crop guidance.

        crop_conds = (crop_src_cond, crop_tgt_cond, crop_scale)
            When provided, each noise sample also computes a velocity difference
            using the crop-resized conditions and adds it as an extra edit signal:
                v_Δ^(s) = [v_tgt(full) - v_src(full)]
                         + crop_scale * [v_tgt(crop) - v_src(crop)]
            The same ε and z_src_t / z_tgt_t are reused for both evaluations,
            keeping the signals correlated so PMG amplifies their common direction.

        Returns z_edit at t=0.
        """
        hp     = self.hp
        active = self._active_schedule()
        z_edit = x_src.clone()

        crop_src_cond: Optional[Trellis2Condition] = None
        crop_tgt_cond: Optional[Trellis2Condition] = None
        crop_scale = 0.0
        if crop_conds is not None:
            crop_src_cond, crop_tgt_cond, crop_scale = crop_conds

        for idx, (t_curr, t_next) in enumerate(active):
            dt = t_next - t_curr  # negative

            phi = phi_cache.get(idx, src_cond.neg_cond)

            # CFG is gated to t ≥ cfg_t_min
            omega_tgt_eff = hp.omega_tgt if t_curr >= hp.cfg_t_min else 0.0
            omega_src_eff = hp.omega_src if t_curr >= hp.cfg_t_min else 0.0

            # S independent noise draws — same ε is reused for global + crop
            v_delta_stack: List[torch.Tensor] = []
            for _ in range(hp.S):
                eps = torch.randn_like(x_src)
                z_src_t, z_tgt_t = self._coupling(x_src, z_edit, t_curr, eps)

                # Global velocity difference  (full-image conditions)
                v_tgt_g = self._dense_cfg_velocity(
                    z_tgt_t, t_curr, tgt_cond.cond, phi, omega_tgt_eff
                )
                v_src_g = self._dense_cfg_velocity(
                    z_src_t, t_curr, src_cond.cond, phi, omega_src_eff
                )
                v_delta_s = v_tgt_g - v_src_g

                # Crop velocity difference  (additional local signal)
                if crop_conds is not None:
                    v_tgt_c = self._dense_cfg_velocity(
                        z_tgt_t, t_curr, crop_tgt_cond.cond, phi, omega_tgt_eff
                    )
                    v_src_c = self._dense_cfg_velocity(
                        z_src_t, t_curr, crop_src_cond.cond, phi, omega_src_eff
                    )
                    v_delta_s = v_delta_s + crop_scale * (v_tgt_c - v_src_c)

                v_delta_stack.append(v_delta_s)

            # µ̂_S (full mean) and µ̂_L (partial mean) — PMG on combined signal
            all_v = torch.stack(v_delta_stack, dim=0)   # [S, B, C, R, R, R]
            mu_S  = all_v.mean(dim=0)

            if self.enabled_pmg and hp.S > hp.pmg_L:
                mu_L = all_v[: hp.pmg_L].mean(dim=0)
                u = mu_S + hp.pmg_w * (mu_S - mu_L)
            else:
                u = mu_S

            z_edit = z_edit + dt * u

        return z_edit

    # ------------------------------------------------------------------
    # Stage-1 public entry point
    # ------------------------------------------------------------------

    def run_stage1(
        self,
        x_src:      torch.Tensor,
        src_cond:   Trellis2Condition,
        tgt_cond:   Trellis2Condition,
        crop_conds: Optional[Tuple[Trellis2Condition, Trellis2Condition, float]] = None,
    ) -> torch.Tensor:
        """
        RASI + PMG on the dense Stage-1 occupancy latent z_ss.

        Args:
            x_src      : clean source latent  [B, C, R, R, R]
            src_cond   : DINOv3 condition for full source image
            tgt_cond   : DINOv3 condition for full target image
            crop_conds : (crop_src_cond, crop_tgt_cond, scale) or None.
                         When set, the crop-region velocity difference is added
                         as an extra guidance signal at each denoising step.
                         RASI is NOT re-run for the crop conditions — it optimises
                         ϕ only against the global (full-image) signal.

        Returns:
            z_edit at t=0  [B, C, R, R, R]
        """
        # Stage-1 hits the sparse-structure flow model many times via
        # _dense_cfg_velocity. Under low-VRAM mode TRELLIS.2 keeps models on CPU
        # until used, so move it to GPU for the whole Stage-1 and restore after
        # (mirrors sample_sparse_structure's low_vram handling).
        ss_model = self.wrapper.pipeline.models["sparse_structure_flow_model"]
        low_vram = bool(getattr(self.wrapper.pipeline, "low_vram", False))
        if low_vram:
            ss_model.to(self.device)

        # RASI back-props through the full 1.3B SS flow model; without gradient
        # checkpointing the retained activations of all transformer blocks blow
        # past 24 GB. Toggle per-block checkpointing on for Stage-1 and restore.
        ckpt_blocks = [b for b in getattr(ss_model, "blocks", [])
                       if hasattr(b, "use_checkpoint")]
        prev_ckpt = [b.use_checkpoint for b in ckpt_blocks]
        if self.enabled_rasi:
            for b in ckpt_blocks:
                b.use_checkpoint = True

        try:
            phi_cache: Dict[int, torch.Tensor] = {}
            if self.enabled_rasi:
                phi_cache = self._stage1_rasi_phase(x_src, src_cond)

            return self._stage1_pmg_phase(
                x_src, src_cond, tgt_cond, phi_cache, crop_conds
            )
        finally:
            for b, p in zip(ckpt_blocks, prev_ckpt):
                b.use_checkpoint = p
            if low_vram:
                ss_model.cpu()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # TAR helpers: coordinate intersection  (vectorised)
    # ------------------------------------------------------------------

    @staticmethod
    def _coord_keys(coords: torch.Tensor) -> torch.Tensor:
        """
        Encode (batch, x, y, z) coords as unique int64 keys.
        Assumes each axis fits in 16 bits (R ≤ 65535).
        """
        return (
            coords[:, 0].long() << 48
            | coords[:, 1].long() << 32
            | coords[:, 2].long() << 16
            | coords[:, 3].long()
        )

    @classmethod
    def _intersect_coords(
        cls,
        coords_a: torch.Tensor,   # [N_a, 4]
        coords_b: torch.Tensor,   # [N_b, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (idx_a, idx_b) index tensors such that
        coords_a[idx_a] == coords_b[idx_b] for every i.
        O(N log N) via sorted searchsorted.
        """
        keys_a = cls._coord_keys(coords_a)   # [N_a]
        keys_b = cls._coord_keys(coords_b)   # [N_b]

        # Sort b and binary-search each key of a in sorted b
        sorted_b, sort_idx_b = keys_b.sort()
        pos = torch.searchsorted(sorted_b, keys_a)
        pos = pos.clamp(0, sorted_b.shape[0] - 1)

        hit   = sorted_b[pos] == keys_a         # [N_a] bool
        idx_a = hit.nonzero(as_tuple=True)[0]   # [|I|]
        idx_b = sort_idx_b[pos[idx_a]]          # [|I|]

        return idx_a, idx_b

    # ------------------------------------------------------------------
    # Sparse SLAT full-pass sampler (used by TAR twin forwards)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_sparse_sampler(
        self,
        init_noise:  Any,
        cond:        Trellis2Condition,
        model:       str,           # 'shape' or 'tex'
        num_steps:   int,
        guidance:    float,
        **model_kwargs,
    ) -> Any:
        """
        Standard Euler denoising from t=1 to t=0 for sparse SLATs.
        This is  S_θ^stage  from eq. (10) of the paper.
        """
        z = init_noise
        for k in range(num_steps):
            t_curr = 1.0 - k / num_steps
            t_next = 1.0 - (k + 1) / num_steps
            z = self.wrapper.euler_step(
                z, t_curr, t_next, cond,
                guidance_strength=guidance,
                model=model,
                **model_kwargs,
            )
        return z

    # ------------------------------------------------------------------
    # TAR: Twin-Agreement Residual Injection  (Sec. 3.4, eq. 10-11)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_tar(
        self,
        z_src_enc:  Any,                 # source SC-VAE encoding, normalised, on C_src
        src_cond:   Trellis2Condition,
        tgt_cond:   Trellis2Condition,
        model:      str,                 # 'shape' or 'tex'
        coords_tgt: torch.Tensor,        # C_tgt voxel coords  [N_tgt, 4]
        num_steps:  int   = 25,
        guidance:   float = 10.0,        # 1 + omega_tgt  for the main forward
        **model_kwargs,
    ) -> Any:
        """
        Twin-Agreement Residual Injection for one sparse SLAT stage.

        Algorithm 1 lines 22-32:
          1. Native target forward  → z_tgt
          2. Source-conditioned twin forward (same init noise) → z_twin_src
          3. Per-token disagreement d_i = ||z_tgt[i] - z_twin_src[i]||_2
          4. p_keep via robust quantile clipping  (eq. 10)
          5. Blend source SC-VAE residuals back onto high-agreement tokens (eq. 11)

        Args:
            z_src_enc  : normalised source encoding on C_src  (from normalize_*_slat)
            src_cond   : source DINOv3 condition
            tgt_cond   : target DINOv3 condition
            model      : 'shape' or 'tex'
            coords_tgt : C_tgt  integer voxel coordinates  [N_tgt, 4]
            num_steps  : sampler steps (should match Stage-1 T)
            guidance   : CFG strength for the target pass  (1 + omega_tgt)

        Returns:
            Blended SparseTensor in normalised latent space, on C_tgt coordinates.
        """
        hp     = self.hp
        device = self.device

        # --- Build initial noise on C_tgt ---
        # Noise lives in the *latent* (output) space of the flow model. For the
        # texture stage the model's in_channels already bundles the concatenated
        # shape latent (in = tex_latent + shape_latent), so sizing noise by
        # in_channels would double-count it once concat_cond is added in forward.
        # Use out_channels (the pure latent dim) for both stages.
        n_tgt = coords_tgt.shape[0]
        in_ch = (self.wrapper.shape_latent_channels if model == "shape"
                 else self.wrapper.tex_latent_channels)
        noise_feats = torch.randn(n_tgt, in_ch, device=device, dtype=torch.float32)
        coords_tgt  = coords_tgt.contiguous()   # flex_gemm sparse conv requires it
        init_noise  = SparseTensor(coords=coords_tgt, feats=noise_feats)

        # --- Native target forward  (eq. 10 left) ---
        z_tgt = self._run_sparse_sampler(
            init_noise, tgt_cond, model, num_steps, guidance, **model_kwargs
        )

        # --- Condition-swapped twin  (eq. 10 right, same init noise) ---
        init_noise_twin = SparseTensor(
            coords=coords_tgt,
            feats=noise_feats.clone(),
        )
        z_twin_src = self._run_sparse_sampler(
            init_noise_twin, src_cond, model, num_steps, guidance, **model_kwargs
        )

        if not self.enabled_tar:
            return z_tgt

        # --- Per-token preserve-confidence  (eq. 10 disagreement → p_flow) ---
        d = (z_tgt.feats - z_twin_src.feats).norm(dim=-1)   # [N_tgt]
        q_lo = torch.quantile(d, hp.tar_alpha)
        q_hi = torch.quantile(d, hp.tar_beta)
        p_keep = (1.0 - (d - q_lo).clamp(min=0.0) / (q_hi - q_lo + 1e-8)).clamp(0.0, 1.0)

        # --- Intersection I = C_tgt ∩ C_src ---
        idx_tgt, idx_src = self._intersect_coords(coords_tgt, z_src_enc.coords)

        if idx_tgt.numel() == 0:
            return z_tgt   # no overlap; preserve target as-is

        # --- Norm-clipped residual  r_i = clip_τ(z_src_enc[i] - z_tgt[i])  ---
        f_tgt_I = z_tgt.feats[idx_tgt]         # [|I|, C]
        f_src_I = z_src_enc.feats[idx_src]     # [|I|, C]
        r       = f_src_I - f_tgt_I            # [|I|, C]
        r_norm  = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        r       = r * (hp.tar_tau / r_norm).clamp(max=1.0)   # norm-clip

        # --- Agreement-gated blend  (eq. 11) ---
        p_I    = p_keep[idx_tgt].unsqueeze(-1)                   # [|I|, 1]
        gate   = (p_keep[idx_tgt] >= hp.tar_theta).float().unsqueeze(-1)  # [|I|, 1]

        z_out_feats = z_tgt.feats.clone()
        z_out_feats[idx_tgt] = f_tgt_I + hp.tar_lambda * p_I * gate * r

        return z_tgt.replace(feats=z_out_feats)

    # ------------------------------------------------------------------
    # Convenience: run TAR on both geometry and material stages
    # ------------------------------------------------------------------

    @torch.no_grad()
    def edit_sparse_stages(
        self,
        z_src_shape_enc: Any,              # normalised geometry SC-VAE encoding
        z_src_tex_enc:   Any,              # normalised texture SC-VAE encoding
        src_cond:        Trellis2Condition,
        tgt_cond:        Trellis2Condition,
        coords_tgt:      torch.Tensor,     # C_tgt  [N_tgt, 4]
        num_steps:       int   = 25,
        guidance:        float = 10.0,
    ) -> Tuple[Any, Any]:
        """
        Run TAR for Stage 2 (geometry) and Stage 3 (material).

        The material sampler is conditioned on the geometry SLAT; we pass it
        as concat_cond following TRELLIS 2.0 conventions.

        Returns:
            (z_shape, z_tex) — both normalised SparseTensors on C_tgt.
        """
        z_shape = self.run_tar(
            z_src_shape_enc, src_cond, tgt_cond,
            model="shape",
            coords_tgt=coords_tgt,
            num_steps=num_steps,
            guidance=guidance,
        )

        # Stage 3: material is geometry-conditioned
        z_tex = self.run_tar(
            z_src_tex_enc, src_cond, tgt_cond,
            model="tex",
            coords_tgt=coords_tgt,
            num_steps=num_steps,
            guidance=guidance,
            concat_cond=z_shape,   # geometry-conditioned material pass
        )

        return z_shape, z_tex

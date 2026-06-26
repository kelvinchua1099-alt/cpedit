"""
TrellisWrapper: wraps Trellis2ImageTo3DPipeline for 3D latent-space editing.

Architecture recap (TRELLIS 2.0)
---------------------------------
- sparse_structure_flow_model   : decides which voxels to activate (dense grid → coords)
- shape_slat_flow_model_512/1024: per-voxel shape features  (SparseTensor)
- tex_slat_flow_model_512/1024  : per-voxel texture features (SparseTensor)
- shape_slat_decoder            : SparseTensor → geometry Mesh
- tex_slat_decoder              : SparseTensor → texture voxels → PBR

FlowEdit does NOT require inversion.  Source and target trajectories both
start from the same random z_T; structural fidelity comes from the source
condition (image features of rendered views), not from ODE inversion.

This wrapper exposes only the primitives needed by the FlowEdit loop:
  get_cond → normalize → predict_velocity / euler_step → denormalize → decode_latent

Timestep convention
-------------------
Flow model expects t ∈ [0, 1000].  External API here uses t ∈ [0, 1] and
multiplies by 1000 internally (matching _inference_model in FlowEulerSampler).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from PIL import Image

try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor
    from trellis2.representations import MeshWithVoxel
except ImportError as exc:
    raise ImportError(
        "trellis2 package not found. "
        "Install via the project's setup.sh or: "
        "pip install git+https://github.com/microsoft/TRELLIS.2"
    ) from exc


# ---------------------------------------------------------------------------
# Condition container
# ---------------------------------------------------------------------------

@dataclass
class Trellis2Condition:
    """Holds cond / neg_cond tensors ready to be passed to a flow model."""
    cond:     torch.Tensor
    neg_cond: torch.Tensor


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class TrellisWrapper:
    """
    Thin wrapper around Trellis2ImageTo3DPipeline for FlowEdit-based 3D editing.

    FlowEdit workflow
    -----------------
    1.  wrapper  = TrellisWrapper.from_config(cfg)
    2.  src_cond = wrapper.get_cond([src_view, ...], resolution=512)
    3.  tgt_cond = wrapper.get_cond([tgt_image],     resolution=512)
    4.  shape_slat, tex_slat, res = wrapper.run_source(src_image)
    5.  shape_z0 = wrapper.normalize_shape_slat(shape_slat)
    6.  tex_z0   = wrapper.normalize_tex_slat(tex_slat)
    7.  z_T_shape = wrapper.make_noise_like(shape_z0, wrapper.shape_in_channels)
    8.  z_T_tex   = wrapper.make_noise_like(tex_z0,   wrapper.tex_in_channels)
    9.  # --- FlowEdit loop (implemented in flow_edit.py) ---
    10. # use predict_velocity / euler_step with src_cond and tgt_cond
    11. shape_edit = wrapper.denormalize_shape_slat(shape_z0_edit)
    12. tex_edit   = wrapper.denormalize_tex_slat(tex_z0_edit)
    13. meshes     = wrapper.decode_latent(shape_edit, tex_edit, res)
    """

    def __init__(
        self,
        pipeline: Trellis2ImageTo3DPipeline,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.pipeline   = pipeline
        self.cfg        = cfg
        self.device     = device
        self.resolution = cfg.model.trellis.resolution  # 512 / 1024 / 1536

        # Select shape/tex flow models by configured resolution.
        # Cascade pipelines use 512 for structure, 1024 for hi-res SLAT.
        res_key           = "512" if self.resolution <= 512 else "1024"
        self._shape_model = pipeline.models[f"shape_slat_flow_model_{res_key}"]
        self._tex_model   = pipeline.models[f"tex_slat_flow_model_{res_key}"]

        # ── Sparse-Structure VAE ────────────────────────────────────────
        # The image-to-3D pipeline only *loads* the SS decoder (generation
        # starts from noise, so no occupancy ever needs encoding). FlowEdit,
        # however, must encode the SOURCE occupancy into the SS flow-model's
        # latent space (z_s ∈ [B, 8, 16, 16, 16]) to use it as x_src. We load
        # the matching SS encoder (the training-time counterpart of the decoder
        # the pipeline already uses) from the original TRELLIS-1 repo.
        self.ss_decoder = pipeline.models.get("sparse_structure_decoder")
        self.ss_encoder = self._load_ss_encoder(cfg)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "TrellisWrapper":
        """Build wrapper from Hydra / OmegaConf config (see configs/base.yaml)."""
        device   = torch.device(cfg.device)
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(cfg.model.trellis.ckpt)
        pipeline.to(device)
        pipeline.eval()
        return cls(pipeline, cfg, device)

    # ------------------------------------------------------------------
    # Sparse-Structure VAE  (encode source occupancy / decode edited latent)
    # ------------------------------------------------------------------

    # SS VAE native input/output occupancy resolution (16³ latent → 64³ occ).
    SS_OCC_RES = 64

    @staticmethod
    def _load_ss_encoder(cfg):
        """
        Load the SparseStructureEncoder that matches the pipeline's SS decoder.

        The checkpoint defaults to the original TRELLIS-1 SS VAE
        (`ss_enc_conv3d_16l8_fp16`: in_channels=1, latent_channels=8) which is
        the exact training-time counterpart of the SS decoder referenced by
        TRELLIS.2's pipeline.json. Override with cfg.model.trellis.ss_encoder_ckpt.
        """
        from trellis2 import models as t2_models

        ckpt = cfg.model.trellis.get(
            "ss_encoder_ckpt",
            "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16",
        )
        enc = t2_models.from_pretrained(ckpt)
        enc.eval()
        return enc

    @torch.no_grad()
    def ss_encode(self, occ: torch.Tensor) -> torch.Tensor:
        """
        Encode a binary occupancy grid → SS flow-model latent.

        Args:
            occ : [B, 1, 64, 64, 64] float occupancy (1.0 occupied, 0.0 empty).

        Returns:
            z_s : [B, 8, 16, 16, 16] latent (posterior mean, deterministic).
        """
        enc = self.ss_encoder
        enc.to(self.device)
        z = enc(occ.to(self.device))            # sample_posterior=False → mean
        if self.pipeline.low_vram:
            enc.cpu()
        return z

    @torch.no_grad()
    def ss_decode_occupancy(self, z_s: torch.Tensor) -> torch.Tensor:
        """
        Decode an SS latent → binary occupancy at the VAE's native 64³ grid.

        Args:
            z_s : [B, 8, 16, 16, 16] latent.

        Returns:
            occ : [B, 1, 64, 64, 64] bool occupancy (decoder logit > 0).
        """
        dec = self.ss_decoder
        if dec is None:
            raise RuntimeError(
                "pipeline.models['sparse_structure_decoder'] not loaded — cannot "
                "decode the Stage-1 latent. Check TRELLIS.2's pipeline.json."
            )
        dec.to(self.device)
        occ = dec(z_s.to(self.device)) > 0      # [B, 1, 64, 64, 64]
        if self.pipeline.low_vram:
            dec.cpu()
        return occ

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def get_cond(
        self,
        images: List[Image.Image],
        resolution: int = 512,
    ) -> Trellis2Condition:
        """
        Encode PIL images into flow-model conditioning tensors.

        Args:
            images:     One or more RGB PIL images (e.g. rendered views of source).
            resolution: Feature extraction resolution — 512 or 1024.

        Returns:
            Trellis2Condition(.cond, .neg_cond) where neg_cond is all-zeros.
        """
        d = self.pipeline.get_cond(images, resolution=resolution)
        return Trellis2Condition(cond=d["cond"], neg_cond=d["neg_cond"])

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def normalize_shape_slat(self, slat: SparseTensor) -> SparseTensor:
        """Denormalized shape SLAT (mesh space) → normalized flow latent."""
        std  = torch.tensor(self.pipeline.shape_slat_normalization["std"],  device=slat.device)[None]
        mean = torch.tensor(self.pipeline.shape_slat_normalization["mean"], device=slat.device)[None]
        return (slat - mean) / std

    def denormalize_shape_slat(self, slat: SparseTensor) -> SparseTensor:
        """Normalized flow latent → denormalized shape SLAT (mesh space)."""
        std  = torch.tensor(self.pipeline.shape_slat_normalization["std"],  device=slat.device)[None]
        mean = torch.tensor(self.pipeline.shape_slat_normalization["mean"], device=slat.device)[None]
        return slat * std + mean

    def normalize_tex_slat(self, slat: SparseTensor) -> SparseTensor:
        """Denormalized texture SLAT → normalized flow latent."""
        std  = torch.tensor(self.pipeline.tex_slat_normalization["std"],  device=slat.device)[None]
        mean = torch.tensor(self.pipeline.tex_slat_normalization["mean"], device=slat.device)[None]
        return (slat - mean) / std

    def denormalize_tex_slat(self, slat: SparseTensor) -> SparseTensor:
        """Normalized flow latent → denormalized texture SLAT."""
        std  = torch.tensor(self.pipeline.tex_slat_normalization["std"],  device=slat.device)[None]
        mean = torch.tensor(self.pipeline.tex_slat_normalization["mean"], device=slat.device)[None]
        return slat * std + mean

    # ------------------------------------------------------------------
    # Core velocity prediction  (used inside the FlowEdit loop)
    # ------------------------------------------------------------------

    def _to_model_t(self, t: float, batch_size: int, device: torch.device) -> torch.Tensor:
        """Scale t ∈ [0,1] → [0,1000] and broadcast to [batch_size]."""
        return torch.tensor([1000.0 * t] * batch_size, dtype=torch.float32, device=device)

    def _batch_size(self, x: SparseTensor) -> int:
        return int(x.coords[:, 0].max().item()) + 1

    def predict_velocity(
        self,
        x_t:               SparseTensor,
        t:                 float,
        condition:         Trellis2Condition,
        guidance_strength: float = 7.5,
        model:             str   = "tex",
        **model_kwargs,
    ) -> SparseTensor:
        """
        Predict rectified-flow velocity  v(x_t, t | condition)  with CFG.

        CFG formula (matches FlowEulerCfgSampler):
            v = guidance_strength * v_cond + (1 - guidance_strength) * v_uncond

        Args:
            x_t:               normalized flow latent at timestep t
            t:                 float in [0, 1]
            condition:         Trellis2Condition from get_cond()
            guidance_strength: CFG scale  (1.0 = conditional only, no blending)
            model:             'shape' or 'tex'
            **model_kwargs:    extra kwargs forwarded to the flow model (e.g. concat_cond)

        Returns:
            SparseTensor velocity with same coordinates as x_t
        """
        flow_model = self._shape_model if model == "shape" else self._tex_model
        B        = self._batch_size(x_t)
        t_tensor = self._to_model_t(t, B, x_t.device)

        if self.pipeline.low_vram:
            flow_model.to(self.device)

        v_cond = flow_model(x_t, t_tensor, condition.cond, **model_kwargs)

        if guidance_strength != 1.0:
            v_neg = flow_model(x_t, t_tensor, condition.neg_cond, **model_kwargs)
            v = guidance_strength * v_cond + (1.0 - guidance_strength) * v_neg
        else:
            v = v_cond

        if self.pipeline.low_vram:
            flow_model.cpu()

        return v

    # ------------------------------------------------------------------
    # ODE step primitives  (used inside the FlowEdit loop)
    # ------------------------------------------------------------------

    def euler_step(
        self,
        x_t:               SparseTensor,
        t_curr:            float,
        t_next:            float,
        condition:         Trellis2Condition,
        guidance_strength: float = 7.5,
        model:             str   = "tex",
        **model_kwargs,
    ) -> SparseTensor:
        """
        Single Euler step:  x_{t_next} = x_t - (t_curr - t_next) * v(x_t, t_curr).

        t_next < t_curr for denoising (FlowEdit goes t: 1 → 0).
        """
        v  = self.predict_velocity(x_t, t_curr, condition, guidance_strength, model, **model_kwargs)
        dt = t_curr - t_next
        return x_t.replace(feats=x_t.feats - dt * v.feats)

    def heun_step(
        self,
        x_t:               SparseTensor,
        t_curr:            float,
        t_next:            float,
        condition:         Trellis2Condition,
        guidance_strength: float = 7.5,
        model:             str   = "tex",
        **model_kwargs,
    ) -> SparseTensor:
        """Heun (trapezoidal) step — 2nd-order, 2× NFE per step."""
        dt      = t_curr - t_next
        v1      = self.predict_velocity(x_t, t_curr, condition, guidance_strength, model, **model_kwargs)
        x_euler = x_t.replace(feats=x_t.feats - dt * v1.feats)
        v2      = self.predict_velocity(x_euler, t_next, condition, guidance_strength, model, **model_kwargs)
        return x_t.replace(feats=x_t.feats - 0.5 * dt * (v1.feats + v2.feats))

    # ------------------------------------------------------------------
    # Decode latent → 3D
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat:   SparseTensor,
        resolution: Optional[int] = None,
    ) -> List[MeshWithVoxel]:
        """
        Decode denormalized shape + texture SLaTs to MeshWithVoxel objects.

        Call denormalize_shape_slat / denormalize_tex_slat before passing here.

        Returns:
            List of MeshWithVoxel, one per batch element.
        """
        return self.pipeline.decode_latent(shape_slat, tex_slat, resolution or self.resolution)

    # ------------------------------------------------------------------
    # Source asset generation  (bootstraps the editing pipeline)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_source(
        self,
        image:         Image.Image,
        seed:          int  = 42,
        preprocess:    bool = True,
        pipeline_type: Optional[str] = None,
    ) -> Tuple[SparseTensor, SparseTensor, int]:
        """
        Run full image→3D to obtain denormalized (shape_slat, tex_slat, resolution).

        These are the raw SLaTs in mesh space; pass through normalize_*_slat()
        before using them in the editing pipeline.

        Args:
            image:         RGB PIL image of the source object.
            seed:          random seed.
            preprocess:    whether to run background removal & crop.
            pipeline_type: '512' / '1024' / '1024_cascade' / '1536_cascade'.

        Returns:
            (shape_slat, tex_slat, resolution)
        """
        _, (shape_slat, tex_slat, res) = self.pipeline.run(
            image,
            num_samples=1,
            seed=seed,
            preprocess_image=preprocess,
            return_latent=True,
            pipeline_type=pipeline_type,
        )
        return shape_slat, tex_slat, res

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def make_noise_like(self, slat: SparseTensor, in_channels: int) -> SparseTensor:
        """
        Random noise SparseTensor sharing voxel coordinates with `slat`.

        Used to sample the shared z_T from which both source and target
        FlowEdit trajectories start.
        """
        noise = torch.randn(slat.coords.shape[0], in_channels,
                            dtype=torch.float32, device=slat.device)
        return slat.replace(feats=noise)

    @property
    def shape_in_channels(self) -> int:
        return self._shape_model.in_channels

    @property
    def tex_in_channels(self) -> int:
        return self._tex_model.in_channels

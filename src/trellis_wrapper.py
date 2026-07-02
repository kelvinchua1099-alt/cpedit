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

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    # Default SS-VAE encoder (training-time counterpart of the decoder that
    # TRELLIS.2 already ships; non-gated).  Overridable via
    # cfg.model.trellis.ss_encoder_ckpt.
    _DEFAULT_SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "TrellisWrapper":
        """Build wrapper from Hydra / OmegaConf config (see configs/base.yaml)."""
        device   = torch.device(cfg.device)
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(cfg.model.trellis.ckpt)
        # The custom editing code (vs3d) calls the flow models directly, bypassing
        # the pipeline's per-op low_vram device shuffling.  Keep every model
        # resident on-device so those direct calls don't hit CPU/GPU mismatches.
        # Models are already .eval()'d inside Pipeline.__init__ — there is no
        # pipeline.eval() in this build.
        pipeline.low_vram = False
        pipeline.to(device)
        wrapper = cls(pipeline, cfg, device)
        wrapper._load_ss_vae()
        return wrapper

    # ------------------------------------------------------------------
    # Sparse-Structure VAE (Stage-1 latent space)
    # ------------------------------------------------------------------

    def _load_ss_vae(self) -> None:
        """
        Load the SS encoder (TRELLIS.2 ships only the SS *decoder*).  Stage-1's
        flow model operates on the latent z_s ∈ [B, 8, 16, 16, 16]; the encoder
        maps a 64³ occupancy grid into that latent, the decoder maps back.
        """
        from trellis2 import models as _t2_models

        src = self.cfg.model.trellis.get("ss_encoder_ckpt", self._DEFAULT_SS_ENCODER)
        self.ss_encoder = _t2_models.from_pretrained(src).to(self.device).eval()
        self.ss_decoder = self.pipeline.models["sparse_structure_decoder"]
        self.ss_occ_res = 64   # native occupancy grid of the 16l8 SS VAE

    @torch.no_grad()
    def ss_encode(self, occ: torch.Tensor) -> torch.Tensor:
        """Occupancy [B,1,64,64,64] → latent z_s [B,8,16,16,16] (float32)."""
        dt = next(self.ss_encoder.parameters()).dtype
        z = self.ss_encoder(occ.to(self.device, dtype=dt))
        return z.float()

    @torch.no_grad()
    def ss_decode(self, z_s: torch.Tensor) -> torch.Tensor:
        """Latent z_s [B,8,16,16,16] → occupancy logits [B,1,64,64,64] (float32)."""
        dt = next(self.ss_decoder.parameters()).dtype
        return self.ss_decoder(z_s.to(self.device, dtype=dt)).float()

    # ------------------------------------------------------------------
    # Stage-level VRAM placement
    # ------------------------------------------------------------------
    # With every model resident (low_vram=False) forward editing fits in 24 GB,
    # but RASI's *backward* pass through the SS flow model does not.  During
    # Stage-1 only the SS models are needed, so we offload the (large) SLAT flow
    # models + decoders to CPU, then restore them for Stage-2/3 TAR.

    _SS_KEEP = ("sparse_structure_flow_model", "sparse_structure_decoder")

    def offload_for_stage1(self) -> None:
        """Keep only the SS models on-device; move the rest to CPU."""
        for k, m in self.pipeline.models.items():
            m.to(self.device if k in self._SS_KEEP else "cpu")
        self.ss_encoder.to(self.device)
        torch.cuda.empty_cache()

    def restore_all(self) -> None:
        """Move every model back on-device for the SLAT (TAR) stages."""
        for m in self.pipeline.models.values():
            m.to(self.device)
        torch.cuda.empty_cache()

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

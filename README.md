# cpedit — VS3D Crop-Guided 3D Editing on TRELLIS.2

A minimal, modular pipeline for **image-conditioned 3D asset editing** built on top of
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (the 4B O-Voxel image-to-3D model).
Given a source image and a 2D-edited target image, it produces an edited 3D asset using a
FlowEdit-style scheme (RASI + PMG in the sparse-structure latent space, TAR on the
shape/texture SLATs), with an optional **crop-guidance** signal that strengthens the edit
inside the changed region.

The headline experiment is an ablation: **`full` (with crop guidance)** vs
**`no_crop`** on [`yejunliang23/Nano3D-Edit-100k`](https://huggingface.co/datasets/yejunliang23/Nano3D-Edit-100k).

---

## 1. What this fork changed vs. a stock checkout

This codebase was written against an API that didn't match the released TRELLIS.2 build.
The following changes make it run end-to-end against `microsoft/TRELLIS.2-4B`:

1. **Sparse-Structure VAE wired into Stage-1** (`src/trellis_wrapper.py`, `src/pipeline.py`).
   TRELLIS.2's pipeline only loads the SS **decoder**; its SS flow model works in a
   latent space `z_s ∈ [B, 8, 16, 16, 16]`, *not* on raw occupancy. We load the matching
   `SparseStructureEncoder` (the TRELLIS-1 `ss_enc_conv3d_16l8` weights, the training-time
   counterpart of the decoder TRELLIS.2 already uses) and:
   - `_build_stage1_src_latent`: scatter the source SLAT coords onto a 64³ occupancy grid
     (rescaled from the runtime-inferred cascade grid) → `ss_encode` → `x_src` latent.
   - `_decode_stage1_to_coords`: decode the edited latent through the SS decoder to a 64³
     occupancy, lift active voxels back onto the source grid (no exception-swallowing).
   - `_infer_grid_res`: detect the source SLAT lattice at runtime so Stage-1 (64³) and
     Stage-2/3 stay on a consistent grid.
2. **GLB export fixed** (`src/pipeline.py::_export_glb`). `MeshWithVoxel` has no
   `.export()`/`.render()`; `.save()` only writes `.ply/.npz/.vxz`. GLB extraction now goes
   through `o_voxel.postprocess.to_glb(...).export(...)` (remesh + decimate + UV + bake),
   exactly as in `TRELLIS.2/example.py`.

> **Known limitation (resolution coupling).** `configs/base.yaml` uses `resolution: 1024`,
> whose SLAT coords live on a grid finer than the SS VAE's native 64³. Stage-1 edits the
> coarse 64³ structure and lifts results back onto the source grid, so TAR's source-residual
> injection is exact only on the shared 64³-aligned lattice (identity preservation in
> absolute terms is conservative). **Both ablation arms are affected identically**, so the
> `full` vs `no_crop` comparison stays valid. For higher absolute fidelity set
> `model.trellis.resolution: 512`.

---

## 2. System requirements

- **OS**: Linux. **GPU**: NVIDIA, ≥24 GB VRAM (verified on RTX 4090, driver 570.x / CUDA 12.8 runtime).
- **CUDA Toolkit 12.4** (`nvcc`) on `PATH` — must match the Torch CUDA build (12.4). `gcc/g++ 11`.
- **Disk**: ~20 GB for checkpoints + dataset shards.

---

## 3. Python environment

> ⚠️ **The single most important constraint: keep `torch == 2.4.1+cu124`.**
> Several deps (e.g. `transformers>=5`) will silently pull the latest PyPI torch
> (`2.x+cu130`), which breaks CUDA (driver too old) **and** every CUDA-extension build
> (nvcc 12.4 vs torch-cu13.0 mismatch). After *any* bulk install, re-check
> `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"` → must be
> `12.4 True`. Install torch-dependent extensions with `--no-deps` so they never re-resolve torch.

Package manager: **uv** for normal wheels; **pip `--no-build-isolation --no-deps`** for the
CUDA extensions (uv's isolated builds re-pull torch and can't drive `--no-build-isolation`
source builds cleanly).

### 3.1 Base wheels

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Torch MUST be cu124 (matches nvcc 12.4 + the driver)
uv pip install --system torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Core deps (note: transformers pinned <5 so it does NOT bump torch; 4.56+ has DINOv3)
uv pip install --system \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh "transformers>=4.56,<5" pandas lpips zstandard kornia timm omegaconf safetensors
uv pip install --system \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# flex_gemm (below) needs Autotuner.do_bench → triton 3.2.0 (works with torch 2.4.1)
uv pip install --system --no-deps triton==3.2.0
```

### 3.2 Attention backend (flash-attn) — prebuilt wheel, no source build

Match the Torch ABI (`cxx11abiFALSE`) and Python tag (`cp311`):

```bash
uv pip install --system \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
```

### 3.3 CUDA extensions (compile against system torch)

Order matters: **cumesh → flex_gemm → o-voxel → nvdiffrast → nvdiffrec**
(o-voxel depends on cumesh). Eigen headers are required by o-voxel.

```bash
apt-get install -y libeigen3-dev          # provides /usr/include/eigen3

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"          # RTX 4090 = sm_89
export MAX_JOBS=8
export CPLUS_INCLUDE_PATH=/usr/include/eigen3:$CPLUS_INCLUDE_PATH

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git   /tmp/ext/CuMesh
git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM
git clone -b v0.4.0    https://github.com/NVlabs/nvdiffrast.git     /tmp/ext/nvdiffrast
git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec

pip install --no-build-isolation --no-deps /tmp/ext/CuMesh
pip install --no-build-isolation --no-deps /tmp/ext/FlexGEMM
pip install --no-build-isolation --no-deps /path/to/TRELLIS.2/o-voxel
pip install --no-build-isolation --no-deps /tmp/ext/nvdiffrast
pip install --no-build-isolation --no-deps /tmp/ext/nvdiffrec
```

A correct install prints, on `import trellis2`:
`[SPARSE] Conv backend: flex_gemm; Attention backend: flash_attn`.

### 3.4 Evaluation deps

```bash
uv pip install --system scikit-image scipy open-clip-torch pyrender
# open-clip-torch depends on torch → add --no-deps if it tries to upgrade torch
```

---

## 4. Checkpoints

Set `CKPT_DIR` and download the 4B model into `$CKPT_DIR/trellis2-o-voxel-4b`
(matches `configs/base.yaml: model.trellis.ckpt`).

```bash
export CKPT_DIR=/workspace/ckpts
huggingface-cli download microsoft/TRELLIS.2-4B --local-dir $CKPT_DIR/trellis2-o-voxel-4b
```

Pulled automatically at runtime (HF cache) — all need a HuggingFace login; the two gated
repos need a one-click access request on their model pages:

| Repo | Role | Gated |
|------|------|-------|
| `microsoft/TRELLIS-image-large` (`ss_enc/ss_dec_conv3d_16l8_fp16`) | SS VAE encoder (this fork) + decoder | no |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | image conditioning encoder | **yes** |
| `briaai/RMBG-2.0` | background removal (preprocess) | **yes** |

> `model.trellis.ss_encoder_ckpt` can override the SS encoder source (defaults to the
> TRELLIS-1 path above).

---

## 5. Dataset (`yejunliang23/Nano3D-Edit-100k`)

Sharded as `v1_100k/editing_assets/part_XXX.tar.gz` (1000 `uid_*` objects each). Each
`uid_N/` contains `source.png` (512²), `edit.png` (1024²), `edit_512.png` (512²),
`src_mesh.glb`, `tar_mesh.glb`, `src_slat.pt`, `tar_slat.pt`, …

> **Use `source.png` + `edit_512.png`** (both 512²). `source.png` and `edit.png` differ in
> size, which `diff_crop_resize` (crop mode) rejects. `tar_mesh.glb` is the GT for PSNR/SSIM/LPIPS;
> `src_mesh.glb` is the reference for identity preservation.

```bash
huggingface-cli download yejunliang23/Nano3D-Edit-100k \
  v1_100k/editing_assets/part_000.tar.gz --repo-type dataset --local-dir data/nano3d_raw
tar xzf data/nano3d_raw/v1_100k/editing_assets/part_000.tar.gz -C data/nano3d \
  uid_0/source.png uid_0/edit_512.png uid_0/src_mesh.glb uid_0/tar_mesh.glb   # etc.
```

---

## 6. Running

```bash
export CKPT_DIR=/workspace/ckpts
export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/path/to/TRELLIS.2:$PYTHONPATH   # trellis2 is source-only, NOT pip-installed
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # helps fit in 24 GB

# full (with crop guidance)
python scripts/run_vs3d.py \
  --src data/nano3d/uid_0/source.png \
  --tgt data/nano3d/uid_0/edit_512.png \
  --out results/full/uid_0

# no_crop ablation
python scripts/run_vs3d.py \
  --src data/nano3d/uid_0/source.png \
  --tgt data/nano3d/uid_0/edit_512.png \
  --out results/no_crop/uid_0 --no-crop
```

Outputs per run: `result_00.glb`, `crop_src.png`/`crop_tgt.png` (full mode),
`metadata.json`. Override any config key inline, e.g.
`--set editing.crop.guidance_scale=0.8` or `--set model.trellis.resolution=512`.

---

## 7. Layout

```
src/
  preprocess.py       diff_crop_resize (edit-region crop)
  trellis_wrapper.py  TRELLIS.2 adapter + SS VAE encode/decode (this fork)
  vs3d.py             RASI / PMG / TAR editor
  guidance.py         two-view guidance helpers
  pipeline.py         EditPipeline orchestration (Steps 1-7)
  eval.py             metrics (PSNR/SSIM/LPIPS/CLIP/identity)
scripts/
  run_vs3d.py         single-object CLI (--src/--tgt/--out/--no-crop/--set)
  run.py, ablation.py
configs/
  base.yaml full.yaml no_crop.yaml no_vs3d.yaml no_two_view.yaml
```

"""
run.py — config-driven single VS3D edit + evaluation.

Usage
-----
  # from a source reference image + 2D edit target
  python scripts/run.py --config full \
      --src data/nano3d/uid_0/source.png \
      --tgt data/nano3d/uid_0/edit_512.png \
      --out outputs/results/full/uid_0 \
      --gt-mesh  data/nano3d/uid_0/tar_mesh.glb \
      --src-mesh data/nano3d/uid_0/src_mesh.glb

  # from a 3D asset file (rendered to a reference view first)
  python scripts/run.py --config full \
      --asset data/chair.glb --tgt data/chair_edit.png --out outputs/results/full/chair

  --config  : config name (full / no_crop / no_vs3d / no_two_view) OR a path to a yaml
  --set K=V : OmegaConf dotlist overrides, e.g. --set model.trellis.resolution=512

Config resolution: configs/base.yaml  ⊕  configs/<config>.yaml  ⊕  --set overrides.
After the pipeline runs, src/eval.py computes lightweight metrics into metrics.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from omegaconf import OmegaConf
from PIL import Image

from src.pipeline import EditPipeline
from src.eval import evaluate_result


# ---------------------------------------------------------------------------
# Config loading  (base ⊕ exp ⊕ overrides, 'defaults' stripped)
# ---------------------------------------------------------------------------

def load_config(config: str, overrides: Optional[List[str]] = None):
    cfg_dir = os.path.join(_ROOT, "configs")
    base_cfg = OmegaConf.load(os.path.join(cfg_dir, "base.yaml"))

    # accept a bare name ("full") or a path ("configs/full.yaml", "/abs/x.yaml")
    if config.endswith(".yaml") or os.path.sep in config:
        exp_path = config if os.path.isabs(config) else os.path.join(_ROOT, config)
    else:
        exp_path = os.path.join(cfg_dir, f"{config}.yaml")
    exp_cfg = OmegaConf.load(exp_path)

    if "defaults" in exp_cfg:
        exp_cfg = OmegaConf.masked_copy(
            exp_cfg, [k for k in exp_cfg if k != "defaults"]
        )

    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    cfg,
    out_dir: str,
    src: Optional[str] = None,
    tgt: str = None,
    asset: Optional[str] = None,
    gt_mesh: Optional[str] = None,
    src_mesh: Optional[str] = None,
    elevation: float = 15.0,
    azimuth: float = 30.0,
    pipeline: Optional[EditPipeline] = None,
) -> dict:
    """Run one edit and evaluate it.  Reuses `pipeline` (a loaded model) if given."""
    if pipeline is None:
        pipeline = EditPipeline.from_config(cfg)

    img_tgt = Image.open(tgt).convert("RGB")

    if asset:
        result = pipeline.run_from_asset(
            asset_path=asset, img_tgt=img_tgt, output_dir=out_dir,
            elevation=elevation, azimuth=azimuth,
        )
    else:
        result = pipeline.run(
            Image.open(src).convert("RGB"), img_tgt, output_dir=out_dir,
        )

    metrics = evaluate_result(
        out_dir, result["meta"],
        gt_mesh=gt_mesh, src_mesh=src_mesh, tgt_image=tgt, save=True,
    )
    result["metrics"] = metrics
    _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    meta = result.get("meta", {})
    met = result.get("metrics", {})
    print("─" * 60)
    print(f"  exp            : {meta.get('exp_name')}")
    print(f"  output dir     : {result['output_dir']}")
    print(f"  voxels (C_tgt) : {meta.get('n_voxels_tgt', '?')}")
    print(f"  crop active    : {meta.get('crop_guidance_active', False)}")
    if met.get("chamfer_to_gt") is not None:
        print(f"  chamfer→GT     : {met['chamfer_to_gt']:.5f}")
    if met.get("chamfer_to_src") is not None:
        print(f"  chamfer→src    : {met['chamfer_to_src']:.5f}")
    for p in meta.get("glb_paths", []):
        print(f"  saved GLB      : {p}")
    print("─" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VS3D config-driven single run")
    p.add_argument("--config", default="full",
                   help="config name (full/no_crop/no_vs3d/no_two_view) or yaml path")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", help="source reference image")
    g.add_argument("--asset", help="3D asset file (.glb/.obj/.ply) to render + edit")
    p.add_argument("--tgt", required=True, help="2D edit target image")
    p.add_argument("--out", default=None, help="output dir (auto if omitted)")
    p.add_argument("--gt-mesh", default=None, help="ground-truth edited mesh (eval)")
    p.add_argument("--src-mesh", default=None, help="source mesh (identity eval)")
    p.add_argument("--elev", type=float, default=15.0, help="asset render elevation")
    p.add_argument("--azim", type=float, default=30.0, help="asset render azimuth")
    p.add_argument("--set", nargs="*", default=[], metavar="K=V",
                   help="OmegaConf overrides, e.g. --set model.trellis.resolution=512")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, args.set)

    stem = os.path.splitext(os.path.basename(args.asset or args.src))[0]
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out or os.path.join("outputs", "results", cfg_name, stem)

    run_single(
        cfg, out_dir,
        src=args.src, tgt=args.tgt, asset=args.asset,
        gt_mesh=args.gt_mesh, src_mesh=args.src_mesh,
        elevation=args.elev, azimuth=args.azim,
    )


if __name__ == "__main__":
    main()

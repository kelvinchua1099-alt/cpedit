"""
run_vs3d.py — run the VS3D editing pipeline.

两种使用方式
-----------
1. 带 crop（推荐，利用额外的 crop 引导信号）:
   python scripts/run_vs3d.py \\
       --src  data/kirby.png \\
       --tgt  data/kirby_edited.png \\
       --out  outputs/with_crop

2. 不带 crop（消融对比）:
   python scripts/run_vs3d.py \\
       --src  data/kirby.png \\
       --tgt  data/kirby_edited.png \\
       --out  outputs/no_crop \\
       --no-crop

Config 加载顺序:  base.yaml  →  full.yaml / no_crop.yaml  →  命令行 overrides (--set)

示例
----
# 修改 guidance_scale 而不改 yaml:
python scripts/run_vs3d.py --src a.png --tgt b.png --set editing.crop.guidance_scale=0.8
"""
from __future__ import annotations

import argparse
import os
import sys

# 把 TrellisCropEdit 根目录加入路径，使 src.* 可以 import
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from omegaconf import OmegaConf
from PIL import Image

from src.pipeline import EditPipeline


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg(config_name: str, overrides: list):
    """
    Load base.yaml, then merge config_name.yaml on top, then apply
    any dot-notation overrides from the command line.

    config_name: "full" or "no_crop" (filename without .yaml)
    overrides:   list of "key=value" strings  e.g. ["editing.crop.guidance_scale=0.8"]
    """
    cfg_dir  = os.path.join(_ROOT, "configs")
    base_cfg = OmegaConf.load(os.path.join(cfg_dir, "base.yaml"))
    exp_cfg  = OmegaConf.load(os.path.join(cfg_dir, f"{config_name}.yaml"))

    # Drop the Hydra 'defaults' key before merging
    if "defaults" in exp_cfg:
        exp_cfg = OmegaConf.masked_copy(
            exp_cfg, [k for k in exp_cfg if k != "defaults"]
        )

    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg


# ---------------------------------------------------------------------------
# Setting A: with crop
# ---------------------------------------------------------------------------

def run_with_crop(
    src_path:  str,
    tgt_path:  str,
    out_dir:   str,
    overrides: list | None = None,
) -> dict:
    """
    Run VS3D with crop preprocessing + crop velocity guidance signal.

    Stage-1 velocity update per noise draw:
        v_Δ = [v(z | c_tgt_full) − v(z | c_src_full)]
            + guidance_scale × [v(z | c_tgt_crop) − v(z | c_src_crop)]

    The second term amplifies the edit signal inside the detected region
    and is zero outside it (because c_tgt_crop ≈ c_src_crop outside the diff).
    PMG then extrapolates the combined signal.
    """
    cfg = _load_cfg("full", overrides or [])
    print(f"[with_crop]  guidance_scale={cfg.editing.crop.guidance_scale}  "
          f"out → {out_dir}")

    pipeline = EditPipeline.from_config(cfg)
    result   = pipeline.run(
        Image.open(src_path).convert("RGB"),
        Image.open(tgt_path).convert("RGB"),
        output_dir=out_dir,
    )
    _print_summary(result)
    return result


# ---------------------------------------------------------------------------
# Setting B: without crop
# ---------------------------------------------------------------------------

def run_without_crop(
    src_path:  str,
    tgt_path:  str,
    out_dir:   str,
    overrides: list | None = None,
) -> dict:
    """
    Run VS3D without crop preprocessing or crop guidance signal.

    Stage-1 velocity update per noise draw:
        v_Δ = v(z | c_tgt_full) − v(z | c_src_full)

    RASI, PMG, TAR remain active; only the crop extra signal is absent.
    """
    cfg = _load_cfg("no_crop", overrides or [])
    print(f"[no_crop]    crop disabled  out → {out_dir}")

    pipeline = EditPipeline.from_config(cfg)
    result   = pipeline.run(
        Image.open(src_path).convert("RGB"),
        Image.open(tgt_path).convert("RGB"),
        output_dir=out_dir,
    )
    _print_summary(result)
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(result: dict) -> None:
    meta = result.get("meta", {})
    print("─" * 52)
    print(f"  output dir     : {result['output_dir']}")
    print(f"  voxels (C_tgt) : {meta.get('n_voxels_tgt', '?')}")
    print(f"  crop active    : {meta.get('crop_guidance_active', False)}")
    crop = meta.get("crop", {})
    if crop.get("found_change"):
        print(f"  crop bbox      : {crop.get('bbox_crop')}")
    for p in meta.get("glb_paths", []):
        print(f"  saved          : {p}")
    print("─" * 52)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VS3D 3D asset editing  (with or without crop guidance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--src",     required=True,  help="Source (reference) image")
    p.add_argument("--tgt",     required=True,  help="Edited target image")
    p.add_argument("--out",     default=None,   help="Output directory (auto if omitted)")
    p.add_argument("--no-crop", action="store_true",
                   help="Disable crop preprocessing and crop guidance signal")
    p.add_argument("--set",     nargs="*", default=[], metavar="KEY=VALUE",
                   help="Config overrides, e.g. --set editing.crop.guidance_scale=0.8")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    src_stem = os.path.splitext(os.path.basename(args.src))[0]
    mode     = "no_crop" if args.no_crop else "with_crop"
    out_dir  = args.out or os.path.join("outputs", f"{src_stem}_{mode}")

    if args.no_crop:
        run_without_crop(args.src, args.tgt, out_dir, args.set)
    else:
        run_with_crop(args.src, args.tgt, out_dir, args.set)


if __name__ == "__main__":
    main()

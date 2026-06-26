"""
run_vs3d.py — VS3D editing pipeline entry point.

主要接口：从 3D 资产文件 (.glb/.obj/.ply) 直接编辑
---------------------------------------------------
  python scripts/run_vs3d.py \\
      --asset  data/chair.glb \\
      --tgt    data/chair_redpaint.png \\
      --out    outputs/chair_edit

管线会自动把 .glb 渲染成一张参考图（输出到 rendered_src.png），
然后用这张图 + 编辑图走完 VS3D 流程。

备用接口：直接提供参考图（跳过渲染）
--------------------------------------
  python scripts/run_vs3d.py \\
      --src  data/chair_photo.png \\
      --tgt  data/chair_redpaint.png \\
      --out  outputs/chair_edit

开关
----
  --no-crop      关闭 diff-crop 预处理和 crop 引导信号（消融对比用）
  --elev FLOAT   渲染仰角，默认 15°
  --azim FLOAT   渲染方位角，默认 30°（3/4 侧视）
  --set K=V      覆盖 config 值，如 --set editing.crop.guidance_scale=0.8

Config 加载：base.yaml → full.yaml (有 crop) / no_crop.yaml (无 crop) → --set 覆盖
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from omegaconf import OmegaConf
from PIL import Image

from src.pipeline import EditPipeline


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_cfg(config_name: str, overrides: List[str]):
    cfg_dir  = os.path.join(_ROOT, "configs")
    base_cfg = OmegaConf.load(os.path.join(cfg_dir, "base.yaml"))
    exp_cfg  = OmegaConf.load(os.path.join(cfg_dir, f"{config_name}.yaml"))

    if "defaults" in exp_cfg:
        exp_cfg = OmegaConf.masked_copy(
            exp_cfg, [k for k in exp_cfg if k != "defaults"]
        )

    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


# ---------------------------------------------------------------------------
# Primary: from 3D asset file
# ---------------------------------------------------------------------------

def run_from_asset(
    asset_path: str,
    tgt_path:   str,
    out_dir:    str,
    no_crop:    bool = False,
    elevation:  float = 15.0,
    azimuth:    float = 30.0,
    overrides:  Optional[List[str]] = None,
) -> dict:
    """
    从 .glb/.obj/.ply 3D 资产文件开始编辑。

    资产会被渲染成参考图（rendered_src.png 保存到输出目录），
    然后走完和 run_with_crop / run_without_crop 完全一样的 VS3D 流程。
    """
    cfg_name = "no_crop" if no_crop else "full"
    cfg      = _load_cfg(cfg_name, overrides or [])

    print(f"[asset]  {asset_path}  elev={elevation}°  azim={azimuth}°")
    print(f"[asset]  crop={'off' if no_crop else 'on'}  out → {out_dir}")

    pipeline = EditPipeline.from_config(cfg)
    result   = pipeline.run_from_asset(
        asset_path  = asset_path,
        img_tgt     = Image.open(tgt_path).convert("RGB"),
        output_dir  = out_dir,
        elevation   = elevation,
        azimuth     = azimuth,
    )
    _print_summary(result)
    return result


# ---------------------------------------------------------------------------
# Secondary: from source image (backward-compatible)
# ---------------------------------------------------------------------------

def run_with_crop(
    src_path:  str,
    tgt_path:  str,
    out_dir:   str,
    overrides: Optional[List[str]] = None,
) -> dict:
    """全图条件 + crop 引导信号（推荐）。"""
    cfg = _load_cfg("full", overrides or [])
    print(f"[img/with_crop]  guidance_scale={cfg.editing.crop.guidance_scale}  out → {out_dir}")
    pipeline = EditPipeline.from_config(cfg)
    result   = pipeline.run(
        Image.open(src_path).convert("RGB"),
        Image.open(tgt_path).convert("RGB"),
        output_dir=out_dir,
    )
    _print_summary(result)
    return result


def run_without_crop(
    src_path:  str,
    tgt_path:  str,
    out_dir:   str,
    overrides: Optional[List[str]] = None,
) -> dict:
    """仅全图条件，无 crop 信号（消融对比）。"""
    cfg = _load_cfg("no_crop", overrides or [])
    print(f"[img/no_crop]  crop disabled  out → {out_dir}")
    pipeline = EditPipeline.from_config(cfg)
    result   = pipeline.run(
        Image.open(src_path).convert("RGB"),
        Image.open(tgt_path).convert("RGB"),
        output_dir=out_dir,
    )
    _print_summary(result)
    return result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(result: dict) -> None:
    meta = result.get("meta", {})
    print("─" * 56)
    print(f"  output dir     : {result['output_dir']}")
    if "asset_path" in meta:
        print(f"  source asset   : {meta['asset_path']}")
    print(f"  voxels (C_tgt) : {meta.get('n_voxels_tgt', '?')}")
    print(f"  crop active    : {meta.get('crop_guidance_active', False)}")
    crop = meta.get("crop", {})
    if crop.get("found_change"):
        print(f"  crop bbox      : {crop.get('bbox_crop')}")
    for p in meta.get("glb_paths", []):
        print(f"  saved GLB      : {p}")
    if os.path.exists(os.path.join(result["output_dir"], "rendered_src.png")):
        print(f"  rendered src   : {result['output_dir']}/rendered_src.png")
    print("─" * 56)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VS3D 3D asset editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--asset", metavar="FILE",
        help="[主要] 3D 资产文件路径 (.glb / .obj / .ply / .stl)"
    )
    src_group.add_argument(
        "--src", metavar="IMG",
        help="[备用] 源参考图路径（跳过 3D 渲染）"
    )

    p.add_argument("--tgt",      required=True,  metavar="IMG",
                   help="2D 编辑目标图路径")
    p.add_argument("--out",      default=None,   metavar="DIR",
                   help="输出目录（不填则自动生成）")
    p.add_argument("--no-crop",  action="store_true",
                   help="关闭 diff-crop 预处理和 crop 引导信号")
    p.add_argument("--elev",     type=float, default=15.0, metavar="DEG",
                   help="渲染仰角（仅 --asset 模式，默认 15°）")
    p.add_argument("--azim",     type=float, default=30.0, metavar="DEG",
                   help="渲染方位角（仅 --asset 模式，默认 30°）")
    p.add_argument("--set",      nargs="*", default=[], metavar="KEY=VALUE",
                   help="Config 覆盖，如 --set editing.crop.guidance_scale=0.8")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Auto output dir
    stem    = os.path.splitext(os.path.basename(args.asset or args.src))[0]
    mode    = "no_crop" if args.no_crop else "with_crop"
    out_dir = args.out or os.path.join("outputs", f"{stem}_{mode}")

    if args.asset:
        run_from_asset(
            asset_path = args.asset,
            tgt_path   = args.tgt,
            out_dir    = out_dir,
            no_crop    = args.no_crop,
            elevation  = args.elev,
            azimuth    = args.azim,
            overrides  = args.set,
        )
    elif args.no_crop:
        run_without_crop(args.src, args.tgt, out_dir, args.set)
    else:
        run_with_crop(args.src, args.tgt, out_dir, args.set)


if __name__ == "__main__":
    main()

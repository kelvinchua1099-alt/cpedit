"""
ablation.py — run the crop-guidance ablation matrix on one object.

Runs (by default) the four configs:
    full          crop ✓  vs3d ✓  two-view ✓   (headline method)
    no_crop       crop ✗  vs3d ✓  two-view ✓   (main ablation: no crop-resize)
    no_vs3d       crop ✓  vs3d ✗  two-view ✓   (remove RASI/PMG/TAR)
    no_two_view   crop ✓  vs3d ✓  two-view ✗   (remove SplitFlow two-view fusion)

The 16 GB TRELLIS.2 backbone is loaded ONCE and shared across all configs; only
the lightweight VS3DEditor is rebuilt per config.  Each result goes to
    <out-root>/<config_name>/<stem>/
and a combined table is written to <out-root>/ablation_summary.json.

Usage
-----
  python scripts/ablation.py \
      --src data/nano3d/uid_0/source.png \
      --tgt data/nano3d/uid_0/edit_512.png \
      --gt-mesh  data/nano3d/uid_0/tar_mesh.glb \
      --src-mesh data/nano3d/uid_0/src_mesh.glb
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pipeline import EditPipeline
from src.trellis_wrapper import TrellisWrapper
from src.vs3d import VS3DEditor
from scripts.run import load_config, run_single

DEFAULT_CONFIGS = ["full", "no_crop", "no_vs3d", "no_two_view"]


def run_ablation(
    src: Optional[str],
    tgt: str,
    asset: Optional[str] = None,
    gt_mesh: Optional[str] = None,
    src_mesh: Optional[str] = None,
    out_root: str = "outputs/results",
    configs: Optional[List[str]] = None,
    overrides: Optional[List[str]] = None,
    elevation: float = 15.0,
    azimuth: float = 30.0,
) -> dict:
    configs = configs or DEFAULT_CONFIGS
    stem = os.path.splitext(os.path.basename(asset or src))[0]

    # Load the backbone ONCE from the first config (base is shared across all).
    cfg0 = load_config(configs[0], overrides)
    print(f"[ablation] loading TRELLIS.2 backbone once (res={cfg0.model.trellis.resolution}) ...")
    wrapper = TrellisWrapper.from_config(cfg0)

    summary = {}
    for name in configs:
        cfg = load_config(name, overrides)
        out_dir = os.path.join(out_root, name, stem)
        print(f"\n========== [{name}] -> {out_dir} ==========")

        # Rebuild only the editor for this config; reuse the loaded wrapper.
        editor = VS3DEditor.from_config(wrapper, cfg)
        pipeline = EditPipeline(wrapper, editor, cfg)

        try:
            result = run_single(
                cfg, out_dir,
                src=src, tgt=tgt, asset=asset,
                gt_mesh=gt_mesh, src_mesh=src_mesh,
                elevation=elevation, azimuth=azimuth,
                pipeline=pipeline,
            )
            summary[name] = result.get("metrics", {})
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary[name] = {"error": f"{type(e).__name__}: {e}"}

    os.makedirs(out_root, exist_ok=True)
    summary_path = os.path.join(out_root, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_table(summary)
    print(f"\n[ablation] summary saved -> {summary_path}")
    return summary


def _print_table(summary: dict) -> None:
    print("\n" + "=" * 72)
    print(f"{'config':<14}{'chamfer→GT':>13}{'chamfer→src':>13}{'#voxels':>10}{'crop':>7}")
    print("-" * 72)
    for name, m in summary.items():
        if "error" in m:
            print(f"{name:<14}  ERROR: {m['error'][:44]}")
            continue
        def fmt(x): return f"{x:.5f}" if isinstance(x, (int, float)) else "   -   "
        print(f"{name:<14}{fmt(m.get('chamfer_to_gt')):>13}"
              f"{fmt(m.get('chamfer_to_src')):>13}"
              f"{str(m.get('n_voxels_tgt','-')):>10}"
              f"{str(m.get('crop_guidance_active','-')):>7}")
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VS3D crop-guidance ablation")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", help="source reference image")
    g.add_argument("--asset", help="3D asset file to render + edit")
    p.add_argument("--tgt", required=True, help="2D edit target image")
    p.add_argument("--gt-mesh", default=None, help="ground-truth edited mesh")
    p.add_argument("--src-mesh", default=None, help="source mesh (identity ref)")
    p.add_argument("--out-root", default="outputs/results", help="output root dir")
    p.add_argument("--configs", nargs="*", default=None,
                   help=f"configs to run (default: {' '.join(DEFAULT_CONFIGS)})")
    p.add_argument("--elev", type=float, default=15.0)
    p.add_argument("--azim", type=float, default=30.0)
    p.add_argument("--set", nargs="*", default=[], metavar="K=V",
                   help="OmegaConf overrides applied to every config")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_ablation(
        src=args.src, tgt=args.tgt, asset=args.asset,
        gt_mesh=args.gt_mesh, src_mesh=args.src_mesh,
        out_root=args.out_root, configs=args.configs,
        overrides=args.set, elevation=args.elev, azimuth=args.azim,
    )


if __name__ == "__main__":
    main()

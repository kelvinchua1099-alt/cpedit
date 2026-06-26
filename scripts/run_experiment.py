"""
run_experiment.py — batch the full vs. no_crop ablation over a Nano3D subset.

Loads the 16 GB TRELLIS.2 backbone **once** (shared TrellisWrapper) and reuses it
across every (object, mode) pair, rebuilding only the lightweight VS3DEditor per
mode. For each object it runs both `full` (crop guidance) and `no_crop`.

Inputs per object directory:  source.png  (--src)  +  edit_512.png  (--tgt)
  (source.png is 512², edit.png is 1024²; diff_crop_resize needs equal sizes,
   so we use edit_512.png.)

Failures are recorded to results/errors.jsonl and never silently dropped.

Usage:
  PYOPENGL_PLATFORM=egl CKPT_DIR=/workspace/ckpts \
  PYTHONPATH=/workspace/TRELLIS.2 \
  python -B scripts/run_experiment.py --data_root data/nano3d --results_root results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from omegaconf import OmegaConf
from PIL import Image

from src.trellis_wrapper import TrellisWrapper
from src.vs3d import VS3DEditor
from src.pipeline import EditPipeline


def load_cfg(config_name: str):
    cfg_dir = os.path.join(_ROOT, "configs")
    base = OmegaConf.load(os.path.join(cfg_dir, "base.yaml"))
    exp = OmegaConf.load(os.path.join(cfg_dir, f"{config_name}.yaml"))
    if "defaults" in exp:
        exp = OmegaConf.masked_copy(exp, [k for k in exp if k != "defaults"])
    return OmegaConf.merge(base, exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/nano3d")
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--modes", nargs="+", default=["full", "no_crop"])
    ap.add_argument("--src_name", default="source.png")
    ap.add_argument("--tgt_name", default="edit_512.png")
    args = ap.parse_args()

    os.makedirs(args.results_root, exist_ok=True)
    cfgs = {m: load_cfg(m) for m in args.modes}

    # Shared backbone — built once from the first mode's config (model section is
    # identical across modes; only editing.* differs).
    print("Loading TRELLIS.2 backbone (shared across all runs)...", flush=True)
    wrapper = TrellisWrapper.from_config(cfgs[args.modes[0]])

    uids = sorted(d for d in os.listdir(args.data_root)
                  if os.path.isdir(os.path.join(args.data_root, d)))
    errors = []

    for uid in uids:
        obj = os.path.join(args.data_root, uid)
        src_p = os.path.join(obj, args.src_name)
        tgt_p = os.path.join(obj, args.tgt_name)
        if not (os.path.exists(src_p) and os.path.exists(tgt_p)):
            errors.append({"uid": uid, "mode": "*", "reason": "missing src/tgt image"})
            continue

        img_src = Image.open(src_p).convert("RGB")
        img_tgt = Image.open(tgt_p).convert("RGB")

        for mode in args.modes:
            out_dir = os.path.join(args.results_root, mode, uid)
            os.makedirs(out_dir, exist_ok=True)
            print(f"[{mode}] {uid}", flush=True)
            try:
                editor = VS3DEditor.from_config(wrapper, cfgs[mode])
                pipe = EditPipeline(wrapper, editor, cfgs[mode])
                res = pipe.run(img_src, img_tgt, output_dir=out_dir)
                glbs = res["meta"].get("glb_paths", [])
                print(f"  OK  voxels={res['meta'].get('n_voxels_tgt')}  glb={glbs}", flush=True)
            except Exception as e:
                tb = traceback.format_exc()
                errors.append({"uid": uid, "mode": mode,
                               "error": f"{type(e).__name__}: {e}", "traceback": tb})
                print(f"  ERROR: {e}\n{tb[-800:]}", flush=True)

    with open(os.path.join(args.results_root, "errors.jsonl"), "w") as f:
        for e in errors:
            f.write(json.dumps(e) + "\n")
    print(f"\nDone. {len(errors)} errors → {args.results_root}/errors.jsonl", flush=True)


if __name__ == "__main__":
    main()

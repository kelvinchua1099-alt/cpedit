"""
batch_3way.py — 3-way crop comparison with the CORRECT metrics.

For each uid runs three settings — full (multi-crop+SplitFlow), full_pure
(pure-threshold single crop), no_crop — and reports:
  • DINO-I  ↑ : max DINO cosine of multi-view renders vs edit_512 (edit realization)
  • CD->src ↓ : chamfer to the source mesh (identity preservation)
plus the source-mesh DINO-I baseline ("do nothing").

The 16 GB backbone + DINOv2 are loaded ONCE.  After each uid a summary file
outputs/batch3/uid_<i>.json is written (watched for incremental reporting).
"""
from __future__ import annotations
import json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import nvdiffrast.torch as dr
from PIL import Image

from scripts.run import load_config, run_single
from scripts.eval_correct import render_views, Dino
from src.trellis_wrapper import TrellisWrapper
from src.vs3d import VS3DEditor
from src.pipeline import EditPipeline

CONFIGS = ["full", "full_pure", "no_crop"]
OUT = "outputs/batch3"


def main():
    uids = sys.argv[1:] or ["uid_1", "uid_2", "uid_3", "uid_4", "uid_5"]
    os.makedirs(OUT, exist_ok=True)

    cfg0 = load_config(CONFIGS[0], ["model.trellis.resolution=512"])
    print(f"[3way] loading backbone once (res={cfg0.model.trellis.resolution}) ...", flush=True)
    wrapper = TrellisWrapper.from_config(cfg0)
    glctx = dr.RasterizeCudaContext()
    dino = Dino()

    for uid in uids:
        jpath = os.path.join(OUT, f"{uid}.json")
        if os.path.exists(jpath):
            prev = json.load(open(jpath))
            if all("error" not in (prev.get(c) or {}) for c in CONFIGS):
                print(f"[3way] skip {uid} (already done)", flush=True)
                continue

        d = f"data/nano3d/{uid}"
        src, tgt = f"{d}/source.png", f"{d}/edit_512.png"
        gt, sm = f"{d}/tar_mesh.glb", f"{d}/src_mesh.glb"
        edit_img = Image.open(tgt)
        row = {"uid": uid}

        # baseline: source mesh vs edit image (both judges)
        try:
            row["source_dino"] = dino.dino_i_multi(render_views(sm, glctx), edit_img)
        except Exception:
            row["source_dino"] = None

        for name in CONFIGS:
            cfg = load_config(name, ["model.trellis.resolution=512"])
            editor = VS3DEditor.from_config(wrapper, cfg)
            pipeline = EditPipeline(wrapper, editor, cfg)
            out_dir = os.path.join(OUT, name, uid)
            print(f"\n===== [{uid} / {name}] =====", flush=True)
            try:
                result = run_single(cfg, out_dir, src=src, tgt=tgt,
                                    gt_mesh=gt, src_mesh=sm, pipeline=pipeline)
                glb = os.path.join(out_dir, "result_00.glb")
                dino_scores = dino.dino_i_multi(render_views(glb, glctx), edit_img) if os.path.exists(glb) else {}
                m = result.get("meta", {})
                row[name] = {
                    "dino": dino_scores,
                    "cd_src": result.get("metrics", {}).get("chamfer_to_src"),
                    "n_crops": (m.get("crop") or {}).get("n_crops", 0),
                    "voxels": m.get("n_voxels_tgt"),
                }
            except Exception as e:
                import traceback; traceback.print_exc()
                row[name] = {"error": f"{type(e).__name__}: {e}"}

        json.dump(row, open(os.path.join(OUT, f"{uid}.json"), "w"), indent=2)
        print(f"### UID_DONE {uid}", flush=True)

    print("### BATCH3 DONE", flush=True)


if __name__ == "__main__":
    main()

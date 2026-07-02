"""
profile_pipeline.py — Tier-0 profiling: where does wall time go per setting?

Wraps the 7 pipeline phases with CUDA-synced timers and counts the number of
flow-model forwards inside Stage-1 (via _dense_cfg_velocity) so we can see the
(1 + N_crops) scaling directly.  Runs one uid across full / full_pure / no_crop.

Usage: python scripts/profile_pipeline.py [uid]   (default uid_1)
"""
import os, sys, time, json
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
from PIL import Image

from scripts.run import load_config
from src.trellis_wrapper import TrellisWrapper
from src.vs3d import VS3DEditor
from src.pipeline import EditPipeline

CONFIGS = ["full", "full_pure", "no_crop"]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.t = defaultdict(float)
        self.fwd_calls = 0
        self.fwd_time = 0.0

    def wrap(self, obj, name, key):
        orig = getattr(obj, name)
        def wrapped(*a, **k):
            sync(); t0 = time.perf_counter()
            r = orig(*a, **k)
            sync(); self.t[key] += time.perf_counter() - t0
            return r
        setattr(obj, name, wrapped)
        return orig

    def wrap_fwd(self, editor):
        orig = editor._dense_cfg_velocity
        def wrapped(*a, **k):
            sync(); t0 = time.perf_counter()
            r = orig(*a, **k)
            sync(); self.fwd_time += time.perf_counter() - t0
            self.fwd_calls += 1
            return r
        editor._dense_cfg_velocity = wrapped


def main():
    uid = sys.argv[1] if len(sys.argv) > 1 else "uid_1"
    d = f"data/nano3d/{uid}"
    img_src = Image.open(f"{d}/source.png")
    img_tgt = Image.open(f"{d}/edit_512.png")

    cfg0 = load_config(CONFIGS[0], ["model.trellis.resolution=512"])
    print(f"[profile] loading backbone once (res=512), uid={uid} ...", flush=True)
    wrapper = TrellisWrapper.from_config(cfg0)

    rows = {}
    for name in CONFIGS:
        cfg = load_config(name, ["model.trellis.resolution=512"])
        editor = VS3DEditor.from_config(wrapper, cfg)
        pipeline = EditPipeline(wrapper, editor, cfg)
        tm = Timer()
        # phase timers
        tm.wrap(pipeline, "_preprocess",       "1_preprocess_crop")
        tm.wrap(wrapper,  "run_source",         "2_source_asset")
        tm.wrap(pipeline, "_build_conditions",  "3_dinov3_cond")
        tm.wrap(editor,   "run_stage1",         "4_stage1_rasi_pmg")
        tm.wrap(editor,   "edit_sparse_stages", "5_tar_stage23")
        tm.wrap(wrapper,  "decode_latent",      "6_decode")
        tm.wrap(pipeline, "_save_outputs",      "7_save_glb")
        tm.wrap_fwd(editor)

        out_dir = f"outputs/profile/{name}/{uid}"
        os.makedirs(out_dir, exist_ok=True)
        sync(); t0 = time.perf_counter()
        res = pipeline.run(img_src, img_tgt, output_dir=out_dir)
        sync(); total = time.perf_counter() - t0

        n_crops = res["meta"].get("n_crop_signals", 0)
        rows[name] = {
            "total": total,
            "phases": dict(tm.t),
            "fwd_calls": tm.fwd_calls,
            "fwd_time": tm.fwd_time,
            "n_crops": n_crops,
        }
        print(f"\n===== {name}  (n_crops={n_crops}) total={total:.1f}s =====", flush=True)
        for k in sorted(tm.t):
            print(f"  {k:<22} {tm.t[k]:7.1f}s  ({100*tm.t[k]/total:4.1f}%)")
        print(f"  [stage1] flow-model forwards: {tm.fwd_calls}  "
              f"(cumulative {tm.fwd_time:.1f}s incl. per-call sync overhead)")

    json.dump(rows, open("outputs/profile/summary.json", "w"), indent=2)

    # comparison table
    print("\n\n================ PHASE COMPARISON (seconds) ================")
    keys = sorted({k for r in rows.values() for k in r["phases"]})
    hdr = f"{'phase':<22}" + "".join(f"{c:>12}" for c in CONFIGS)
    print(hdr)
    for k in keys:
        print(f"{k:<22}" + "".join(f"{rows[c]['phases'].get(k,0):>12.1f}" for c in CONFIGS))
    print(f"{'TOTAL':<22}" + "".join(f"{rows[c]['total']:>12.1f}" for c in CONFIGS))
    print(f"{'stage1_fwd_calls':<22}" + "".join(f"{rows[c]['fwd_calls']:>12d}" for c in CONFIGS))
    print(f"{'n_crops':<22}" + "".join(f"{rows[c]['n_crops']:>12d}" for c in CONFIGS))


if __name__ == "__main__":
    main()

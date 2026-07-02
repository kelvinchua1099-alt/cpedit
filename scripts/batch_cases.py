"""
batch_cases.py — run full vs no_crop over several dataset objects and compare.

Loads the TRELLIS.2 backbone ONCE and reuses it across every (uid, config) run.
Outputs go to  outputs/batch/<config>/<uid>/  and a combined table is written to
outputs/batch/batch_summary.json.

Usage:
  python scripts/batch_cases.py --uids uid_1 uid_2 uid_3 uid_4 uid_5
"""
from __future__ import annotations
import argparse, json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.run import load_config, run_single
from src.trellis_wrapper import TrellisWrapper
from src.vs3d import VS3DEditor
from src.pipeline import EditPipeline
from src.eval import chamfer_distance

CONFIGS = ["full", "no_crop"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/nano3d")
    p.add_argument("--uids", nargs="+", default=["uid_1", "uid_2", "uid_3", "uid_4", "uid_5"])
    p.add_argument("--out-root", default="outputs/batch")
    p.add_argument("--set", nargs="*", default=["model.trellis.resolution=512"])
    args = p.parse_args()

    cfg0 = load_config(CONFIGS[0], args.set)
    print(f"[batch] loading backbone once (res={cfg0.model.trellis.resolution}) ...", flush=True)
    wrapper = TrellisWrapper.from_config(cfg0)

    results = {}
    for uid in args.uids:
        d = os.path.join(args.data_root, uid)
        src = os.path.join(d, "source.png"); tgt = os.path.join(d, "edit_512.png")
        gt = os.path.join(d, "tar_mesh.glb"); sm = os.path.join(d, "src_mesh.glb")
        if not (os.path.exists(src) and os.path.exists(tgt)):
            print(f"[batch] skip {uid}: missing files"); continue
        results[uid] = {"src_vs_tar": chamfer_distance(sm, gt) if os.path.exists(sm) and os.path.exists(gt) else None}
        for name in CONFIGS:
            cfg = load_config(name, args.set)
            editor = VS3DEditor.from_config(wrapper, cfg)
            pipeline = EditPipeline(wrapper, editor, cfg)
            out_dir = os.path.join(args.out_root, name, uid)
            print(f"\n===== [{uid} / {name}] -> {out_dir} =====", flush=True)
            try:
                r = run_single(cfg, out_dir, src=src, tgt=tgt, gt_mesh=gt, src_mesh=sm, pipeline=pipeline)
                results[uid][name] = r.get("metrics", {})
            except Exception as e:
                import traceback; traceback.print_exc()
                results[uid][name] = {"error": f"{type(e).__name__}: {e}"}
        with open(os.path.join(args.out_root, "batch_summary.json"), "w") as f:
            os.makedirs(args.out_root, exist_ok=True); json.dump(results, f, indent=2)

    _print_table(results)
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "batch_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[batch] saved -> {os.path.join(args.out_root, 'batch_summary.json')}")


def _g(m, k):
    return m.get(k) if isinstance(m, dict) else None


def _print_table(results):
    print("\n" + "=" * 92)
    print(f"{'uid':<8}{'full→GT':>10}{'nocrop→GT':>11}{'Δ(f-nc)':>10}"
          f"{'full→src':>10}{'nocrop→src':>11}{'src↔tar':>10}{'  winner(→GT)':>14}")
    print("-" * 92)
    fwins = ncwins = 0
    fgt = ncgt = 0.0; n = 0
    for uid, r in results.items():
        f, nc = r.get("full", {}), r.get("no_crop", {})
        a, b = _g(f, "chamfer_to_gt"), _g(nc, "chamfer_to_gt")
        fs, bs = _g(f, "chamfer_to_src"), _g(nc, "chamfer_to_src")
        st = r.get("src_vs_tar")
        def fmt(x): return f"{x:.5f}" if isinstance(x, (int, float)) else "   -   "
        win = "-"
        if isinstance(a, float) and isinstance(b, float):
            win = "full" if a < b else "no_crop"
            fwins += a < b; ncwins += b < a
            fgt += a; ncgt += b; n += 1
        d = (a - b) if isinstance(a, float) and isinstance(b, float) else None
        print(f"{uid:<8}{fmt(a):>10}{fmt(b):>11}{fmt(d):>10}{fmt(fs):>10}{fmt(bs):>11}{fmt(st):>10}{win:>14}")
    print("-" * 92)
    if n:
        print(f"{'MEAN':<8}{fgt/n:>10.5f}{ncgt/n:>11.5f}{(fgt-ncgt)/n:>10.5f}")
        print(f"\nfull wins (lower →GT): {fwins}/{n}   no_crop wins: {ncwins}/{n}")
    print("=" * 92)


if __name__ == "__main__":
    main()

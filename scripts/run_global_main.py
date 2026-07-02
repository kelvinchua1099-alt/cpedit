"""
run_global_main.py — run the full pipeline with crop.main_direction=global
(SplitFlow main axis = global vΔ instead of largest crop) on a few uids, and
compare DINO-I / CD->src against the existing `full` (largest-crop) results.
"""
import json, os, sys
import nvdiffrast.torch as dr
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scripts.run import load_config, run_single
from scripts.eval_correct import render_views, Dino
from src.trellis_wrapper import TrellisWrapper
from src.vs3d import VS3DEditor
from src.pipeline import EditPipeline

OUT = "outputs/batch_gm"
CFG_NAME = "full_global_main"


def main():
    uids = sys.argv[1:] or ["uid_1", "uid_2", "uid_3", "uid_4", "uid_5"]
    os.makedirs(OUT, exist_ok=True)

    cfg0 = load_config(CFG_NAME, ["model.trellis.resolution=512"])
    print(f"[gm] loading backbone once (res=512) ...", flush=True)
    wrapper = TrellisWrapper.from_config(cfg0)
    glctx = dr.RasterizeCudaContext()
    dino = Dino()

    for uid in uids:
        d = f"data/nano3d/{uid}"
        src, tgt = f"{d}/source.png", f"{d}/edit_512.png"
        gt, sm = f"{d}/tar_mesh.glb", f"{d}/src_mesh.glb"
        edit_img = Image.open(tgt)

        cfg = load_config(CFG_NAME, ["model.trellis.resolution=512"])
        editor = VS3DEditor.from_config(wrapper, cfg)
        assert editor.hp.crop_main_global, "main_direction=global not picked up!"
        pipeline = EditPipeline(wrapper, editor, cfg)
        out_dir = os.path.join(OUT, uid)
        print(f"\n===== [{uid} / global-main] =====", flush=True)
        try:
            result = run_single(cfg, out_dir, src=src, tgt=tgt,
                                gt_mesh=gt, src_mesh=sm, pipeline=pipeline)
            glb = os.path.join(out_dir, "result_00.glb")
            dino_scores = dino.dino_i_multi(render_views(glb, glctx), edit_img) if os.path.exists(glb) else {}
            m = result.get("meta", {})
            row = {
                "uid": uid,
                "dino": dino_scores,
                "cd_src": result.get("metrics", {}).get("chamfer_to_src"),
                "n_crops": (m.get("crop") or {}).get("n_crops", 0),
                "voxels": m.get("n_voxels_tgt"),
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            row = {"uid": uid, "error": f"{type(e).__name__}: {e}"}
        json.dump(row, open(os.path.join(OUT, f"{uid}.json"), "w"), indent=2)
        print(f"### GM_DONE {uid} {row.get('dino')}", flush=True)

    # comparison vs existing full
    print("\n\n======= global-main  vs  full (largest-crop) =======")
    print(f"{'uid':<8}{'GM D2':>8}{'full D2':>9}{'ΔD2':>8} | {'GM D3':>8}{'full D3':>9} | {'GM CD':>8}{'full CD':>9}")
    for uid in uids:
        gm = json.load(open(os.path.join(OUT, f"{uid}.json")))
        fp = f"outputs/batch3/{uid}.json"
        fu = (json.load(open(fp)).get("full") or {}) if os.path.exists(fp) else {}
        if "error" in gm or "error" in fu:
            print(f"{uid:<8} (error)"); continue
        g2 = gm["dino"].get("dinov2"); f2 = (fu.get("dino") or {}).get("dinov2")
        g3 = gm["dino"].get("dinov3"); f3 = (fu.get("dino") or {}).get("dinov3")
        gcd = gm.get("cd_src"); fcd = fu.get("cd_src")
        print(f"{uid:<8}{g2:>8.4f}{f2:>9.4f}{g2-f2:>+8.4f} | {g3:>8.4f}{f3:>9.4f} | {gcd:>8.4f}{fcd:>9.4f}")
    print("### GM BATCH DONE")


if __name__ == "__main__":
    main()

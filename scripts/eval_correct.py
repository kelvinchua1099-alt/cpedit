"""
eval_correct.py — Nano3D-style evaluation:
  • DINO-I  (↑): DINO cosine between MULTI-VIEW renders of the edited 3D and the
                 2D edit image (edit_512.png). Measures edit realization.
  • CD->src (↓): Chamfer to the SOURCE mesh. Measures identity preservation.
  • baseline: the SOURCE mesh's own DINO-I vs the edit image ("do nothing").

DINO-I is reported as the MAX cosine over rendered views (best-matching view,
robust to orientation), which is how well ANY view matches the target edit image.
"""
import sys, os, json
import numpy as np
import torch
import nvdiffrast.torch as dr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.render_grid import load_mesh, vnorm, mvp, DEV
from src.eval import chamfer_distance
from transformers import AutoImageProcessor, AutoModel

AZIMS = [0, 45, 90, 135, 180, 225, 270, 315]
EL = 15
RES = 448


def render_views(path, glctx):
    v, f, uv, tex = load_mesh(path)
    vn = vnorm(v, f)
    out = []
    for az in AZIMS:
        m = mvp(az, EL)
        clip = (torch.cat([v, torch.ones_like(v[:, :1])], 1) @ m.T)[None]
        rast, _ = dr.rasterize(glctx, clip, f, resolution=[RES, RES])
        nz = torch.nn.functional.normalize(dr.interpolate(vn[None], rast, f)[0], dim=-1)[..., 2].clamp(0, 1)
        if uv is not None:
            col = dr.texture(tex, dr.interpolate(uv[None], rast, f)[0], filter_mode="linear")
        else:
            col = torch.ones(1, RES, RES, 3, device=DEV) * 0.6
        shade = (0.45 + 0.55 * nz)[..., None]
        col = col * shade * 0.75 + shade * 0.25
        col = dr.antialias(col.contiguous(), rast, clip, f)
        mask = (rast[..., 3:] > 0).float()
        col = col * mask + (1 - mask)
        out.append(Image.fromarray((col[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)))
    return out


_DINO_MODELS = {
    "dinov2": "facebook/dinov2-base",
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


class Dino:
    """DINO-I judge(s).  dinov2 = independent judge; dinov3 = matches the pipeline's
    conditioning encoder (model-view, optimistic/circular).  Reports both."""

    def __init__(self, which=("dinov2", "dinov3")):
        # Kept on CPU by default; moved to GPU only during evaluation so the
        # ~1.5 GB of DINO weights don't steal VRAM from the pipeline / to_glb.
        self.enc = {}
        for k in which:
            try:
                proc = AutoImageProcessor.from_pretrained(_DINO_MODELS[k])
                model = AutoModel.from_pretrained(_DINO_MODELS[k]).eval()
                self.enc[k] = (proc, model)
            except Exception as ex:
                print(f"[dino] skip {k}: {ex}")

    @torch.no_grad()
    def _emb(self, k, pil):
        proc, model = self.enc[k]
        inp = proc(images=pil.convert("RGB"), return_tensors="pt").to(DEV)
        cls = model(**inp).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1)

    def dino_i_multi(self, views, edit_img):
        """Return {encoder_name: max cosine over views}. Moves each encoder to
        GPU just for its pass, then back to CPU to free VRAM."""
        out = {}
        for k, (proc, model) in self.enc.items():
            model.to(DEV)
            try:
                e = self._emb(k, edit_img)
                out[k] = max(float((self._emb(k, v) * e).sum()) for v in views)
            finally:
                model.cpu()
                torch.cuda.empty_cache()
        return out

    def dino_i(self, views, edit_img):   # backward-compat (DINOv2 max, mean)
        k = "dinov2" if "dinov2" in self.enc else next(iter(self.enc))
        _, model = self.enc[k]
        model.to(DEV)
        try:
            e = self._emb(k, edit_img)
            sims = [float((self._emb(k, v) * e).sum()) for v in views]
        finally:
            model.cpu(); torch.cuda.empty_cache()
        return max(sims), float(np.mean(sims))


def cd_to_src(glb, src_mesh):
    if not (glb and src_mesh and os.path.exists(glb) and os.path.exists(src_mesh)):
        return None
    return chamfer_distance(glb, src_mesh)


def main():
    uids = sys.argv[1:] or ["uid_1", "uid_2", "uid_3", "uid_4", "uid_5"]
    glctx = dr.RasterizeCudaContext()
    dino = Dino()
    results = {}
    for uid in uids:
        edit = Image.open(f"data/nano3d/{uid}/edit_512.png")
        srcm = f"data/nano3d/{uid}/src_mesh.glb"
        row = {}
        # baseline: source mesh vs edit image
        base_views = render_views(srcm, glctx)
        row["source_dinoI_max"], row["source_dinoI_mean"] = dino.dino_i(base_views, edit)
        for cfg in ["full", "no_crop"]:
            glb = f"outputs/batch/{cfg}/{uid}/result_00.glb"
            if not os.path.exists(glb):
                row[cfg] = {"error": "missing"}; continue
            views = render_views(glb, glctx)
            dmax, dmean = dino.dino_i(views, edit)
            row[cfg] = {"dinoI_max": dmax, "dinoI_mean": dmean, "cd_to_src": cd_to_src(glb, srcm)}
        results[uid] = row
        print(f"[{uid}] done", flush=True)
    json.dump(results, open("outputs/eval_correct.json", "w"), indent=2)
    _table(results)


def _table(R):
    print("\n" + "=" * 96)
    print("DINO-I (max over views) ↑ = edit realization vs edit_512   |   CD->src ↓ = preservation")
    print(f"{'uid':<7}{'src(baseline)':>14}{'full DINO-I':>13}{'noc DINO-I':>12}{'full CD-src':>13}{'noc CD-src':>12}{'  edit-winner':>14}")
    print("-" * 96)
    for uid, r in R.items():
        f = r.get("full", {}); nc = r.get("no_crop", {})
        def F(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "  -  "
        fd, nd = f.get("dinoI_max"), nc.get("dinoI_max")
        win = "-"
        if isinstance(fd, float) and isinstance(nd, float):
            win = "full" if fd > nd else "no_crop"
        print(f"{uid:<7}{F(r.get('source_dinoI_max')):>14}{F(fd):>13}{F(nd):>12}"
              f"{F(f.get('cd_to_src')):>13}{F(nc.get('cd_to_src')):>12}{win:>14}")
    print("=" * 96)


if __name__ == "__main__":
    main()

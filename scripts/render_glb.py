"""
render_glb.py — headless TEXTURED renders of GLB meshes via nvdiffrast's CUDA
rasterizer (no GL context / X display needed).

Usage:
  python scripts/render_glb.py out.png labelA=a.glb labelB=b.glb ...
Rows = meshes, cols = views.  Samples the GLB's baked texture (albedo) and
mixes a little normal shading so dark meshes stay legible.  Orientation is
corrected with a 180° flip about X (TRELLIS/o_voxel GLBs come in upside-down
relative to a Y-up camera).
"""
import sys
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw

DEV = "cuda"
H = W = 340
VIEWS = [20.0, 140.0, 260.0]     # azimuths (deg)
ELEV = 15.0
FLIP = torch.tensor([1.0, -1.0, -1.0], device=DEV)   # Rx(180): fix upside-down


def load_mesh(path):
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values() if hasattr(g, "vertices")])
    v = torch.tensor(np.asarray(m.vertices, np.float32), device=DEV) * FLIP
    f = torch.tensor(np.asarray(m.faces, np.int32), device=DEV)
    v = v - (v.amax(0) + v.amin(0)) / 2.0
    v = v / (v.norm(dim=1).max() + 1e-8)

    # texture + uv (fall back to a mid-gray albedo)
    uv = tex = None
    vis = getattr(m, "visual", None)
    try:
        if vis is not None and getattr(vis, "uv", None) is not None:
            uv = torch.tensor(np.asarray(vis.uv, np.float32), device=DEV)
            img = vis.material.baseColorTexture
            if img is not None:
                a = np.asarray(img.convert("RGB"), np.float32) / 255.0
                tex = torch.tensor(a, device=DEV)[None]
    except Exception:
        uv = tex = None
    if uv is None or tex is None:
        # try per-vertex color, else constant gray
        try:
            vc = np.asarray(vis.vertex_colors, np.float32)[:, :3] / 255.0
            return v, f, ("vc", torch.tensor(vc, device=DEV))
        except Exception:
            return v, f, ("const", torch.tensor([0.6, 0.6, 0.6], device=DEV))
    return v, f, ("tex", uv, tex)


def vertex_normals(v, f):
    vn = torch.zeros_like(v)
    tri = v[f]
    fn = torch.nn.functional.normalize(
        torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1), dim=1)
    for k in range(3):
        vn.index_add_(0, f[:, k].long(), fn)
    return torch.nn.functional.normalize(vn, dim=1)


def look_at(az, el, dist=2.6):
    a, e = np.radians(az), np.radians(el)
    eye = np.array([dist*np.cos(e)*np.sin(a), dist*np.sin(e), dist*np.cos(e)*np.cos(a)], np.float32)
    f = -eye / np.linalg.norm(eye); up = np.array([0, 1, 0], np.float32)
    s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    V = np.eye(4, dtype=np.float32); V[0, :3], V[1, :3], V[2, :3] = s, u, -f; V[:3, 3] = -V[:3, :3] @ eye
    fov, near, far = np.radians(40), 0.1, 10.0; t = np.tan(fov/2)
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 1/t; P[1, 1] = 1/t; P[2, 2] = -(far+near)/(far-near)
    P[2, 3] = -2*far*near/(far-near); P[3, 2] = -1
    return torch.tensor(P @ V, device=DEV)


def render(v, f, albedo, glctx):
    vn = vertex_normals(v, f)
    out = []
    for az in VIEWS:
        mvp = look_at(az, ELEV)
        clip = (torch.cat([v, torch.ones_like(v[:, :1])], 1) @ mvp.T)[None]
        rast, _ = dr.rasterize(glctx, clip, f, resolution=[H, W])
        nrm, _ = dr.interpolate(vn[None], rast, f)
        ndotl = torch.nn.functional.normalize(nrm, dim=-1)[..., 2].clamp(0, 1)
        if albedo[0] == "tex":
            uv, tex = albedo[1], albedo[2]
            uvi, _ = dr.interpolate(uv[None], rast, f)
            col = dr.texture(tex, uvi, filter_mode="linear")
        elif albedo[0] == "vc":
            col, _ = dr.interpolate(albedo[1][None], rast, f)
        else:
            col = albedo[1].view(1, 1, 1, 3).expand(1, H, W, 3)
        shade = (0.45 + 0.55 * ndotl)[..., None]
        # mix 25% normal-gray so near-black albedo still shows geometry
        col = col * shade * 0.75 + shade * 0.25
        col = dr.antialias(col.contiguous(), rast, clip, f)
        mask = (rast[..., 3:] > 0).float()
        col = col * mask + (1 - mask)
        out.append((col[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return out


def main():
    out = sys.argv[1]
    items = [a.split("=", 1) for a in sys.argv[2:]]
    glctx = dr.RasterizeCudaContext()
    rows = []
    for label, path in items:
        v, f, albedo = load_mesh(path)
        rows.append((label, np.concatenate(render(v, f, albedo, glctx), axis=1)))
    im = Image.fromarray(np.concatenate([r for _, r in rows], axis=0))
    d = ImageDraw.Draw(im)
    for i, (label, _) in enumerate(rows):
        d.text((6, i * H + 6), label, fill=(220, 30, 30))
    im.save(out)
    print("saved", out, im.size)


if __name__ == "__main__":
    main()
